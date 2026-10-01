"""
tests/test_spsa.py
==================
The SPSA optimiser (#316): its gradient estimate, its convergence, its cost in
loss evaluations, and training a shot-based hybrid model with it.
"""

from __future__ import annotations

import math
from typing import Any

import pennylane as qml
import pytest
import torch

from hqnn_forge.models import HybridBinaryClassifier
from hqnn_forge.training import SPSA, train_model

Z = 5.0


def _quadratic(d: int, seed: int = 0) -> tuple[torch.nn.Parameter, torch.Tensor, Any]:
    g = torch.Generator().manual_seed(seed)
    theta = torch.nn.Parameter(torch.randn(d, generator=g, dtype=torch.float64))
    target = torch.randn(d, generator=g, dtype=torch.float64)
    scales = torch.linspace(0.5, 2.0, d, dtype=torch.float64)

    def loss() -> torch.Tensor:
        return 0.5 * (scales * (theta - target) ** 2).sum()

    return theta, target, loss


class TestGradientEstimate:
    def test_unbiased_on_a_quadratic(self) -> None:
        # A central difference is exact for a quadratic along any direction, and
        # E[Δ_i Δ_j] = δ_ij, so E[ĝ] is the gradient exactly.
        theta, target, loss = _quadratic(6)
        opt = SPSA([theta], perturbation=0.3)
        exact = torch.linspace(0.5, 2.0, 6, dtype=torch.float64) * (theta.detach() - target)
        draws = torch.stack([opt.gradient_estimate(loss)[0] for _ in range(4000)])
        se = draws.std(0) / math.sqrt(draws.shape[0])
        assert ((draws.mean(0) - exact).abs() <= Z * se).all()

    def test_leaves_the_parameters_where_they_were(self) -> None:
        theta, _, loss = _quadratic(4)
        before = theta.detach().clone()
        SPSA([theta]).gradient_estimate(loss)
        torch.testing.assert_close(theta.detach(), before, rtol=0, atol=0)

    def test_directions_are_plus_or_minus_one(self) -> None:
        theta, _, _ = _quadratic(50)
        opt = SPSA([theta], perturbation=1.0)
        # L = θ_0 is linear, so ĝ_i = (L+ − L−)/(2c) · Δ_i = Δ_0 Δ_i: every
        # entry is ±1, and entry 0 is Δ_0² = 1.
        (g,) = opt.gradient_estimate(lambda: theta[0])
        assert set(g.tolist()) <= {-1.0, 1.0} and g[0] == 1.0


class TestSteps:
    def test_converges_on_a_quadratic(self) -> None:
        theta, target, loss = _quadratic(10)
        opt = SPSA([theta], lr=0.5, perturbation=0.2, stability=20)
        start = float(loss().detach())
        for _ in range(600):
            opt.step(loss)
        assert float(loss().detach()) < 1e-3 * start
        assert (theta.detach() - target).abs().max() < 0.05

    @pytest.mark.parametrize("d", [3, 300])
    def test_two_loss_evaluations_per_step_whatever_the_size(self, d: int) -> None:
        theta, _, loss = _quadratic(d)
        calls = {"n": 0}

        def counted() -> torch.Tensor:
            calls["n"] += 1
            return loss()

        opt = SPSA([theta])
        for _ in range(7):
            opt.step(counted)
        assert calls["n"] == 14

    def test_returns_the_mean_of_the_two_evaluations(self) -> None:
        values = iter([3.0, 1.0])
        theta = torch.nn.Parameter(torch.zeros(2))
        assert SPSA([theta]).step(lambda: next(values)) == 2.0

    def test_gains_follow_spall_s_schedules(self) -> None:
        # With L = w·θ, ĝ = (w·Δ)Δ and the step is exactly a_k (w·Δ)Δ, so
        # the step length reveals a_k; c_k cancels for a linear loss.
        theta = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        opt = SPSA([theta], lr=0.8, alpha=0.602, stability=3.0)
        for k in range(5):
            before = theta.item()
            opt.step(lambda: 2.0 * theta.sum())
            assert abs(before - theta.item()) == pytest.approx(0.8 / (k + 1 + 3.0) ** 0.602 * 2.0)

    def test_parameter_groups_take_their_own_lr(self) -> None:
        a = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        opt = SPSA([{"params": [a], "lr": 1.0}, {"params": [b], "lr": 0.1}])
        opt.step(lambda: a.sum() + b.sum())
        assert abs(a.item()) == pytest.approx(10 * abs(b.item()))

    def test_reproducible_with_a_generator(self) -> None:
        results = []
        for _ in range(2):
            theta, _, loss = _quadratic(5)
            opt = SPSA([theta], generator=torch.Generator().manual_seed(7))
            for _ in range(10):
                opt.step(loss)
            results.append(theta.detach().clone())
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_frozen_parameters_are_left_alone(self) -> None:
        theta, _, loss = _quadratic(3)
        frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        SPSA([theta, frozen]).step(loss)
        torch.testing.assert_close(frozen.detach(), torch.ones(2), rtol=0, atol=0)


class TestCommonRandomNumbers:
    def test_both_evaluations_see_the_same_torch_draws(self) -> None:
        # A loss with dropout-like noise from the global RNG: with the same
        # draws on both sides, the difference is exactly the deterministic
        # part's, so ĝ = (w·Δ)Δ with no noise term.
        theta = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
        w = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)

        def noisy() -> torch.Tensor:
            return (w * theta).sum() + 100 * torch.rand(1, dtype=torch.float64).sum()

        opt = SPSA([theta], perturbation=0.1)
        for _ in range(20):
            (g,) = opt.gradient_estimate(noisy)
            delta = g / g.abs().max()  # the ±1 direction, up to sign
            torch.testing.assert_close(g, (w * delta).sum() * delta, rtol=1e-9, atol=1e-9)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"lr": 0.0}, "lr must be > 0"),
            ({"perturbation": -1.0}, "perturbation must be > 0"),
            ({"alpha": 0.0}, "alpha and gamma"),
            ({"stability": -1.0}, "stability ≥ 0"),
        ],
    )
    def test_bad_settings(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SPSA([torch.nn.Parameter(torch.zeros(1))], **kwargs)

    def test_step_needs_a_closure(self) -> None:
        with pytest.raises(ValueError, match="needs a closure"):
            SPSA([torch.nn.Parameter(torch.zeros(1))]).step()


def _data(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, 4, generator=g)
    return x, (x[:, 0] - x[:, 1] > 0).float()


class TestTrainModel:
    def test_trains_a_shot_based_model_without_backward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        model = HybridBinaryClassifier(
            n_input_features=4,
            n_qubits=2,
            n_layers=1,
            device_name="default.qubit",
            diff_method="parameter-shift",
            shots=2000,
        )
        x, y = _data()
        forward = {"n": 0}
        original = model.forward

        def counted(inp: torch.Tensor) -> torch.Tensor:
            forward["n"] += 1
            return original(inp)

        monkeypatch.setattr(model, "forward", counted)

        def exact_loss() -> float:
            # The same weights, read out exactly, to judge progress without
            # the shot noise of the model itself.
            ref = HybridBinaryClassifier(
                n_input_features=4,
                n_qubits=2,
                n_layers=1,
                device_name="default.qubit",
                diff_method="backprop",
            )
            ref.load_state_dict(model.state_dict())
            with torch.no_grad():
                return float(torch.nn.BCEWithLogitsLoss()(ref.eval()(x).squeeze(-1), y))

        before = exact_loss()
        opt = SPSA(model.parameters(), lr=1.0, perturbation=0.1, stability=20)
        history = train_model(
            model, torch.nn.BCEWithLogitsLoss(), opt, x, y, max_epochs=40, batch_size=32
        )
        assert history.n_epochs == 40
        # One batch per epoch, two forward passes per SPSA step, and no gradient.
        assert forward["n"] == 2 * 40
        assert all(p.grad is None for p in model.parameters())
        # Measured: 0.627 -> 0.502 (a factor 0.80) with 2000 shots.
        assert exact_loss() < 0.9 * before

    def test_reaches_a_loss_comparable_to_adam_with_exact_gradients(self) -> None:
        # Full batch of 64, the same model and seed.  Measured: Adam (lr 0.05)
        # reaches 0.346 in 60 steps; SPSA (lr 1.0, c 0.1, A 50) 0.524 in 500
        # and 0.487 in 1000.  Per step SPSA is far slower; see the module
        # docstring for when it is cheaper in circuit evaluations.
        x, y = _data(64)
        losses = {}
        for name, steps in (("adam", 60), ("spsa", 1000)):
            torch.manual_seed(0)
            model = HybridBinaryClassifier(
                n_input_features=4,
                n_qubits=2,
                n_layers=1,
                device_name="default.qubit",
                diff_method="backprop",
            )
            opt: torch.optim.Optimizer = (
                SPSA(model.parameters(), lr=1.0, perturbation=0.1, stability=50)
                if name == "spsa"
                else torch.optim.Adam(model.parameters(), lr=0.05)
            )
            history = train_model(
                model, torch.nn.BCEWithLogitsLoss(), opt, x, y, max_epochs=steps, batch_size=64
            )
            losses[name] = history.train_loss[-1]
        assert losses["spsa"] < 1.6 * losses["adam"]

    def test_adam_on_the_head_beats_spsa_on_everything(self) -> None:
        # #316's split: SPSA on the encoder and circuit, Adam on the head from
        # the same two evaluations.  Measured at the final weights, seed 0:
        # SPSA on all 19 parameters 0.473 after 1000 steps, the split 0.156,
        # at the same two circuit evaluations per sample per step.
        x, y = _data(64)
        losses = {}
        for name in ("spsa", "split"):
            torch.manual_seed(0)
            model = HybridBinaryClassifier(
                n_input_features=4,
                n_qubits=2,
                n_layers=1,
                device_name="default.qubit",
                diff_method="backprop",
            )
            head = list(model.head.parameters())
            rest = [p for n, p in model.named_parameters() if not n.startswith("head.")]
            opt = (
                SPSA(
                    rest,
                    lr=1.0,
                    perturbation=0.1,
                    stability=50,
                    gradient_optimizer=torch.optim.Adam(head, lr=0.05),
                )
                if name == "split"
                else SPSA(model.parameters(), lr=1.0, perturbation=0.1, stability=50)
            )
            train_model(
                model, torch.nn.BCEWithLogitsLoss(), opt, x, y, max_epochs=1000, batch_size=64
            )
            with torch.no_grad():
                losses[name] = float(torch.nn.BCEWithLogitsLoss()(model(x).squeeze(-1), y))
        assert losses["split"] < 0.5 * losses["spsa"]

    def test_the_head_split_costs_no_extra_circuit_executions_with_shots(self) -> None:
        # Backpropagating to the head alone must not run the parameter-shift
        # backward pass: the executions are those of SPSA on everything,
        # 2 per sample per step.  Measured: 192 both ways.
        x, y = _data()
        counts = {}
        for split in (False, True):
            torch.manual_seed(0)
            model = HybridBinaryClassifier(
                n_input_features=4,
                n_qubits=2,
                n_layers=1,
                device_name="default.qubit",
                diff_method="parameter-shift",
                shots=2000,
            )
            head = list(model.head.parameters())
            head_before = [p.detach().clone() for p in head]
            rest = [p for n, p in model.named_parameters() if not n.startswith("head.")]
            opt = (
                SPSA(rest, gradient_optimizer=torch.optim.Adam(head, lr=0.05))
                if split
                else SPSA(model.parameters())
            )
            with qml.Tracker(model.quantum_layer.qlayer.qnode.device) as tracker:
                train_model(
                    model, torch.nn.BCEWithLogitsLoss(), opt, x, y, max_epochs=3, batch_size=32
                )
            counts[split] = tracker.totals["executions"]
            assert all(p.grad is None for p in rest)
            if split:
                assert all(p.grad is not None for p in head)
                assert all(not torch.equal(p, b) for p, b in zip(head, head_before, strict=True))
        assert counts[True] == counts[False] == 2 * 32 * 3


class TestGradientOptimizer:
    def test_the_head_gets_the_mean_gradient_of_the_two_evaluations(self) -> None:
        # L = (w·θ)(v·h): the gradient in h at θ ± cΔ is (w·θ ± c w·Δ) v, whose
        # mean is the gradient at θ, (w·θ) v, exactly.  SGD with lr 1 moves h
        # by minus that.
        theta = torch.nn.Parameter(torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64))
        h = torch.nn.Parameter(torch.tensor([0.5, 2.0], dtype=torch.float64))
        w = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
        v = torch.tensor([-3.0, 0.5], dtype=torch.float64)
        expected = h.detach() - (w @ theta.detach()) * v
        opt = SPSA([theta], perturbation=0.4, gradient_optimizer=torch.optim.SGD([h], lr=1.0))
        opt.step(lambda: (w @ theta) * (v @ h))
        torch.testing.assert_close(h.detach(), expected, rtol=1e-12, atol=1e-12)

    def test_the_spsa_step_is_unchanged_and_still_two_evaluations(self) -> None:
        # h enters additively and is not perturbed, so L+ − L− and with it
        # the SPSA step are exactly those of SPSA without the split.
        results = []
        for split in (False, True):
            theta, _, quadratic = _quadratic(4)
            h = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            calls = {"n": 0}

            def loss(
                q: Any = quadratic, h: torch.nn.Parameter = h, calls: dict[str, int] = calls
            ) -> torch.Tensor:
                calls["n"] += 1
                return q() + (h**2).sum()

            opt = SPSA([theta], gradient_optimizer=torch.optim.SGD([h], lr=0.1) if split else None)
            for _ in range(5):
                opt.step(loss)
            assert calls["n"] == 10
            results.append(theta.detach().clone())
            if split:
                # dL/dh = 2h, so each SGD step with lr 0.1 scales h by 0.8.
                torch.testing.assert_close(
                    h.detach(), torch.full((2,), 0.8**5, dtype=torch.float64)
                )
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_gradient_estimate_ignores_it(self) -> None:
        theta, _, loss = _quadratic(3)
        h = torch.nn.Parameter(torch.ones(1))
        SPSA([theta], gradient_optimizer=torch.optim.SGD([h], lr=1.0)).gradient_estimate(loss)
        assert h.grad is None

    def test_shared_parameters_raise(self) -> None:
        theta = torch.nn.Parameter(torch.zeros(2))
        with pytest.raises(ValueError, match="shares parameters"):
            SPSA([theta], gradient_optimizer=torch.optim.Adam([theta]))

    def test_a_gradient_free_one_raises(self) -> None:
        with pytest.raises(ValueError, match="must use gradients"):
            SPSA(
                [torch.nn.Parameter(torch.zeros(1))],
                gradient_optimizer=SPSA([torch.nn.Parameter(torch.zeros(1))]),
            )

    def test_a_float_closure_raises(self) -> None:
        theta = torch.nn.Parameter(torch.zeros(1))
        h = torch.nn.Parameter(torch.zeros(1))
        opt = SPSA([theta], gradient_optimizer=torch.optim.SGD([h], lr=1.0))
        with pytest.raises(ValueError, match="loss tensor"):
            opt.step(lambda: float((theta + h).sum().detach()))


def test_multiclass_labels_are_checked_before_the_first_spsa_step() -> None:
    # train_model checks multiclass labels on the first logits; with SPSA the
    # first logits come from its closure, which runs before any update.
    from hqnn_forge.models import MulticlassHybridClassifier

    torch.manual_seed(0)
    model = MulticlassHybridClassifier(
        n_input_features=4,
        n_qubits=2,
        n_layers=1,
        n_classes=3,
        device_name="default.qubit",
        diff_method="backprop",
    )
    before = [p.detach().clone() for p in model.parameters()]
    opt = SPSA(model.parameters(), lr=1.0, perturbation=0.1)
    x, y = torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 0, 1])
    with pytest.raises(ValueError, match="needs class labels in"):
        train_model(model, torch.nn.CrossEntropyLoss(), opt, x, y, max_epochs=1)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    y_ok = torch.tensor([0, 1, 2, 2, 0, 1])
    history = train_model(model, torch.nn.CrossEntropyLoss(), opt, x, y_ok, max_epochs=1)
    assert history.n_epochs == 1
