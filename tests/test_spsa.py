"""
tests/test_spsa.py
==================
The SPSA optimiser (#316): its gradient estimate, its convergence, its cost in
loss evaluations, and training a shot-based hybrid model with it.
"""

from __future__ import annotations

import math
from typing import Any

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
