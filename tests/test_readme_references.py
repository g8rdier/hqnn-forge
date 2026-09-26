"""
tests/test_readme_references.py
===============================
Every source cited in a module docstring's ``References`` section is listed
in the README's ``References`` section, with the same year (#168).

Only module-level ``References`` blocks are checked; a work mentioned in
passing in prose elsewhere need not be listed.  Citations are compared by
surnames and year, so ``King, G. & Zeng, L. (2001)`` in a docstring matches
``King & Zeng (2001)`` in the README, and the same authors with a different
year is reported as a mismatch rather than as a missing entry.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "hqnn_forge"
README = ROOT / "README.md"

# "* Authors (YYYY)" at the start of a bullet: the authors run up to the year.
_BULLET = re.compile(r"^\s*[*-]\s+(?P<authors>[^()\n]+?)\s*\((?P<year>\d{4})\)", re.MULTILINE)
# A numpy-style section header: a title line underlined with dashes.
_SECTION = re.compile(r"^(?P<title>\S[^\n]*)\n-{3,}\n", re.MULTILINE)
# Initials such as "G.", "T.-Y." following a surname.
_INITIALS = re.compile(r",?\s*\b[A-Z]\.(?:-[A-Z]\.)*,?")


def normalise_authors(authors: str) -> str:
    """``"Lin, T.-Y., et al."`` → ``"Lin et al."``; ``"King, G. & Zeng, L."`` → ``"King & Zeng"``."""
    return " ".join(_INITIALS.sub(" ", authors).replace(",", " ").split())


def citations(text: str) -> list[tuple[str, str]]:
    """``(normalised authors, year)`` for each bullet in ``text``."""
    return [(normalise_authors(m["authors"]), m["year"]) for m in _BULLET.finditer(text)]


def references_block(docstring: str) -> str | None:
    """The body of the ``References`` section, up to the next section header."""
    headers = list(_SECTION.finditer(docstring))
    for i, header in enumerate(headers):
        if header["title"].strip() == "References":
            end = headers[i + 1].start() if i + 1 < len(headers) else len(docstring)
            return docstring[header.end() : end]
    return None


def docstring_citations() -> dict[tuple[str, str], list[str]]:
    """Each cited ``(authors, year)`` with the modules citing it."""
    cited: dict[tuple[str, str], list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)
        block = references_block(docstring or "")
        for citation in citations(block or ""):
            cited.setdefault(citation, []).append(str(path.relative_to(ROOT)))
    return cited


def readme_citations() -> set[tuple[str, str]]:
    text = README.read_text(encoding="utf-8")
    start = text.index("\n## References\n")
    end = text.find("\n## ", start + 1)
    return set(citations(text[start : end if end != -1 else len(text)]))


class TestParser:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Lin, T.-Y., et al.", "Lin et al."),
            ("King, G. & Zeng, L.", "King & Zeng"),
            ("Schuld, Sweke & Meyer", "Schuld Sweke & Meyer"),
            ("Pérez-Salinas et al.", "Pérez-Salinas et al."),
            ("Wilcoxon", "Wilcoxon"),
        ],
    )
    def test_normalise_authors(self, raw: str, expected: str) -> None:
        assert normalise_authors(raw) == expected

    def test_block_stops_at_the_next_section(self) -> None:
        doc = (
            "Intro\n\nReferences\n----------\n* A et al. (2001) x\n\nNotes\n-----\n* B (2002) y\n"
        )
        assert citations(references_block(doc) or "") == [("A et al.", "2001")]

    def test_readme_bullets_parse(self) -> None:
        text = "- Jones & Gacon (2020) — *Efficient calculation*\n- Wilcoxon (1945) — *x*\n"
        assert citations(text) == [("Jones & Gacon", "2020"), ("Wilcoxon", "1945")]


class TestReadmeListsDocstringSources:
    def test_the_scan_finds_the_citations(self) -> None:
        """Guard against a parser that finds nothing and so passes vacuously."""
        cited = docstring_citations()
        assert len(cited) >= 20, sorted(cited)
        assert ("King & Zeng", "2001") in cited
        assert ("Lin et al.", "2017") in cited

    def test_every_docstring_citation_is_in_the_readme(self) -> None:
        listed = readme_citations()
        years: dict[str, list[str]] = {}
        for authors, year in listed:
            years.setdefault(authors, []).append(year)
        problems = []
        for (authors, year), modules in sorted(docstring_citations().items()):
            if (authors, year) in listed:
                continue
            where = ", ".join(modules)
            if authors in years:
                problems.append(
                    f"{authors} ({year}) in {where}: README dates it "
                    f"({', '.join(sorted(years[authors]))})"
                )
            else:
                problems.append(f"{authors} ({year}) in {where}: missing from README References")
        assert not problems, "\n".join(problems)
