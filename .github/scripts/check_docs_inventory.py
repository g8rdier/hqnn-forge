"""Fail unless every public name of every documented package has an API entry.

Run by the docs CI job after ``mkdocs build --strict`` (#322).  For each page
under ``docs/api/`` that renders a package or module (``::: hqnn_forge.x``),
every name in that module's ``__all__`` must appear in the built site's
``objects.inv``: a name exported but missing from the reference is a hole in
the documentation that the strict build does not see.  ``__all__`` is read
statically with griffe, so the check needs neither torch nor PennyLane.
"""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import griffe

DIRECTIVE = re.compile(r"^::: (hqnn_forge(?:\.[A-Za-z_]\w*)*)\s*$", re.MULTILINE)


def inventory(site: Path) -> set[str]:
    raw = (site / "objects.inv").read_bytes()
    body = zlib.decompress(raw.split(b"\n", 4)[4]).decode()
    return {line.split(" ", 1)[0] for line in body.splitlines() if line}


def missing(docs: Path, site: Path) -> list[str]:
    package = griffe.load("hqnn_forge", search_paths=["."])
    documented = inventory(site)
    problems = []
    for page in sorted((docs / "api").glob("*.md")):
        for path in DIRECTIVE.findall(page.read_text()):
            obj = package if path == "hqnn_forge" else package[path[len("hqnn_forge.") :]]
            if not obj.is_module:
                continue
            for name in sorted(obj.exports or ()):
                if f"{path}.{name}" not in documented:
                    problems.append(f"{page.name}: {path}.{name} is exported but not documented")
    return problems


def main() -> int:
    problems = missing(Path("docs"), Path("site"))
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
