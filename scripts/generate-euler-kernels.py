#!/usr/bin/env python3
"""Emit concrete Rust entry names from the shared family declarations."""

import argparse
from pathlib import Path
import re
import sys


def draws(declaration):
    """Mirror `Family::draws`: identifiers read by the clauses after the noise list."""
    code = re.sub(r"//[^\n]*", "", declaration)
    noise = re.search(r"\bnoise\s*\([^)]*\)", code)
    if noise is None:
        raise ValueError("a family declaration has no noise list")
    clauses = code[noise.end() :]

    def reads(name):
        return re.search(rf"(?<![A-Za-z0-9_$]){name}(?![A-Za-z0-9_])", clauses.replace("$", "")) is not None

    gamma = reads("gm") or reads("gm2")
    return gamma or reads("nj") or reads("js"), gamma


def declarations(directory):
    headers = []
    for source in sorted(directory.glob("*.rs")):
        text = source.read_text(encoding="utf-8")
        matches = list(re.finditer(r"^\s*(\d+)\s*=>\s*([A-Za-z_]\w*)\s*\{", text, re.M))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            headers.append((int(match.group(1)), match.group(2), draws(text[match.end() : end])))
    headers.sort()
    if not headers or [code for code, _, _ in headers] != list(range(len(headers))):
        raise ValueError("family codes must be unique, dense and start at zero")
    if len({name for _, name, _ in headers}) != len(headers):
        raise ValueError("family names must be unique")
    return headers


def sources(headers):
    files = {}
    modules = []
    for start in range(0, len(headers), 32):
        module = f"families_{start}_{min(start + 31, len(headers) - 1)}"
        modules.extend([f"mod {module};", f"pub use {module}::*;"])
        lines = []
        for code, family, (reads_jumps, reads_gamma) in headers[start : start + 32]:
            for jumps in (False, True)[: 1 + reads_jumps]:
                for gamma in (False, True)[: 1 + reads_gamma]:
                    lines.append(
                        f"entrypoint!(euler_{code}_{int(jumps)}_{int(gamma)}, "
                        f"{family}, {str(jumps).lower()}, {str(gamma).lower()});"
                    )
        files[f"{module}.rs"] = "\n".join(lines) + "\n"
    files["mod.rs"] = "\n".join(modules) + "\n"
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    definitions = root / "stochastic-rs-stochastic/src/euler/families/definitions"
    destination = root / "stochastic-rs-metal-kernels/src/euler/entries"
    expected = sources(declarations(definitions))
    actual = {path.name for path in destination.glob("*.rs")}
    stale = actual - expected.keys()
    changed = [
        name
        for name, contents in expected.items()
        if not (destination / name).exists()
        or (destination / name).read_text(encoding="utf-8") != contents
    ]
    if arguments.check:
        if stale or changed:
            print("Euler entries are stale; run scripts/generate-euler-kernels.py", file=sys.stderr)
            return 1
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name in stale:
            (destination / name).unlink()
        for name in changed:
            (destination / name).write_text(expected[name], encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
