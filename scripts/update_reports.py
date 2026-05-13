"""
Post-processing script: parse compare_accuracy.txt and update all reports.

Run after `make eval-compare` completes:
    uv run python scripts/update_reports.py

Updates:
  - README.md           accuracy table C column
  - CHANGELOG.md        [Unreleased] section results
  - docs/report.md      Phase 5 result tables
  - docs/demo_report.md accuracy tables
  - docs/figures/       regenerate all charts
"""

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] update_reports: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "results"


# ── parse results ─────────────────────────────────────────────────────────────

@dataclass
class CompareResults:
    acc_a: float
    acc_b: float
    acc_c: float
    a_pct: str
    b_pct: str
    c_pct: str
    b_correct: int
    b_total: int
    c_correct: int
    c_total: int
    per_type: dict  # {type: {"b": float, "c": float, "n": int}}


def parse_results(path: Path) -> CompareResults:
    text = path.read_text()
    log.info("Parsing %s", path)

    def _pct(label: str) -> tuple[float, str]:
        m = re.search(rf"{label}.*?([\d.]+)\s*\(([\d.]+)%\)", text)
        if not m:
            raise ValueError(f"Cannot find {label} in results file")
        return float(m.group(1)), f"{m.group(2)}%"

    def _counts(label: str) -> tuple[int, int]:
        m = re.search(rf"{label}.*?\[(\d+)/(\d+)\]", text)
        if not m:
            return 0, 0
        return int(m.group(1)), int(m.group(2))

    acc_a, a_pct = _pct("Group A")
    acc_b, b_pct = _pct("Group B")
    acc_c, c_pct = _pct("Group C")
    b_correct, b_total = _counts("Group B")
    c_correct, c_total = _counts("Group C")

    per_type = {}
    for m in re.finditer(r"(\w+)\s*:\s*B=([\d.]+)\s+C=([\d.]+)\s+\(n=(\d+)\)", text):
        per_type[m.group(1)] = {
            "b": float(m.group(2)),
            "c": float(m.group(3)),
            "n": int(m.group(4)),
        }

    log.info("A=%.2f%%  B=%.2f%%  C=%.2f%%", acc_a * 100, acc_b * 100, acc_c * 100)
    return CompareResults(
        acc_a=acc_a, acc_b=acc_b, acc_c=acc_c,
        a_pct=a_pct, b_pct=b_pct, c_pct=c_pct,
        b_correct=b_correct, b_total=b_total,
        c_correct=c_correct, c_total=c_total,
        per_type=per_type,
    )


# ── file helpers ──────────────────────────────────────────────────────────────

def replace_once(path: Path, old: str, new: str, label: str = "") -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        log.warning("Pattern not found in %s: %s", path.name, label or repr(old[:60]))
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    log.info("Updated %s: %s", path.name, label)
    return True


def replace_re(path: Path, pattern: str, repl: str, label: str = "") -> bool:
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, repl, text)
    if n == 0:
        log.warning("Pattern not found in %s: %s", path.name, label or pattern[:60])
        return False
    path.write_text(new_text, encoding="utf-8")
    log.info("Updated %s (%d replacements): %s", path.name, n, label)
    return True


# ── update README.md ──────────────────────────────────────────────────────────

def update_readme(r: CompareResults) -> None:
    path = ROOT / "README.md"
    log.info("Updating README.md ...")

    b_pct = f"{r.acc_b * 100:.2f}%"
    c_pct = f"{r.acc_c * 100:.2f}%"
    delta_ab = (r.acc_b - r.acc_a) * 100
    delta_bc = (r.acc_c - r.acc_b) * 100

    # Replace placeholder ≈89.11% in C column if present
    replace_re(path,
               r"≈89\.11%",
               c_pct,
               "C column placeholder")

    # Update any existing per-type table sunglasses/cup/glasses rows
    for occ, info in r.per_type.items():
        b_v = f"{info['b'] * 100:.2f}%"
        c_v = f"{info['c'] * 100:.2f}%"
        # Replace old C values that are clearly wrong (12.52% era)
        replace_re(path,
                   rf"(\|\s*{occ}[^\|]*\|[^\|]*\|[^\|]*\|)\s*[\d.]+%(\s*\|)",
                   rf"\g<1> {c_v}\2",
                   f"per-type C for {occ}")

    log.info("README.md done")


# ── update CHANGELOG.md ───────────────────────────────────────────────────────

def update_changelog(r: CompareResults) -> None:
    path = ROOT / "CHANGELOG.md"
    log.info("Updating CHANGELOG.md ...")
    c_pct = f"{r.acc_c * 100:.2f}%"
    b_pct = f"{r.acc_b * 100:.2f}%"

    # Replace "运行中" note with actual result
    replace_re(path,
               r"eval-compare.*?运行中.*",
               f"- **eval-compare v3 完成**：A={r.a_pct}  B={b_pct}  C={c_pct}（全量，22,452 query）",
               "eval-compare running note")

    # Update any C=≈89.11% placeholders
    replace_re(path, r"C≈89\.11%|C=≈89\.11%", f"C={c_pct}", "C placeholder")

    log.info("CHANGELOG.md done")


# ── update docs/report.md ─────────────────────────────────────────────────────

def update_report(r: CompareResults) -> None:
    path = ROOT / "docs" / "report.md"
    log.info("Updating docs/report.md ...")
    c_pct = f"{r.acc_c * 100:.2f}%"
    b_pct = f"{r.acc_b * 100:.2f}%"

    # Replace pending note and ≈ placeholders
    replace_re(path, r"≈89\.11%", c_pct, "C placeholder in report.md")
    replace_re(path, r"\*待运行\*", f"**{c_pct}**（全量 {r.c_correct}/{r.c_total}）", "pending note")

    # Update per-type C values in result table
    for occ, info in r.per_type.items():
        c_v = f"{info['c'] * 100:.2f}%"
        replace_re(path,
                   rf"(\|\s*{occ}[^\|]*\|[^\|]*\|[^\|]*\|)\s*[\d.]+%(\s*\|)",
                   rf"\g<1> {c_v}\2",
                   f"per-type C for {occ}")

    log.info("docs/report.md done")


# ── update docs/demo_report.md ────────────────────────────────────────────────

def update_demo_report(r: CompareResults) -> None:
    path = ROOT / "docs" / "demo_report.md"
    log.info("Updating docs/demo_report.md ...")
    c_pct = f"{r.acc_c * 100:.2f}%"
    b_pct = f"{r.acc_b * 100:.2f}%"

    # Replace pending placeholder
    replace_re(path, r"\*≈89\.11%\*", f"**{c_pct}**", "C placeholder in demo_report")
    replace_re(path, r"≈89\.11%", c_pct, "C placeholder in demo_report")

    # Update per-type breakdown table rows (C column)
    for occ, info in r.per_type.items():
        c_v = f"{info['c'] * 100:.2f}%"
        b_v = f"{info['b'] * 100:.2f}%"
        replace_re(path,
                   rf"(\|\s*\*\*{occ}[^|]*\|[^|]*\|[^|]*\|)\s*[\d.]+%(\s*\|)",
                   rf"\g<1> {c_v}\2",
                   f"demo per-type C for {occ}")

    log.info("docs/demo_report.md done")


# ── regenerate charts ─────────────────────────────────────────────────────────

def regenerate_charts() -> None:
    log.info("Regenerating charts ...")
    for script in ["scripts/plot_results.py", "scripts/generate_demo_report.py"]:
        cmd = ["uv", "run", "python", script]
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("Failed: %s\n%s", script, result.stderr[-500:])
        else:
            log.info("Done: %s", script)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    result_path = RESULTS_DIR / "compare_accuracy.txt"
    if not result_path.exists():
        log.error("compare_accuracy.txt not found — run `make eval-compare` first")
        sys.exit(1)

    r = parse_results(result_path)

    update_readme(r)
    update_changelog(r)
    update_report(r)
    update_demo_report(r)
    regenerate_charts()

    log.info("All reports updated. Run `git diff` to review changes.")


if __name__ == "__main__":
    main()
