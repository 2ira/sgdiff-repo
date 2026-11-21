#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build_tables.py
if command -v "${REPORT_TECTONIC:-tectonic}" >/dev/null 2>&1; then
  "${REPORT_TECTONIC:-tectonic}" --keep-logs report.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
elif command -v xelatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error report.tex
  bibtex report
  xelatex -interaction=nonstopmode -halt-on-error report.tex
  xelatex -interaction=nonstopmode -halt-on-error report.tex
else
  printf '%s\n' 'Please install Tectonic or XeLaTeX + BibTeX (TeX Live with ctex/Fandol).' >&2
  exit 1
fi
