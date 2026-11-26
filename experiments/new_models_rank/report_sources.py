"""Package a standalone TeX source bundle after figures and tables are generated."""
import argparse,hashlib,json,re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main(a):
    text=(a.results/'report.tex').read_text()
    figures=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',text)
    assert len(figures)==len(set(figures))==17
    names=['report.tex','references.bib',*figures]
    for name in names:
        assert (a.results/name).is_file(),name
    instructions='''# Standalone Chinese TeX report\n\nAll numerical tables are embedded in report.tex. All 17 PDF figures and\nreferences.bib are included. No model, GPU, dataset or network API is needed.\n\nCompile with Tectonic 0.17.0:\n\n    tectonic report.tex\n\nOn first use Tectonic downloads its TeX/CTeX/Fandol dependencies. Once cached,\ncompilation works offline. Alternatively use a recent TeX Live installation:\n\n    xelatex report.tex\n    bibtex report\n    xelatex report.tex\n    xelatex report.tex\n\nThe Chinese fontset is Fandol. Source result tables and CPU replay are in the\nrepository results/new-models-node-rank and experiments/new_models_rank.\n'''
    with ZipFile(a.results/'report_sources.zip','w',ZIP_DEFLATED,compresslevel=9) as z:
        for name in names:z.write(a.results/name,name)
        z.writestr('README.md',instructions)
        z.writestr('SHA256SUMS',''.join(hashlib.sha256((a.results/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in names))
    print('Standalone TeX bundle:',len(figures),'figures')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    main(p.parse_args())
