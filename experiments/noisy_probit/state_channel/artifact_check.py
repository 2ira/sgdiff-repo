"""PDF pagination, text geometry, compiler, and artifact hash checks."""
import hashlib,json,re,subprocess
from pathlib import Path
import xml.etree.ElementTree as E


def main():
    root=Path('deliverables/noisy_probit_state_channel_english');tmp=Path('tmp/state_channel_qa');tmp.mkdir(parents=True,exist_ok=True)
    checks={}
    for name,expected in [('slides',16),('report',4)]:
        pdf=root/(name+'.pdf');bbox=tmp/(name+'_bbox.html')
        subprocess.run(['pdftotext','-bbox',str(pdf),str(bbox)],check=True)
        # TeX extensible delimiter fonts map some glyphs to XML control characters.
        # Strip these only for geometry parsing. The rendered PDF is unchanged.
        s=re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',bbox.read_text())
        tree=E.fromstring(s);ns={'x':'http://www.w3.org/1999/xhtml'};pages=tree.findall('.//x:page',ns)
        assert len(pages)==expected,(name,len(pages));bad=[]
        for i,p in enumerate(pages,1):
            w,h=float(p.attrib['width']),float(p.attrib['height'])
            for word in p.findall('.//x:word',ns):
                x0,y0,x1,y1=[float(word.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
                if x0<0 or y0<0 or x1>w+.1 or y1>h+.1:bad.append((i,word.text))
        assert not bad,(name,bad)
        log=(root/(name+'.log')).read_text()
        assert 'Overfull' not in log
        assert not re.search(r'(Reference|Citation).*undefined',log)
        checks[name]=dict(pages=len(pages),out_of_page_words=bad,overfull_boxes=0,undefined_references=0,
                          sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),bytes=pdf.stat().st_size)
    checks['visual_review']='pending final rendered-page inspection'
    (root/'artifact_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
