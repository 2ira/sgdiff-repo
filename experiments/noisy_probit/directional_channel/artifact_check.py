"""Check final PDF pagination, text geometry, references, and file hashes."""
import hashlib,json,re,subprocess,xml.etree.ElementTree as E
from pathlib import Path


def main():
    root=Path('deliverables/noisy_probit_directional_english');tmp=Path('tmp/directional_qa');tmp.mkdir(parents=True,exist_ok=True);out={}
    for name,count in [('slides',12),('report',4)]:
        p=root/(name+'.pdf');bbox=tmp/(name+'.html');subprocess.run(['pdftotext','-bbox',str(p),str(bbox)],check=True)
        s=re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',bbox.read_text());tree=E.fromstring(s);ns={'x':'http://www.w3.org/1999/xhtml'};pages=tree.findall('.//x:page',ns)
        assert len(pages)==count,(name,len(pages));bad=[]
        for i,page in enumerate(pages,1):
            width,height=float(page.attrib['width']),float(page.attrib['height'])
            for word in page.findall('.//x:word',ns):
                x0,y0,x1,y1=[float(word.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
                if x0<0 or y0<0 or x1>width+.1 or y1>height+.1:bad.append((i,word.text))
        log=(root/(name+'.log')).read_text();assert not bad,(name,bad);assert 'Overfull' not in log
        assert not re.search(r'(Reference|Citation).*undefined',log)
        out[name]=dict(pages=count,out_of_page_words=bad,overfull_boxes=0,undefined_references=0,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    out['visual_review']='pending final rendered-page inspection'
    (root/'artifact_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
