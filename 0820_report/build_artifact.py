#!/usr/bin/env python3
"""Build report_artifact.html from report.html by inlining figs/*.png as base64 data URIs.
Single source of truth: edit report.html, run this, republish."""
import base64, re
s=open('report.html').read()
def emb(m):
    b=base64.b64encode(open('figs/'+m.group(1),'rb').read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt='
s2, n = re.subn(r'<img src="figs/([^"]+)" alt=', emb, s)
open('report_artifact.html','w').write(s2)
print(f'embedded {n} figures, {len(s2)} bytes')
