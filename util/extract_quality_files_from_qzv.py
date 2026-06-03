#!/usr/bin/env python3
import sys
import zipfile
import os
import shutil

qzv = sys.argv[1]

with zipfile.ZipFile(qzv, 'r') as z:
    for name in z.namelist():
        if name.endswith('-seven-number-summaries.tsv'):
            outname = os.path.basename(name)
            if os.path.exists(outname):
                print(f"{outname} exists, skipping.", file=sys.stderr)
                continue
            with z.open(name) as fin, open(outname,'wb') as tsv:
                shutil.copyfileobj(fin,tsv)
                print(f"extracted {outname}.")