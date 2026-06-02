#!/usr/bin/env python3
import sys
import os
import zipfile
import csv
import io
import json
from openai import OpenAI
# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-4o-mini"
amplicon_length = 0
endsize = 20

def summarize_tail(tsv_data, last_n=50):
    reader = csv.reader(io.StringIO(tsv_data), delimiter='\t')
    header = next(reader)
    # Keep first column (percentile label) + last N position columns
    out = io.StringIO()
    writer = csv.writer(out, delimiter='\t')
    writer.writerow([header[0]] + header[-last_n:])
    for row in reader:
        if row[0].strip() == 'count':
            continue
        writer.writerow([row[0]] + row[-last_n:])
    return out.getvalue()


demux_file = ""
if len(sys.argv) == 1:
    demux_file = "demux_summary.qzv"
elif len(sys.argv) == 2:
    demux_file = sys.argv[1]
elif len(sys.argv) == 3:
    with open(sys.argv[1]) as f_tsv:
        fwd_data = summarize_tail( f_tsv.read(), endsize)
    with open(sys.argv[2]) as r_tsv:
        rev_data = summarize_tail( r_tsv.read(), endsize)

if demux_file: 
    with zipfile.ZipFile(demux_file, 'r') as z:
        for name in z.namelist():
            if name.endswith('forward-seven-number-summaries.tsv'):
                with z.open(name) as f:
                    fwd_data = summarize_tail( f.read().decode('utf-8').replace('\r', ''), endsize )
            elif name.endswith('reverse-seven-number-summaries.tsv'):
                with z.open(name) as f:
                    rev_data = summarize_tail( f.read().decode('utf-8').replace('\r', ''), endsize )






prompt = f"""You are analyzing quality scores from 16S amplicon sequencing data 
to recommend DADA2 denoise-paired truncation parameters.

The data below shows per-position Phred quality score percentiles for the 
tail end of forward and reverse reads. Rows are percentile levels (2%, 9%, 
25%, 50%, 75%, 91%, 98%). Columns are base positions. 

Forward reads (last {endsize} positions):
{fwd_data}

Reverse reads (last {endsize} positions):
{rev_data}

Amplicon length: {amplicon_length} bp (0 if unknown)

Recommend trunc-len-f and trunc-len-r for DADA2 denoise-paired using these criteria:
- 25th percentile should remain above 25 at the truncation point
- Median (50%) should remain above 30 at the truncation point
- If amplicon length is known: trunc_len_f + trunc_len_r - amplicon_length > 20 
  (sufficient merge overlap)
- If amplicon length is 0 (unknown): be conservative, favor retaining read length
- Truncate before any sustained quality drop, not just single-position dips
- Reverse reads typically degrade faster — expect a shorter trunc-len-r

Respond ONLY with valid JSON, no markdown formatting:
{{"trunc_len_f": N, "trunc_len_r": N, "reasoning": "..."}}"""

#print(prompt)

response = client.chat.completions.create(
    model=model,
    messages=[
        #{"role": "system", "content": "You are an expert at analyzing scientific queries about NGS diagnostics."},
        {"role": "user", "content": prompt}
    ],  
    response_format={"type": "json_object"},
    temperature=0.3  # Lower temperature for more consistent decomposition
)   

result = json.loads(str(response.choices[0].message.content))

# Add original data to result
#result["original_data"] = data

print(result)
print(f"{result['trunc_len_f']=}")
print(f"{result['trunc_len_r']=}")
print(f"{result['reasoning']=}")