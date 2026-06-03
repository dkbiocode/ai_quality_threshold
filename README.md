# ai_quality_threshold

Use an LLM to automatically suggest trimming thresholds based on quality criteria. The output may be used in automation to skip a manual step, furlow poor quality samples, or simply evaluate quality.

<a href="img/ai_quality_threshold_plot_good.png"><img src="img/ai_quality_threshold_plot_good.png" width=500px></a><a href="img/ai_quality_threshold_plot_degrade_global_2.png"><img src="img/ai_quality_threshold_plot_degrade_global_2.png" width=500px></a>

A good set of paired fastq reads *left* and a degraded set *right*. The red area shows the AI recommended trim. Click on the image to see more detail, including the AI generated rationale below the plots.

### Description

This repo contains a script to ask AI to apply specific rules to data in order to suggest quality trimming parameters. The performance depends on the data, so this repo also contains a framework for randomly degrading the quality scores of a given pair of fastq files. In order to do this efficiently on large files, I developed [fastq_chunk](github.com/dkbiocode/fastq_chunk), a python package for distributing a user defined function across multiple compute cores. 

## Usage

This repo applies an AI step to estimate truncation parameters from fastq quality quantiles like those produced by the data import steps of Qiime2. 

### Examples

#### Run directly on output from Qiime2 summary object

```
python ai_fastq_choose_thresholds.py demux_summary.qzv
```

#### Run on files of quantiles in the seven-number-summaries.tsv format

```
python ai_fastq_choose_thresholds.py forward-seven-number-summaries.tsv reverse-seven-number-summaries.tsv
```

## Installation:

### API credentials

This script uses OpenAI (yeah, I know). Go to OpenAI and get an API key. This has to be in your environment for the script to make its prompt to the model.

Example .bashrc/.zshrc:

```sh
export OPENAI_API_KEY='sk-Y5lj...'
```

Relevant lines in python:

```python
import os # to read from user ENV
from openai import OpenAI
 
# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### conda

#### install environment

This also installs fastq\_chunk from dkbiocode.

```
conda env create -f environment.yml
conda activate ai-fastqc-choose-point
```

#### test fastq\_chunk module import

This will list the location of the installed module, or give an error if installation failed.

```
python -c "import fastq_chunk; print(fastq_chunk.__file__)"
```

### pip (without conda environment)

```
pip install -r requirements.txt
```

#### test fastq\_chunk module import

This will list the location of the installed module, or give an error if installation failed.

```
python -c "import fastq_chunk; print(fastq_chunk.__file__)"
```
