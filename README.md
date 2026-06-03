
## Usage

This repo applies an AI step to estimate truncation parameters from fastq quality quantiles like those produced by the data import steps of Qiime2. 

This repo contains a script to ask AI to apply specific rules to data in order to suggest quality trimming parameters. The performance depends on the data, so this repo also contains a framework for randomly degrading the quality scores of a given pair of fastq files. In order to do this efficiently on large files, I developed [fastq_chunk](github.com/dkbiocode/fastq_chunk), a python package for distributing a user defined function across multiple compute cores. 

## Installation:

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
