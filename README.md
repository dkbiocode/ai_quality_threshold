
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

