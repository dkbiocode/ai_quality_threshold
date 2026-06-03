from __future__ import annotations
import functools
import logging
import os
import gzip
import shutil
import tempfile
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Callable, Tuple
from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor
from fastq_chunk import FastqRecord, get_read_dimensions, calculate_chunk_size, run_parallel
from quality_summary import quantiles_from_hist, print_seven_number_summary

logger = logging.getLogger(__name__)


@dataclass
class DegradeParams:
    """
    Parameters controlling synthetic quality degradation of FASTQ reads.
    
    Three degradation modes are applied in order:
    
    1. TAIL DEGRADATION: Simulates progressive signal decay toward read ends,
       caused by polymerase exhaustion and phasing accumulation.
       
       tail_start:  Position where quality decay begins. Default 150 means the
                    first 150 bases are unaffected by tail decay.
       tail_slope:  Phred score penalty per position past tail_start. Default 0.15
                    means a read at position 250 loses (250-150)*0.15 = 15 Phred
                    points from tail decay alone.
    
    2. GLOBAL NOISE: Simulates random per-base signal variation from optical
       crosstalk, incomplete cleavage, and general instrument noise.
       
       global_noise: Standard deviation of Gaussian noise added at every position.
                    Default 2.0 means ~95% of bases shift by ±4 Phred points.
    
    3. DROPOUT: Simulates sporadic low-quality positions caused by bubbles,
       surface chemistry failures, or damaged flow cell regions.
       
       dropout_rate:  Fraction of positions affected. Default 0.02 means ~1 in 50
                     bases receives a severe quality drop.
       dropout_floor: Minimum Phred score assigned to dropout positions. Default 0
                     allows dropouts to hit the lowest possible quality.
    
    GENERAL:
    
       seed:     Base random seed. Chunk index is added for per-chunk reproducibility.
       min_qual: Absolute Phred floor after all degradation. Default 0.
       max_qual: Absolute Phred ceiling. Default 40 (standard Illumina max).
    """
    tail_start:    int   = 150
    tail_slope:    float = 0.15
    global_noise:  float = 2.0
    dropout_rate:  float = 0.02
    dropout_floor: float = 0
    seed:          int   = 1123581321
    min_qual:      int   = 0
    max_qual:      int   = 40

att = ['tail_start','tail_slope', 'global_noise', 'dropout_rate', 'dropout_floor', 'seed','min_qual','max_qual']
def degrade_obj_to_str(obj):
    msg = []
    for atty in att: msg.append(f"{atty}={obj.__getattribute__(atty)}")
    return "; ".join(msg)

def degrade_chunk(
    chunk: Iterable[FastqRecord],
    params: DegradeParams,
    rng: np.random.Generator,
    hist: npt.NDArray[np.uint64] | None = None
) -> Iterator[FastqRecord]:
    """Apply random noise to quality scores for each record in chunk."""
    for rec in chunk:
        quals = np.array([ord(c) - 33 for c in rec.qualities])
        read_len = len(quals)

        if params.tail_start < read_len:
            tail_pos = np.maximum(np.arange(read_len) - params.tail_start, 0)
            quals = quals - tail_pos * params.tail_slope

        quals = quals + rng.normal(0, params.global_noise, read_len)

        dropout_mask = rng.random(read_len) < params.dropout_rate
        dropout_vals = rng.uniform(params.dropout_floor, params.dropout_floor + 5, read_len)
        quals = np.where(dropout_mask, np.minimum(quals, dropout_vals), quals)
        quals = np.clip(np.round(quals), params.min_qual, params.max_qual).astype(int)

        if hist is not None:
            for pos, q in enumerate(quals):
                hist[pos,q] += 1

        rec.qualities = ''.join(chr(q + 33) for q in quals)
        yield rec


def degrade_and_write_chunk(
    chunk: list[FastqRecord],
    chunk_idx: int,
    *,
    params: DegradeParams,
    temp_dir: str,
) -> Tuple[str, npt.NDArray[np.uint64]]:
    """Degrade quality scores in chunk and write to a numbered temp file.

    params and temp_dir are keyword-only so functools.partial can bind them,
    leaving (chunk, chunk_idx) as the two-argument worker contract for run_parallel.
    """
    rng = np.random.default_rng(params.seed + chunk_idx)
    temp_path = os.path.join(temp_dir, f"chunk_{chunk_idx:06d}.fastq.gz")
    
    with gzip.open(temp_path, 'wt') as fout:
        read_len = len(chunk[0].sequence)
        hist = np.zeros((read_len, params.max_qual+1), dtype=np.uint64)
        for rec in degrade_chunk(chunk, params, rng, hist):
            fout.write(f"@{rec.name}\n{rec.sequence}\n+\n{rec.qualities}\n")
    return temp_path, hist


def process_streaming(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    params: DegradeParams,
    *,
    chunk_size: int,
    n_workers: int = 4,
    temp_dir: str | None = None,
    executor_class: Callable[..., Executor] = ThreadPoolExecutor
) -> npt.NDArray[np.float64]:
    """Degrade quality scores across the full file using parallel chunk workers.

    Chunks are written to temp files on node-local storage (respects $TMPDIR),
    then concatenated at the raw gzip byte level — no decompress/recompress round-trip.
    The sliding window in run_parallel keeps at most n_workers chunks in memory at once.
    """
    with tempfile.TemporaryDirectory(dir = temp_dir) as tmp:

        worker = functools.partial(
            degrade_and_write_chunk, params=params, temp_dir=tmp
        )
        try:
            results = list(
                run_parallel(input_path, worker, chunk_size=chunk_size, n_workers=n_workers, executor_class=executor_class)
            )
        except Exception:
            logger.exception("parallel degradation failed for %s", input_path)
            raise

        temp_paths, hists = zip(*results)

        # concatenate degraded fastq.gzs
        with open(output_path, 'wb') as fout:
            for path in temp_paths:
                with open(path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)

            logger.info("Finished: concatenating %d chunks → %s", len(temp_paths), output_path)

        total_hist = np.sum(hists, axis=0)
        total_reads = int(total_hist[0].sum())
        quantiles = quantiles_from_hist(total_hist)

        return quantiles, total_reads


def main():

    MEM_PER_THREAD_MB = 50
    N_WORKERS = 6

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    basedir = os.path.dirname(os.path.abspath(__file__))
    R1 = f"{basedir}/sample_data/sample_S2_L001_R1_001.fastq.gz"
    R2 = f"{basedir}/sample_data/sample_S2_L001_R2_001.fastq.gz"


    #logname = datetime.now().strftime("degrade-log.%d-%M-%Y:%H:%m:%S.txt")
    jun_1 = int(datetime(2026,6,1).strftime("%s"))
    timestamp = int(datetime.now().strftime("%s")) % jun_1
    logname = f"degrade-{timestamp}.log"
    # degradation params
    degrade = DegradeParams()

    tmpdir = os.getenv('TMP') or os.getenv('TMPDIR') or '/tmp'
    with open(logname, "w") as log:
        print(degrade_obj_to_str(degrade), file=log)

        for R in [R1, R2]:
            read_dim = get_read_dimensions(R)
            if read_dim is None:
                raise ValueError(f"could not read any records from {R}")
            read_len, bytes_per_read, mem_per_read = read_dim
            chunksize = calculate_chunk_size(mem_per_read, mem_per_thread_mb=MEM_PER_THREAD_MB)
            print(f"{R=}\n{read_len=}, {bytes_per_read=}, {mem_per_read=} {chunksize=}")
            outpath = R.removesuffix('.fastq.gz') + '_degrade.fastq.gz'

            # run it
            quantiles, total_reads = process_streaming(R, outpath, degrade, chunk_size=chunksize, n_workers= N_WORKERS, temp_dir=tmpdir,executor_class=ProcessPoolExecutor)
            tsv_file = outpath.replace('.fastq.gz', '.seven-number-summaries.tsv')
            with open(tsv_file, 'w') as tsv: print(f'#{degrade_obj_to_str(degrade)}', file=tsv)
            print_seven_number_summary(quantiles, tsv_file, total_reads=total_reads)
            # save info
            print(outpath, file=log)

    print(f"{MEM_PER_THREAD_MB=}, {N_WORKERS=}")

if __name__ == "__main__":
    main()
