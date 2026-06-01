from __future__ import annotations
import functools
import logging
import os
import gzip
import shutil
import tempfile
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Iterator

from fastq_chunk import FastqRecord, get_read_dimensions, calculate_chunk_size, run_parallel

logger = logging.getLogger(__name__)


@dataclass
class DegradeParams:
    tail_start:    int
    tail_slope:    float
    global_noise:  float
    dropout_rate:  float
    dropout_floor: float
    seed:          int
    min_qual:      int = 0
    max_qual:      int = 40


def degrade_chunk(
    chunk: Iterable[FastqRecord],
    params: DegradeParams,
    rng: np.random.Generator,
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
        rec.qualities = ''.join(chr(q + 33) for q in quals)
        yield rec


def degrade_and_write_chunk(
    chunk: list[FastqRecord],
    chunk_idx: int,
    *,
    params: DegradeParams,
    temp_dir: str,
) -> str:
    """Degrade quality scores in chunk and write to a numbered temp file.

    params and temp_dir are keyword-only so functools.partial can bind them,
    leaving (chunk, chunk_idx) as the two-argument worker contract for run_parallel.
    """
    rng = np.random.default_rng(params.seed + chunk_idx)
    temp_path = os.path.join(temp_dir, f"chunk_{chunk_idx:06d}.fastq.gz")
    with gzip.open(temp_path, 'wt') as fout:
        for rec in degrade_chunk(chunk, params, rng):
            fout.write(f"@{rec.name}\n{rec.sequence}\n+\n{rec.qualities}\n")
    return temp_path


def process_streaming(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    params: DegradeParams,
    *,
    chunk_size: int,
    n_workers: int = 4,
    temp_dir: str | None = None,
) -> None:
    """Degrade quality scores across the full file using parallel chunk workers.

    Chunks are written to temp files on node-local storage (respects $TMPDIR),
    then concatenated at the raw gzip byte level — no decompress/recompress round-trip.
    The sliding window in run_parallel keeps at most n_workers chunks in memory at once.
    """
    working_dir = temp_dir or tempfile.mkdtemp(prefix="degrade_")
    owned = temp_dir is None  # only clean up dirs we created

    try:
        worker = functools.partial(
            degrade_and_write_chunk, params=params, temp_dir=working_dir
        )
        temp_paths = list(
            run_parallel(input_path, worker, chunk_size=chunk_size, n_workers=n_workers)
        )
        logger.info("concatenating %d chunks → %s", len(temp_paths), output_path)
        with open(output_path, 'wb') as fout:
            for path in temp_paths:
                with open(path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)
    finally:
        if owned:
            shutil.rmtree(working_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    basedir = os.path.dirname(os.path.abspath(__file__))
    R1 = f"{basedir}/sample_data/2629LEI-2_S2_L001_R1_001.fastq.gz"
    R2 = f"{basedir}/sample_data/2629LEI-2_S2_L001_R2_001.fastq.gz"
    for R in [R1, R2]:
        read_dim = get_read_dimensions(R)
        if read_dim is None:
            raise ValueError(f"could not read any records from {R}")
        read_len, bytes_per_read, mem_per_read = read_dim
        chunksize = calculate_chunk_size(mem_per_read)
        print(f"{R=}\n{read_len=}, {bytes_per_read=}, {mem_per_read=} {chunksize=}")


if __name__ == "__main__":
    main()
