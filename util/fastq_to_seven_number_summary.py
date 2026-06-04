#!/usr/bin/env python3
"""Compute per-position 7-number quality summaries for single-end or paired-end FASTQ files."""
from __future__ import annotations
import argparse
import logging
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastq_chunk import get_read_dimensions, calculate_chunk_size, run_parallel, run_parallel_paired
from quality_summary import (
    hist_chunk, hist_chunk_paired,
    quantiles_from_hist, print_seven_number_summary,
)

logger = logging.getLogger(__name__)


def _default_tsv(fastq_path: str) -> str:
    base = fastq_path
    for suffix in ('.fastq.gz', '.fastq', '.fq.gz', '.fq'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return base + '.seven-number-summary.tsv'


def _file_hist(fastq_path: str, n_workers: int) -> np.ndarray:
    """Return a summed (read_len, max_qual+1) histogram for one FASTQ file."""
    read_dim = get_read_dimensions(fastq_path)
    if read_dim is None:
        raise ValueError(f"no reads found in {fastq_path}")
    _, _, mem_per_read = read_dim
    chunk_size = calculate_chunk_size(mem_per_read)
    hists = list(run_parallel(
        fastq_path, hist_chunk,
        chunk_size=chunk_size, n_workers=n_workers, executor_class=ProcessPoolExecutor,
    ))
    return np.sum(hists, axis=0)


def _write_summary(hist: np.ndarray, output_tsv: str, label: str) -> None:
    total_reads = int(hist[0].sum())
    quantiles = quantiles_from_hist(hist)
    print_seven_number_summary(quantiles, output_tsv, total_reads=total_reads)
    logger.info("written: %s (%d reads, %s)", output_tsv, total_reads, label)


def summarize_single(fastq_path: str, output_tsv: str, n_workers: int = 4) -> None:
    hist = _file_hist(fastq_path, n_workers)
    _write_summary(hist, output_tsv, fastq_path)


def summarize_combined(fastq_paths: list[str], output_tsv: str, n_workers: int = 4) -> None:
    """Sum histograms across all files and write one combined summary.

    All files must have the same read length (histogram shapes must match).
    """
    total_hist = None
    for path in fastq_paths:
        h = _file_hist(path, n_workers)
        if total_hist is None:
            total_hist = h
        elif total_hist.shape != h.shape:
            raise ValueError(
                f"read length mismatch: expected {total_hist.shape[0]} bp, "
                f"got {h.shape[0]} bp in {path}"
            )
        else:
            total_hist += h
    _write_summary(total_hist, output_tsv, f"{len(fastq_paths)} files")


def summarize_paired(
    r1_path: str, r2_path: str,
    r1_tsv: str, r2_tsv: str,
    n_workers: int = 4,
) -> None:
    read_dim = get_read_dimensions(r1_path)
    if read_dim is None:
        raise ValueError(f"no reads found in {r1_path}")
    _, _, mem_per_read = read_dim
    chunk_size = calculate_chunk_size(mem_per_read)

    results = list(run_parallel_paired(
        r1_path, r2_path, hist_chunk_paired,
        chunk_size=chunk_size, n_workers=n_workers, executor_class=ProcessPoolExecutor,
    ))
    r1_hists, r2_hists = zip(*results)

    for hists, tsv, label in (
        (r1_hists, r1_tsv, r1_path),
        (r2_hists, r2_tsv, r2_path),
    ):
        _write_summary(np.sum(hists, axis=0), tsv, label)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'fastq', nargs='+', metavar='FASTQ',
        help="Input FASTQ file(s). Without --combine: 1 file (single-end) or 2 files (paired-end). "
             "With --combine: any number of files combined into one summary.",
    )
    parser.add_argument(
        '--combine', action='store_true',
        help="Sum histograms across all input files and write a single combined summary.",
    )
    parser.add_argument(
        '--output', '-o', nargs='+', metavar='TSV',
        help="Output TSV path(s). Defaults to input basename + .seven-number-summary.tsv",
    )
    parser.add_argument('--workers', type=int, default=4, metavar='N')
    args = parser.parse_args()

    if args.combine:
        tsv = (args.output or [_default_tsv(args.fastq[0])])[0]
        summarize_combined(args.fastq, tsv, n_workers=args.workers)
        print(f"Written: {tsv}")

    elif len(args.fastq) == 1:
        tsv = (args.output or [_default_tsv(args.fastq[0])])[0]
        summarize_single(args.fastq[0], tsv, n_workers=args.workers)
        print(f"Written: {tsv}")

    elif len(args.fastq) == 2:
        r1, r2 = args.fastq
        outputs = args.output or [_default_tsv(r1), _default_tsv(r2)]
        if len(outputs) != 2:
            parser.error("--output requires exactly 2 paths for paired-end input")
        summarize_paired(r1, r2, outputs[0], outputs[1], n_workers=args.workers)
        print(f"Written: {outputs[0]}, {outputs[1]}")

    else:
        parser.error("provide 1 (single-end) or 2 (paired-end) FASTQ files, or use --combine for N files")


if __name__ == "__main__":
    main()
