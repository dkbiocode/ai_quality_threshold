#!/usr/bin/env python3
"""Compute per-position 7-number quality summaries for single-end or paired-end FASTQ files."""
from __future__ import annotations
import argparse
import functools
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


def summarize_single(fastq_path: str, output_tsv: str, n_workers: int = 4) -> None:
    read_dim = get_read_dimensions(fastq_path)
    if read_dim is None:
        raise ValueError(f"no reads found in {fastq_path}")
    _, _, mem_per_read = read_dim
    chunk_size = calculate_chunk_size(mem_per_read)

    hists = list(run_parallel(
        fastq_path, hist_chunk,
        chunk_size=chunk_size, n_workers=n_workers, executor_class=ProcessPoolExecutor,
    ))
    total_hist = np.sum(hists, axis=0)
    total_reads = int(total_hist[0].sum())
    quantiles = quantiles_from_hist(total_hist)
    print_seven_number_summary(quantiles, output_tsv, total_reads=total_reads)
    logger.info("written: %s (%d reads)", output_tsv, total_reads)


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

    for hists, tsv in ((r1_hists, r1_tsv), (r2_hists, r2_tsv)):
        total_hist = np.sum(hists, axis=0)
        total_reads = int(total_hist[0].sum())
        quantiles = quantiles_from_hist(total_hist)
        print_seven_number_summary(quantiles, tsv, total_reads=total_reads)
        logger.info("written: %s (%d reads)", tsv, total_reads)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'fastq', nargs='+', metavar='FASTQ',
        help="One FASTQ (single-end) or two FASTQs (paired-end R1 R2)",
    )
    parser.add_argument(
        '--output', '-o', nargs='+', metavar='TSV',
        help="Output TSV path(s). Defaults to input basename + .seven-number-summary.tsv",
    )
    parser.add_argument('--workers', type=int, default=4, metavar='N')
    args = parser.parse_args()

    if len(args.fastq) == 1:
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
        parser.error("provide 1 (single-end) or 2 (paired-end) FASTQ files")


if __name__ == "__main__":
    main()
