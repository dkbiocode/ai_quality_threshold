from __future__ import annotations
import functools
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple

from fastq_chunk import FastqRecord

MAX_QUAL = 41
QUANTILE_LEVELS = [.02, .09, 0.25, 0.5, 0.75, .91, .98]
QUANTILE_LABELS = ['2%', '9%', '25%', '50%', '75%', '91%', '98%']


def hist_chunk(
    chunk: list[FastqRecord],
    chunk_idx: int,
    *,
    max_qual: int = MAX_QUAL,
) -> npt.NDArray[np.uint64]:
    """Accumulate a per-position quality histogram over one chunk of reads.

    Returns an array of shape (read_len, max_qual+1) where hist[pos, q] is the
    number of reads with Phred score q at position pos.
    """
    read_len = len(chunk[0].sequence)
    hist = np.zeros((read_len, max_qual + 1), dtype=np.uint64)
    qual_mat = np.empty((len(chunk), read_len), dtype=np.int16)
    for i, rec in enumerate(chunk):
        qual_mat[i] = np.frombuffer(rec.qualities.encode(), dtype=np.uint8) - 33
    for pos in range(read_len):
        hist[pos] = np.bincount(qual_mat[:, pos], minlength=max_qual + 1)
    return hist


def hist_chunk_paired(
    r1_chunk: list[FastqRecord],
    r2_chunk: list[FastqRecord],
    chunk_idx: int,
    *,
    max_qual: int = MAX_QUAL,
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Accumulate quality histograms for one paired chunk. Returns (hist_r1, hist_r2)."""
    return (
        hist_chunk(r1_chunk, chunk_idx, max_qual=max_qual),
        hist_chunk(r2_chunk, chunk_idx, max_qual=max_qual),
    )


def quantiles_from_hist(
    hist: npt.NDArray[np.uint64],
    q: list[float] = QUANTILE_LEVELS,
) -> npt.NDArray[np.float64]:
    """Compute per-position quantiles from a (read_len, max_qual+1) histogram."""
    result = np.zeros((hist.shape[0], len(q)))
    for pos in range(hist.shape[0]):
        cum = np.cumsum(hist[pos])
        total = cum[-1]
        if total == 0:
            result[pos] = np.nan
            continue
        result[pos] = np.searchsorted(cum, np.array(q) * total)
    return result


def print_seven_number_summary(
    quantiles: npt.NDArray[np.float64],
    outcsv: str,
    total_reads: int | None = None,
) -> None:
    """Append a per-position 7-number quality summary as TSV.

    Columns are 1-based read positions. Rows are the seven quantile percentiles
    preceded by a count row when total_reads is provided.
    """
    df = pd.DataFrame(
        quantiles.T,
        index=pd.Index(QUANTILE_LABELS),
    )
    if total_reads is not None:
        df.loc['count'] = total_reads
        df = df.reindex(['count'] + QUANTILE_LABELS)
    if isinstance(df.columns, pd.RangeIndex):
        df.columns = pd.RangeIndex(len(df.columns)) + 1
    df.to_csv(outcsv, sep="\t", mode="a")
