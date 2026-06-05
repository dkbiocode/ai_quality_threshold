#!/usr/bin/env python3
import csv
import io
import json
import os
import sys
import zipfile
from typing import Sequence, TypedDict, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-4o-mini"
amplicon_length = 0
endsize = 50
min_q_25p =  25  # - 25th percentile should remain above 25 at the truncation point
min_q_50p =  30  # - Median (50%) should remain above 30 at the truncation point
PERCENTILE_LABELS = ['2%', '9%', '25%', '50%', '75%', '91%', '98%']


class Summary(TypedDict):
    comment: Optional[str]
    df: pd.DataFrame


def summarize_tail(tsv_data: str, last_n: int = 50) -> str:
    reader = csv.reader(io.StringIO(tsv_data), delimiter='\t')
    header = next(reader)
    out = io.StringIO()
    writer = csv.writer(out, delimiter='\t')
    writer.writerow([header[0]] + header[-last_n:])
    for row in reader:
        if row[0].strip() == 'count':
            continue
        writer.writerow([row[0]] + row[-last_n:])
    return out.getvalue()


def parse_summary_tsv(tsv_data: str) -> Summary:
    reader = csv.reader(io.StringIO(tsv_data), delimiter='\t')
    comment = None
    header = next(reader)
    if header and header[0].strip().startswith('#'):
        comment = '\t'.join(header)
        header = next(reader)

    positions = [int(x.strip()) for x in header[1:] if x.strip()]
    labels: list[str] = []
    rows: list[list[float]] = []
    for row in reader:
        label = row[0].strip()
        if label:
            labels.append(label)
            rows.append([float(x) for x in row[1:]])

    if not labels:
        raise ValueError('No percentile rows found in TSV')

    df = pd.DataFrame(rows, index=labels, columns=positions, dtype=float)
    missing = [lab for lab in PERCENTILE_LABELS if lab not in df.index]
    if missing:
        raise ValueError(f"Missing percentile rows: {', '.join(missing)}")

    return {'comment': comment, 'df': df}


def plot_quality_summary(
    axes,
    summary: Summary,
    label: str,
    trunc_len: int,
    base_color: str,
    trim_color: str,
    window_start: Optional[int] = None,
    q25_threshold: float = 0,
    q50_threshold: float = 0,
    trim_left: int = 0,
) -> None:
    df: pd.DataFrame = summary['df']
    positions = np.array(df.columns, dtype=int)
    box_bottom = df.loc['25%'].to_numpy()
    box_top = df.loc['75%'].to_numpy()
    box_height = box_top - box_bottom
    bar_colors = [
        trim_color if pos > trunc_len or (trim_left > 0 and pos <= trim_left) else base_color
        for pos in positions
    ]

    axes.bar(
        positions,
        box_height,
        bottom=box_bottom,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.4,
        width=0.8,
        label='IQR (25%-75%)',
    )
    q2  = df.loc['2%'].to_numpy()
    q9  = df.loc['9%'].to_numpy()
    q91 = df.loc['91%'].to_numpy()
    q98 = df.loc['98%'].to_numpy()
    for i, pos in enumerate(positions):
        axes.vlines(pos, q2[i],  q98[i], color='grey',  alpha=0.45, linewidth=0.8)
        axes.vlines(pos, q9[i],  q91[i], color='black', alpha=0.5,  linewidth=1.0)

    axes.plot(positions, df.loc['50%'].to_numpy(), color='black', linewidth=1.4, label='Median (50%)')
    axes.axvline(trunc_len, color='red', linestyle='--', linewidth=1.2, label=f"3' trunc at {trunc_len}")
    if trim_left > 0:
        axes.axvline(trim_left + 0.5, color='darkorange', linestyle='--', linewidth=1.2, label=f"5' trim at {trim_left}")
    if window_start is not None:
        axes.axvline(window_start, color='orange', linestyle=':', linewidth=4.0, label=f'analysis window (pos {window_start})')
    if q25_threshold > 0:
        axes.axhline(q25_threshold, color='steelblue', linestyle='--', linewidth=1.0, alpha=0.8, label=f'min 25th pct ({q25_threshold})')
    if q50_threshold > 0:
        axes.axhline(q50_threshold, color='seagreen', linestyle='--', linewidth=1.0, alpha=0.8, label=f'min median ({q50_threshold})')
    axes.set_title(f'{label} read quality summary')
    axes.set_xlabel('Base position')
    axes.set_ylabel('Phred quality score')
    axes.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    axes.set_ylim(0, max(df.loc['98%'].max(), 40) * 1.05)
    axes.xaxis.set_major_locator(MultipleLocator(10))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}' if x == int(x) else ''))
    axes.legend(fontsize='small', ncol=2)
    axes.grid(axis='y', alpha=0.25)


def build_plot(
    fwd_summary: Summary,
    rev_summary: Summary,
    trunc_len_f: int,
    trunc_len_r: int,
    reasoning: str,
    fwd_window_start: Optional[int] = None,
    rev_window_start: Optional[int] = None,
    trim_left_f: int = 0,
    trim_left_r: int = 0,
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    plot_quality_summary(
        axes[0],
        fwd_summary,
        label='Forward',
        trunc_len=trunc_len_f,
        base_color='skyblue',
        trim_color='red',
        window_start=fwd_window_start,
        q25_threshold=min_q_25p,
        q50_threshold=min_q_50p,
        trim_left=trim_left_f,
    )
    plot_quality_summary(
        axes[1],
        rev_summary,
        label='Reverse',
        trunc_len=trunc_len_r,
        base_color='lightgreen',
        trim_color='red',
        window_start=rev_window_start,
        q25_threshold=min_q_25p,
        q50_threshold=min_q_50p,
        trim_left=trim_left_r,
    )

    fig.text(0.5, 0.01, reasoning, ha='center', va='bottom', fontsize=10, wrap=True, family='monospace')
    fig.tight_layout(rect=(0., 0.12, 1., 1.))

    out_path = 'ai_quality_threshold_plot.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def load_data_from_qzv(qzv_path: str) -> tuple[str, str]:
    """Load full TSV data from QZV archive for both plotting and prompt extraction."""
    fwd_data = ''
    rev_data = ''
    with zipfile.ZipFile(qzv_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('forward-seven-number-summaries.tsv'):
                with z.open(name) as f:
                    fwd_data = f.read().decode('utf-8').replace('\r', '')
            elif name.endswith('reverse-seven-number-summaries.tsv'):
                with z.open(name) as f:
                    rev_data = f.read().decode('utf-8').replace('\r', '')
    if not fwd_data or not rev_data:
        raise ValueError(f'Could not find parsed forward/reverse summary TSV files in {qzv_path}')
    return fwd_data, rev_data


def load_data_from_tsv_files(forward_path: str, reverse_path: str) -> tuple[str, str]:
    """Load full TSV data (not summarized) for both plotting and prompt extraction."""
    with open(forward_path, 'r', newline='') as fwd_handle:
        fwd_data = fwd_handle.read()
    with open(reverse_path, 'r', newline='') as rev_handle:
        rev_data = rev_handle.read()
    return fwd_data, rev_data


def find_max_valid_position(summary: Summary) -> Optional[int]:
    """Return the last position where both quality thresholds are met, or None."""
    df = summary['df']
    positions = np.array(df.columns, dtype=int)
    q25 = df.loc['25%'].to_numpy()
    q50 = df.loc['50%'].to_numpy()
    mask = (q25 >= min_q_25p) & (q50 >= min_q_50p)
    valid = positions[mask]
    return int(valid[-1]) if len(valid) > 0 else None


def compute_window_cols(summary: Summary, valid_count: int = 30, padding: int = 10) -> list:
    """Return column labels for the analysis window: last valid_count positions meeting
    both thresholds, plus padding positions on each side for context.
    Raises ValueError if no positions meet the thresholds."""
    df = summary['df']
    q25 = df.loc['25%'].to_numpy()
    q50 = df.loc['50%'].to_numpy()
    mask = (q25 >= min_q_25p) & (q50 >= min_q_50p)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f'No positions meet quality thresholds '
            f'(25th pct >= {min_q_25p}, median >= {min_q_50p}). '
            f'Data may be too degraded for analysis.'
        )

    tail_indices = valid_indices[-valid_count:]
    start_idx = max(0, int(tail_indices[0]) - padding)
    end_idx = min(len(df.columns) - 1, int(tail_indices[-1]) + padding)
    return list(df.columns[start_idx:end_idx + 1])


def extract_tail_from_summary(summary: Summary, cols: list) -> str:
    """Format the given columns (25th pct and median only) as TSV for the prompt."""
    df: pd.DataFrame = summary['df']
    header_row = ['percentile'] + [str(int(c)) for c in cols]
    rows = ['\t'.join(header_row)]
    for label in ['25%', '50%']:
        data = df.loc[label][cols]
        rows.append(label + '\t' + '\t'.join(f'{v:.1f}' for v in data.to_numpy(dtype=float)))
    return '\n'.join(rows)


def build_prompt(fwd_summary: Summary, rev_summary: Summary, fwd_cols: list, rev_cols: list) -> str:
    """Build the prompt using precomputed analysis windows."""
    fwd_tail = extract_tail_from_summary(fwd_summary, fwd_cols)
    rev_tail = extract_tail_from_summary(rev_summary, rev_cols)
    fwd_max = find_max_valid_position(fwd_summary)
    rev_max = find_max_valid_position(rev_summary)

    return f"""You are analyzing quality scores from 16S amplicon sequencing data
    to recommend DADA2 denoise-paired truncation parameters.

    Quality thresholds: 25th percentile >= {min_q_25p}, median >= {min_q_50p}.
    Both thresholds are met through: forward position {fwd_max}, reverse position {rev_max}.
    Do not suggest a truncation point beyond these positions.

    The data below shows 25th percentile and median Phred quality scores around the
    quality drop-off for each read (positions {int(fwd_cols[0])}–{int(fwd_cols[-1])} forward,
    {int(rev_cols[0])}–{int(rev_cols[-1])} reverse). Columns are base positions.

    Forward reads:
    {fwd_tail}

    Reverse reads:
    {rev_tail}

    Amplicon length: {amplicon_length} bp (0 if unknown)

    Recommend trunc-len-f and trunc-len-r for DADA2 denoise-paired using these criteria:
    - Truncate before any sustained quality drop, not just single-position dips
    - If amplicon length is known: trunc_len_f + trunc_len_r - amplicon_length > 20
      (sufficient merge overlap)
    - If amplicon length is 0 (unknown): be conservative, favor retaining read length
    - Reverse reads typically degrade faster — expect a shorter trunc-len-r

    Respond ONLY with valid JSON, no markdown formatting:
    {{"trunc_len_f": N, "trunc_len_r": N, "reasoning": "..."}}"""


def build_overlap_prompt(
    fwd_summary: Summary,
    rev_summary: Summary,
    fwd_cols: list,
    rev_cols: list,
    initial_trunc_f: int,
    initial_trunc_r: int,
) -> str:
    """Secondary prompt (amplicon length known): refine allocation using trend and overlap budget."""
    fwd_tail = extract_tail_from_summary(fwd_summary, fwd_cols)
    rev_tail = extract_tail_from_summary(rev_summary, rev_cols)
    current_overlap = initial_trunc_f + initial_trunc_r - amplicon_length
    min_sum = amplicon_length + 20

    return f"""You are refining DADA2 truncation parameters for 16S amplicon sequencing data.

    A threshold-based analysis produced an initial suggestion:
      trunc_len_f = {initial_trunc_f}, trunc_len_r = {initial_trunc_r}
      overlap margin = {current_overlap} bp (amplicon {amplicon_length} bp, minimum overlap 20 bp)
      constraint: trunc_len_f + trunc_len_r >= {min_sum}

    Quality data (25th percentile and median) around the transition zone:

    Forward reads (positions {int(fwd_cols[0])}–{int(fwd_cols[-1])}):
    {fwd_tail}

    Reverse reads (positions {int(rev_cols[0])}–{int(rev_cols[-1])}):
    {rev_tail}

    Answer two questions to refine the suggestion:

    1. Trend: Is quality declining steadily in the positions approaching each cutoff,
       or stable right up to it? If declining, truncate a few bases earlier to avoid
       including degrading data. If stable, the initial value is appropriate.

    2. Overlap allocation: trunc_len_f + trunc_len_r must be >= {min_sum}.
       If one read has more stable quality headroom than the other, shift the budget
       toward the read under more pressure. Maximise overall quality within the constraint.

    Respond ONLY with valid JSON, no markdown formatting:
    {{"trunc_len_f": N, "trunc_len_r": N, "reasoning": "..."}}"""


def compute_trim_left(
    summary: Summary,
    ref_pos: int = 20,
    within: float = 2.0,
    drop_tol: float = 3.0,
    run: int = 5,
) -> int:
    """Return the number of 5' bases to trim using a plateau-detection heuristic.

    Scans the median quality from left to right and returns the first index i
    where quality is within `within` Phred of the plateau (position ref_pos) AND
    the next `run` positions all stay above plateau - drop_tol.  Returns 0 if
    quality is already at plateau from the first position, or if the data is too
    short to apply the heuristic.
    """
    median = summary['df'].loc['50%'].to_numpy()
    if len(median) < ref_pos + run:
        return 0
    plateau = float(median[ref_pos - 1])
    for i in range(len(median) - run):
        if abs(float(median[i]) - plateau) <= within:
            if all(float(median[i + j]) >= plateau - drop_tol for j in range(run)):
                return i
    return 0


def check_suggestion(summary: Summary, trunc_len: int, label: str) -> bool:
    df = summary['df']
    positions = np.array(df.columns, dtype=int)
    idx = int(np.argmin(np.abs(positions - trunc_len)))
    actual_pos = int(positions[idx])

    q25 = float(df.loc['25%'].to_numpy()[idx])
    q50 = float(df.loc['50%'].to_numpy()[idx])

    ok = True
    if q25 < min_q_25p:
        print(f'ERROR ({label}): 25th percentile at position {actual_pos} is {q25:.1f}, below threshold {min_q_25p}', file=sys.stderr)
        ok = False
    if q50 < min_q_50p:
        print(f'ERROR ({label}): median at position {actual_pos} is {q50:.1f}, below threshold {min_q_50p}', file=sys.stderr)
        ok = False
    return ok


def main(argv: Sequence[str]) -> int:
    if len(argv) == 1:
        demux_file = 'demux_summary.qzv'
        fwd_data, rev_data = load_data_from_qzv(demux_file)
        plot_trunc_len_f = plot_trunc_len_r = None
    elif len(argv) == 2:
        demux_file = argv[1]
        if demux_file.lower().endswith('.qzv'):
            fwd_data, rev_data = load_data_from_qzv(demux_file)
            plot_trunc_len_f = plot_trunc_len_r = None
        else:
            print('Usage: extract_tail_data.py <demux_summary.qzv> OR <forward.tsv> <reverse.tsv> [<trunc_len_f> <trunc_len_r>]', file=sys.stderr)
            return 1
    elif len(argv) == 3:
        fwd_data, rev_data = load_data_from_tsv_files(argv[1], argv[2])
        plot_trunc_len_f = plot_trunc_len_r = None
    elif len(argv) == 5:
        fwd_data, rev_data = load_data_from_tsv_files(argv[1], argv[2])
        plot_trunc_len_f = int(argv[3])
        plot_trunc_len_r = int(argv[4])
    else:
        print('Usage: extract_tail_data.py <demux_summary.qzv> OR <forward.tsv> <reverse.tsv> [<trunc_len_f> <trunc_len_r>]', file=sys.stderr)
        return 1

    # Parse full summaries for both prompt and plotting
    fwd_summary = parse_summary_tsv(fwd_data)
    rev_summary = parse_summary_tsv(rev_data)

    trim_left_f = compute_trim_left(fwd_summary)
    trim_left_r = compute_trim_left(rev_summary)

    try:
        fwd_cols = compute_window_cols(fwd_summary)
        rev_cols = compute_window_cols(rev_summary)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        fail_result = {
            'pass_fail': 'fail',
            'trim_left_f': None,
            'trim_left_r': None,
            'trunc_len_f': None,
            'trunc_len_r': None,
            'reasoning': None,
        }
        with open('ai_quality_threshold_plot.json', 'w') as f:
            json.dump(fail_result, f, indent=2)
        print('Result written to ai_quality_threshold_plot.json')
        return 1

    prompt = build_prompt(fwd_summary, rev_summary, fwd_cols, rev_cols)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    result = json.loads(str(response.choices[0].message.content))
    trunc_len_f = int(result['trunc_len_f'])
    trunc_len_r = int(result['trunc_len_r'])

    if amplicon_length > 0:
        overlap_prompt = build_overlap_prompt(
            fwd_summary, rev_summary, fwd_cols, rev_cols, trunc_len_f, trunc_len_r
        )
        overlap_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": overlap_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        result = json.loads(str(overlap_response.choices[0].message.content))
        trunc_len_f = int(result['trunc_len_f'])
        trunc_len_r = int(result['trunc_len_r'])

    check_suggestion(fwd_summary, trunc_len_f, 'forward')
    check_suggestion(rev_summary, trunc_len_r, 'reverse')

    result['pass_fail'] = 'pass'
    result['trim_left_f'] = trim_left_f
    result['trim_left_r'] = trim_left_r

    plot_len_f = plot_trunc_len_f if plot_trunc_len_f is not None else trunc_len_f
    plot_len_r = plot_trunc_len_r if plot_trunc_len_r is not None else trunc_len_r

    plot_path = build_plot(fwd_summary, rev_summary, plot_len_f, plot_len_r, result['reasoning'],
                           fwd_window_start=int(fwd_cols[0]), rev_window_start=int(rev_cols[0]),
                           trim_left_f=trim_left_f, trim_left_r=trim_left_r)

    json_path = plot_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"trim_left_f: {trim_left_f}")
    print(f"trim_left_r: {trim_left_r}")
    print(f"trunc_len_f: {result['trunc_len_f']}")
    print(f"trunc_len_r: {result['trunc_len_r']}")
    print(f"reasoning: {result['reasoning']}")
    print(f'Plot written to {plot_path}')
    print(f'Result written to {json_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
