#!/usr/bin/env python3
"""
Compare hard-coded heuristic vs AI for detecting the 5' quality ramp-up cutoff.

Runs both methods on synthetic edge-case quality profiles and prints a side-by-side
comparison, then shows AI reasoning for any cases where the two methods disagree.
"""
import json
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Synthetic edge cases
# Each entry: (label, description, 30-element median quality array)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(42)

def _q(arr):
    return np.clip(np.round(np.asarray(arr, dtype=float), 1), 2, 40)

CASES = [
    (
        "normal_ramp_10bp",
        "Ramp 22→38 over 10 bp, plateau at 38",
        _q(np.concatenate([np.linspace(22, 38, 10), np.full(20, 38)])),
    ),
    (
        "already_high",
        "Stable at 38 from position 1 — no ramp",
        _q(np.full(30, 38.0)),
    ),
    (
        "gradual_ramp_20bp",
        "Slow rise 25→38 over 20 bp",
        _q(np.concatenate([np.linspace(25, 38, 20), np.full(10, 38)])),
    ),
    (
        "noisy_ramp_10bp",
        "10 bp ramp with ±2 Phred noise throughout",
        _q(np.concatenate([np.linspace(22, 38, 10), np.full(20, 38)])
           + _rng.normal(0, 2, 30)),
    ),
    (
        "double_dip",
        "Dip pos 1–4, brief recovery, dip pos 8–11, then plateau",
        _q([20, 22, 24, 26,       # first dip
            35, 36,               # brief recovery
            20, 21, 23, 25,       # second dip
            36, 37, 38, 38, 38,
            38, 38, 38, 38, 38,
            38, 38, 38, 38, 38,
            38, 38, 38, 38, 38]),
    ),
    (
        "late_plateau_20bp",
        "Poor quality (~15) for 20 bp, sharp jump to 38",
        _q(np.concatenate([np.full(20, 15.0), np.full(10, 38.0)])),
    ),
    (
        "single_outlier_dip",
        "High quality (38) except one dip at pos 3 — likely artifact",
        _q([38, 38, 25] + [38] * 27),
    ),
    (
        "very_short_ramp_2bp",
        "Only 2 depressed positions (25, 30), then 38 — borderline",
        _q([25, 30] + [38] * 28),
    ),
    (
        "monotone_degrading",
        "Starts at 38, degrades to 20 — no 5' ramp at all",
        _q(np.linspace(38, 20, 30)),
    ),
    (
        "stepped_ramp",
        "Two-step: pos 1–5 at 20, pos 6–10 at 30, pos 11+ at 38",
        _q(np.concatenate([np.full(5, 20.0), np.full(5, 30.0), np.full(20, 38.0)])),
    ),
]


# ---------------------------------------------------------------------------
# Hard-coded heuristic
# ---------------------------------------------------------------------------

def heuristic_trim_left(
    median_quals: np.ndarray,
    ref_pos: int = 20,
    within: float = 2.0,
    drop_tol: float = 3.0,
    run: int = 5,
) -> int:
    """Return number of 5' bases to trim (0-indexed).

    Scans left to right; returns the first index i where quality is within
    `within` Phred of the plateau at ref_pos AND the next `run` positions all
    stay above (plateau - drop_tol).  Returns 0 if stable from position 1.
    """
    plateau = float(median_quals[ref_pos - 1])
    for i in range(len(median_quals) - run):
        if abs(float(median_quals[i]) - plateau) <= within:
            if all(float(median_quals[i + j]) >= plateau - drop_tol for j in range(run)):
                return i
    return 0


# ---------------------------------------------------------------------------
# AI route
# ---------------------------------------------------------------------------

PROMPT = """\
Determine whether a 5' quality ramp exists, then find where it ends.

Step 1 — Does a ramp exist?
  Compare the median at positions 1-5 to the plateau (position 20 value).
  If ALL of positions 1-5 are within 2 Phred of position 20: no ramp -> trim_left = 0. STOP.
  Otherwise: a ramp is present. Proceed to step 2.

Step 2 — Where does the ramp end?
  Scan left to right. Return the first position P (1-based) where:
    - median quality at P is within 2 Phred of position 20, AND
    - quality does not drop below (plateau - 3) for the next 5 positions.
  trim_left = P - 1.

NOTE: This is shape detection, not quality filtering. A read starting at Phred 25
and rising to 38 has a ramp even though 25 is within an acceptable quality range.
Identify where the profile stabilizes, not where it becomes "good enough."

Median quality at positions 1-{n}:
{header}
{values}

Respond ONLY with valid JSON, no markdown:
{{"trim_left": N, "reasoning": "..."}}"""


def ai_trim_left(median_quals: np.ndarray) -> tuple[int, str]:
    n = len(median_quals)
    header = "pos\t" + "\t".join(str(i + 1) for i in range(n))
    values = "med\t" + "\t".join(f"{v:.1f}" for v in median_quals)
    prompt = PROMPT.format(n=n, header=header, values=values)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    result = json.loads(str(response.choices[0].message.content))
    return int(result["trim_left"]), str(result["reasoning"])


# ---------------------------------------------------------------------------
# Run and display
# ---------------------------------------------------------------------------

def main():
    label_w   = 22
    desc_w    = 44
    profile_w = 30

    header_line = (
        f"{'Case':<{label_w}}  "
        f"{'Description':<{desc_w}}  "
        f"{'Positions 1-10':<{profile_w}}  "
        f"{'H':>3}  {'AI':>3}  {'':4}"
    )
    print(header_line)
    print("-" * len(header_line))

    disagreements = []

    for label, description, median_quals in CASES:
        h = heuristic_trim_left(median_quals)
        ai_val, reasoning = ai_trim_left(median_quals)

        first10 = "  ".join(f"{v:4.1f}" for v in median_quals[:10])
        agree = "✓" if h == ai_val else "✗ <<<"

        print(
            f"{label:<{label_w}}  "
            f"{description:<{desc_w}}  "
            f"{first10:<{profile_w}}  "
            f"{h:>3}  {ai_val:>3}  {agree}"
        )

        if h != ai_val:
            disagreements.append((label, h, ai_val, reasoning))

    print()
    if disagreements:
        print("=" * 70)
        print("DISAGREEMENTS — AI reasoning:\n")
        for label, h, ai_val, reasoning in disagreements:
            print(f"  [{label}]  heuristic={h}  AI={ai_val}")
            print(f"  {reasoning}")
            print()
    else:
        print("All cases agree.")


if __name__ == "__main__":
    main()
