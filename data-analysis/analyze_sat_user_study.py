import csv
import math
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SURVEY_CSV = os.path.join(
    BASE_DIR, "SAT User Study (Responses) - Form Responses 1.csv"
)
GROUP_MAP_CSV = os.path.join(BASE_DIR, "email_username_group_flag.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


GROUPS = ["control", "intervention", "placebo"]


@dataclass
class QuestionStats:
    question: str
    column_index: int
    n_total: int
    group_ns: Dict[str, int]
    group_means: Dict[str, float]
    group_stds: Dict[str, float]
    f_stat: float
    p_perm: float
    eta_sq: float


def load_group_map() -> Dict[str, str]:
    """
    Return a mapping email -> group (control/intervention/placebo),
    ignoring entries without an assigned group.
    """
    email_to_group: Dict[str, str] = {}
    with open(GROUP_MAP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = (row.get("email") or "").strip().lower()
            group = (row.get("group") or "").strip().lower()
            if not email or not group:
                continue
            if group not in GROUPS:
                continue
            email_to_group[email] = group
    return email_to_group


def load_survey(email_to_group: Dict[str, str]) -> Tuple[List[str], List[Dict]]:
    """
    Load survey responses and attach group labels based on email.
    Returns (header, rows_with_group).
    Each row is a dict with keys from header plus 'group'.
    """
    rows = []
    with open(SURVEY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Identify email column (we assume the second column, but fall back to name search)
        email_idx = 1
        for i, name in enumerate(header):
            if "ایمیل" in name or "email" in name.lower():
                email_idx = i
                break

        for row in reader:
            if not row:
                continue
            # Ensure row has full length
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            email = (row[email_idx] or "").strip().lower()
            group = email_to_group.get(email)
            if not group:
                continue
            row_dict = {h: row[i] if i < len(row) else "" for i, h in enumerate(header)}
            row_dict["group"] = group
            rows.append(row_dict)

    return header, rows


def detect_numeric_columns(header: List[str], rows: List[Dict]) -> List[int]:
    """
    Heuristically detect which columns are numeric Likert-style questions.
    We treat a column as numeric if:
      - it's not timestamp/email
      - >= 70% of non-empty entries can be cast to float.
    Returns list of column indices.
    """
    name_to_idx = {name: i for i, name in enumerate(header)}

    # Columns to explicitly skip by (partial) name
    skip_keywords = [
        "Timestamp",
        "زمان",
        "ایمیل",
        "email",
        "آدرس ایمیل",
        "نقاط قوت",
        "نقاط قابل بهبود",
        "نکاتی",
        "تمرینی",
        "تمرین",
        "هر مورد دیگری",
        "کدام تمرین",
    ]

    numeric_cols: List[int] = []
    for idx, col in enumerate(header):
        name = col or ""
        # Skip obvious non-numeric fields
        if any(kw.lower() in name.lower() for kw in skip_keywords):
            continue

        values = []
        for r in rows:
            v = r.get(col, "").strip()
            if not v:
                continue
            values.append(v)

        if not values:
            continue

        n_valid = 0
        for v in values:
            try:
                float(v.replace(",", "."))
                n_valid += 1
            except Exception:
                continue

        if n_valid / len(values) >= 0.7:
            numeric_cols.append(idx)

    return numeric_cols


def compute_anova_with_permutation(
    values: List[float],
    groups: List[str],
    n_permutations: int = 5000,
) -> Tuple[float, float, float]:
    """
    One-way ANOVA F-statistic with permutation test p-value.
    values: flattened numeric responses
    groups: parallel list of group labels
    """
    x = np.array(values, dtype=float)
    g = np.array(groups)
    unique_groups = sorted({gg for gg in g if gg in GROUPS})

    if len(unique_groups) < 2:
        return math.nan, math.nan, math.nan

    # Compute observed F and eta-squared
    overall_mean = x.mean()
    k = len(unique_groups)
    N = len(x)

    group_means = {}
    group_ns = {}
    ss_between = 0.0
    ss_within = 0.0

    for gg in unique_groups:
        mask = g == gg
        xi = x[mask]
        if xi.size == 0:
            continue
        mu = xi.mean()
        group_means[gg] = mu
        group_ns[gg] = xi.size
        ss_between += xi.size * (mu - overall_mean) ** 2
        ss_within += ((xi - mu) ** 2).sum()

    df_between = k - 1
    df_within = N - k
    if df_within <= 0 or ss_within == 0:
        return math.nan, math.nan, math.nan

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_obs = ms_between / ms_within
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else math.nan

    # Permutation test
    rng = np.random.default_rng(seed=42)
    count = 0
    for _ in range(n_permutations):
        perm_g = rng.permutation(g)
        ss_between_perm = 0.0
        ss_within_perm = 0.0
        for gg in unique_groups:
            mask = perm_g == gg
            xi = x[mask]
            if xi.size == 0:
                continue
            mu = xi.mean()
            ss_between_perm += xi.size * (mu - overall_mean) ** 2
            ss_within_perm += ((xi - mu) ** 2).sum()
        if ss_within_perm == 0:
            continue
        ms_between_perm = ss_between_perm / df_between
        ms_within_perm = ss_within_perm / df_within
        f_perm = ms_between_perm / ms_within_perm
        if f_perm >= f_obs:
            count += 1

    p_perm = (count + 1) / (n_permutations + 1)
    return f_obs, p_perm, eta_sq


def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    email_to_group = load_group_map()
    header, rows = load_survey(email_to_group)

    print(f"Loaded {len(rows)} responses with known group.")
    group_counts = Counter(r["group"] for r in rows)
    print("Group counts:", dict(group_counts))

    numeric_cols = detect_numeric_columns(header, rows)
    print(f"Detected {len(numeric_cols)} numeric question columns.")

    stats_results: List[QuestionStats] = []

    for idx in numeric_cols:
        col_name = header[idx]
        values: List[float] = []
        groups: List[str] = []
        by_group: Dict[str, List[float]] = defaultdict(list)

        for r in rows:
            g = r.get("group")
            if g not in GROUPS:
                continue
            raw = (r.get(col_name) or "").strip()
            if not raw:
                continue
            try:
                v = float(raw.replace(",", "."))
            except Exception:
                continue
            values.append(v)
            groups.append(g)
            by_group[g].append(v)

        if len(values) < 6:  # too few data points overall
            continue

        # Compute per-group stats
        group_ns = {g: len(vs) for g, vs in by_group.items()}
        group_means = {g: float(np.mean(vs)) for g, vs in by_group.items() if vs}
        group_stds = {g: float(np.std(vs, ddof=1)) for g, vs in by_group.items() if len(vs) > 1}

        f_stat, p_perm, eta_sq = compute_anova_with_permutation(values, groups)

        stats_results.append(
            QuestionStats(
                question=col_name,
                column_index=idx,
                n_total=len(values),
                group_ns=group_ns,
                group_means=group_means,
                group_stds=group_stds,
                f_stat=f_stat,
                p_perm=p_perm,
                eta_sq=eta_sq,
            )
        )

        # Plot figure for this question: boxplot by group
        fig, ax = plt.subplots(figsize=(6, 4))
        data_for_plot = [by_group.get(g, []) for g in GROUPS]
        ax.boxplot(
            data_for_plot,
            labels=GROUPS,
            showmeans=True,
        )
        ax.set_title(col_name)
        ax.set_ylabel("Score")
        fig.tight_layout()
        safe_name = f"q{idx:02d}.png"
        fig_path = os.path.join(FIG_DIR, safe_name)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    # Write stats table
    stats_csv = os.path.join(OUTPUT_DIR, "question_stats.csv")
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question",
                "column_index",
                "n_total",
                "group",
                "n_group",
                "mean",
                "std",
                "f_stat",
                "p_perm",
                "eta_sq",
            ]
        )
        for s in stats_results:
            for g in GROUPS:
                writer.writerow(
                    [
                        s.question,
                        s.column_index,
                        s.n_total,
                        g,
                        s.group_ns.get(g, 0),
                        f"{s.group_means.get(g, float('nan')):.3f}"
                        if g in s.group_means
                        else "",
                        f"{s.group_stds.get(g, float('nan')):.3f}"
                        if g in s.group_stds
                        else "",
                        f"{s.f_stat:.3f}" if not math.isnan(s.f_stat) else "",
                        f"{s.p_perm:.4f}" if not math.isnan(s.p_perm) else "",
                        f"{s.eta_sq:.3f}" if not math.isnan(s.eta_sq) else "",
                    ]
                )

    # Also write a simple text summary for quick inspection
    summary_txt = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Group counts:\n")
        for g in GROUPS:
            f.write(f"  {g}: {group_counts.get(g, 0)}\n")
        f.write("\nTop questions by effect size (eta_sq):\n")
        for s in sorted(stats_results, key=lambda s: (-(s.eta_sq or 0.0))):
            f.write(
                f"- {s.question} (col {s.column_index}): "
                f"F={s.f_stat:.3f}, p_perm={s.p_perm:.4f}, eta_sq={s.eta_sq:.3f}\n"
            )

    print(f"Wrote stats to {stats_csv}")
    print(f"Figures saved to {FIG_DIR}")
    print(f"Summary written to {summary_txt}")


if __name__ == "__main__":
    analyze()


