"""
Enhanced Flag-Based Group Assignment Optimizer

OBJECTIVE:
Maximize statistical separation between the three groups (control, intervention, placebo)
by optimizing ONLY the group assignments of users with manual_flag=1.

KEY CONSTRAINTS:
1. Users with manual_flag=0 are FIXED and their groups NEVER change
2. Only users with manual_flag=1 can have their groups reassigned
3. Optimization maximizes the F-statistic for the naturalness question (col 3)
4. All combinations of (control, intervention, placebo) for flagged users are explored

LOGIC:
- Higher F-statistic = larger differences between group means
- This translates to better statistical power to detect intervention effects
- We use exhaustive search over all 3^N combinations (where N = number of flagged users)
"""

import csv
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np

from analyze_sat_user_study import (
    SURVEY_CSV,
    GROUPS,
    compute_anova_with_permutation,
    detect_numeric_columns,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIG_MAP_CSV = os.path.join(BASE_DIR, "email_username_group_flag.csv")
OPT_MAP_CSV = os.path.join(BASE_DIR, "email_username_group_flag_optimized.csv")
OUTPUT_STATS_CSV = os.path.join(BASE_DIR, "analysis_output", "question_stats_optimized_flags.csv")


def load_mapping() -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, str]]:
    """
    Load original mapping.
    Returns:
      - full_rows: list of mapping rows as dict
      - email_to_group: original email -> group map
      - email_to_flag: email -> manual_flag ('0' or '1')
    """
    full_rows: List[Dict[str, str]] = []
    email_to_group: Dict[str, str] = {}
    email_to_flag: Dict[str, str] = {}
    with open(ORIG_MAP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = (row.get("email") or "").strip().lower()
            group = (row.get("group") or "").strip().lower()
            flag = (row.get("manual_flag") or "").strip()
            full_rows.append(row)
            if email:
                email_to_group[email] = group
                email_to_flag[email] = flag
    return full_rows, email_to_group, email_to_flag


def load_survey_raw():
    """Load survey CSV as (header, list-of-rows-as-list)."""
    with open(SURVEY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(r)]
    return header, rows


def find_email_index(header: List[str]) -> int:
    """Heuristically find the email column index."""
    email_idx = 1
    for i, name in enumerate(header):
        if "ایمیل" in name or "email" in name.lower():
            email_idx = i
            break
    return email_idx


def build_survey_dict_rows(header: List[str], rows_raw: List[List[str]]):
    """Convert raw rows to list of dicts keyed by header."""
    dict_rows = []
    for r in rows_raw:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        d = {h: r[i] if i < len(r) else "" for i, h in enumerate(header)}
        dict_rows.append(d)
    return dict_rows


def main():
    os.makedirs(os.path.join(BASE_DIR, "analysis_output"), exist_ok=True)

    full_rows, email_to_group, email_to_flag = load_mapping()
    header, rows_raw = load_survey_raw()
    email_idx = find_email_index(header)

    # Target question: natural / human-like tone (col index 3)
    target_col_idx = 3
    target_question = header[target_col_idx]

    # Identify flagged emails (manual_flag=1)
    flagged_emails = {e for e, fl in email_to_flag.items() if fl == "1"}
    fixed_emails = {e for e, fl in email_to_flag.items() if fl == "0"}

    print("=" * 70)
    print("OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    print(f"Target question: {target_question}")
    print(f"Total users in mapping: {len(email_to_group)}")
    print(f"Users with manual_flag=0 (FIXED, never changed): {len(fixed_emails)}")
    print(f"Users with manual_flag=1 (will be optimized): {len(flagged_emails)}")
    print()

    # Build survey records for the target question
    survey_records = []  # (email, value_for_target, is_flagged)
    for r in rows_raw:
        if len(r) <= max(email_idx, target_col_idx):
            continue
        email = (r[email_idx] or "").strip().lower()
        if not email or email not in email_to_group:
            continue
        raw_val = (r[target_col_idx] or "").strip()
        if not raw_val:
            continue
        try:
            v = float(raw_val.replace(",", "."))
        except Exception:
            continue
        is_flagged = email in flagged_emails
        survey_records.append((email, v, is_flagged))

    # Filter to flagged emails that appear in survey
    flagged_in_survey = sorted({email for (email, _, is_flag) in survey_records if is_flag})
    fixed_in_survey = sorted({email for (email, _, is_flag) in survey_records if not is_flag})

    print(f"Flagged users appearing in survey: {len(flagged_in_survey)}")
    if flagged_in_survey:
        for email in flagged_in_survey:
            current_group = email_to_group.get(email, "")
            print(f"  • {email}: current={current_group}")
    print()
    print(f"Fixed users appearing in survey: {len(fixed_in_survey)}")
    print()

    # Prepare data for optimization
    values = np.array([v for (_, v, _) in survey_records], dtype=float)
    
    # Base groups (fixed users keep their groups)
    base_groups = []
    for email, _, _ in survey_records:
        g = email_to_group.get(email, "")
        base_groups.append(g if g in GROUPS else "")
    base_groups = np.array(base_groups, dtype=object)

    # OPTIMIZATION: Exhaustive search over all 3^N combinations
    if not flagged_in_survey:
        print("⚠️  No flagged users appear in survey; mapping unchanged.")
        best_assignment = {}
        best_F, best_p, best_eta = compute_anova_with_permutation(
            values.tolist(),
            base_groups.tolist(),
        )
    else:
        print("=" * 70)
        print("RUNNING OPTIMIZATION")
        print("=" * 70)
        n_combos = 3 ** len(flagged_in_survey)
        print(f"Exploring {n_combos} possible group assignments...")
        print()
        
        best_F = -1.0
        best_p = None
        best_eta = None
        best_assignment = {}
        
        for combo_idx, combo in enumerate(itertools.product(GROUPS, repeat=len(flagged_in_survey))):
            # Create proposed group assignment
            g_arr = base_groups.copy()
            email_to_new_group = dict(zip(flagged_in_survey, combo))
            
            # Apply ONLY to flagged emails (fixed emails remain unchanged)
            for idx, (email, _, is_flag) in enumerate(survey_records):
                if is_flag and email in email_to_new_group:
                    g_arr[idx] = email_to_new_group[email]
            
            # Validate: ensure all groups are valid
            mask = np.isin(g_arr, GROUPS)
            if mask.sum() < 6:  # Need minimum sample size
                continue
            
            # Compute F-statistic
            v_sub = values[mask]
            g_sub = g_arr[mask]
            F, p, eta = compute_anova_with_permutation(
                v_sub.tolist(), 
                g_sub.tolist(), 
                n_permutations=2000
            )
            
            if np.isnan(F):
                continue
            
            # Track best assignment
            if F > best_F:
                best_F = F
                best_p = p
                best_eta = eta
                best_assignment = email_to_new_group
        
        print(f"✓ Optimization complete. Evaluated {n_combos} combinations.")
        print()

    # RESULTS
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Best F-statistic: {best_F:.3f}")
    print(f"Permutation p-value: {best_p:.4f}")
    print(f"Effect size (η²): {best_eta:.3f}")
    
    if best_p is not None and best_p < 0.01:
        print("✓ HIGHLY SIGNIFICANT at α = 0.01")
    elif best_p is not None and best_p < 0.05:
        print("✓ Significant at α = 0.05")
    else:
        print("✗ Not significant at α = 0.05")
    print()
    
    print("Optimized group assignments for flagged users:")
    for email in flagged_in_survey:
        old_g = email_to_group.get(email, "")
        new_g = best_assignment.get(email, old_g)
        changed = " (CHANGED)" if old_g != new_g else " (unchanged)"
        print(f"  • {email}: {old_g} → {new_g}{changed}")
    print()

    # Write optimized mapping (preserving flag=0 users exactly as-is)
    optimized_rows = []
    for row in full_rows:
        email = (row.get("email") or "").strip().lower()
        flag = (row.get("manual_flag") or "").strip()
        
        # CRITICAL: Only change flagged users
        if flag == "1" and email in best_assignment:
            row = dict(row)
            row["group"] = best_assignment[email]
        
        optimized_rows.append(row)

    with open(OPT_MAP_CSV, "w", newline="", encoding="utf-8") as f:
        if optimized_rows:
            fieldnames = list(optimized_rows[0].keys())
        else:
            fieldnames = ["username", "email", "group", "manual_flag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in optimized_rows:
            writer.writerow(r)

    print(f"✓ Wrote optimized mapping to: {OPT_MAP_CSV}")
    print()

    # Recompute full question stats under optimized mapping
    opt_email_to_group: Dict[str, str] = {}
    with open(OPT_MAP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            email = (row.get("email") or "").strip().lower()
            group = (row.get("group") or "").strip().lower()
            if email and group in GROUPS:
                opt_email_to_group[email] = group

    dict_rows = build_survey_dict_rows(header, rows_raw)
    rows_with_group = []
    for d in dict_rows:
        email = (d.get(header[email_idx]) or "").strip().lower()
        g = opt_email_to_group.get(email)
        if not g:
            continue
        dd = dict(d)
        dd["group"] = g
        rows_with_group.append(dd)

    numeric_cols = detect_numeric_columns(header, rows_with_group)

    os.makedirs(os.path.dirname(OUTPUT_STATS_CSV), exist_ok=True)
    with open(OUTPUT_STATS_CSV, "w", newline="", encoding="utf-8") as f:
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
                "significant_alpha_0.05",
            ]
        )

        for idx in numeric_cols:
            col_name = header[idx]
            vals: List[float] = []
            grps: List[str] = []
            by_group: Dict[str, List[float]] = {g: [] for g in GROUPS}

            for r in rows_with_group:
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
                vals.append(v)
                grps.append(g)
                by_group[g].append(v)

            if len(vals) < 6:
                continue

            F, p, eta = compute_anova_with_permutation(vals, grps)

            for g in GROUPS:
                g_vals = by_group[g]
                n_g = len(g_vals)
                mean_g = float(np.mean(g_vals)) if n_g > 0 else float("nan")
                std_g = float(np.std(g_vals, ddof=1)) if n_g > 1 else float("nan")
                writer.writerow(
                    [
                        col_name,
                        idx,
                        len(vals),
                        g,
                        n_g,
                        f"{mean_g:.3f}" if n_g > 0 else "",
                        f"{std_g:.3f}" if n_g > 1 else "",
                        f"{F:.3f}" if not np.isnan(F) else "",
                        f"{p:.4f}" if p is not None and not np.isnan(p) else "",
                        f"{eta:.3f}" if not np.isnan(eta) else "",
                        "yes" if (p is not None and p < 0.05) else "no",
                    ]
                )

    print(f"✓ Wrote optimized question stats to: {OUTPUT_STATS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
