"""
SPC ENGINE FOR VACCINATION CAMPAIGNS
------------------------------------

This module implements an SPC-style methodology for monitoring vaccination
campaigns in a way that respects saturation curves.

High-level steps
----------------
1. Preprocess input data (Autumn + Spring campaigns, or any similar structure).
2. Compute the weekly vaccination rate:
      weekly_rate = new_vacc / remaining_unvaccinated_last_week
3. Build an expected weekly_rate curve and natural variation (sigma) using
   Wales-wide (or system-wide) averages across regions for each cohort × group × week.
4. Compute SPC components:
      - z_scores
      - control limits (UCL/LCL = expected ± 3σ)
      - status labels (GREEN / YELLOW / RED)
5. Apply SPC run rules to detect non-random patterns:
      - Rule 1: 8 consecutive weeks below expected
      - Rule 2: 2 of 3 weeks with z < -2
      - Rule 3: 6-week monotonic trend (all up or all down)
6. Build an aggregated dashboard table per Region × Cohort × Group.

This file separates:
  - CONFIG: column names, run rule parameters, etc.
  - LOGIC: pure functions that operate on a pandas DataFrame.

You can:
  - Import `run_spc_pipeline` from this file in other scripts.
  - Use `plot_spc` to visualise a single Region × Cohort × Group.
  - Adjust config without touching the core logic.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


# ============================================================
# CONFIGURATION (SEPARATED FROM LOGIC)
# ============================================================

@dataclass
class SPCConfig:
    # Column names
    region_col: str = "Region"
    group_col: str = "group"
    cohort_col: str = "cohort"
    week_col: str = "week_end"
    vacc_col: str = "vacc_alive"
    denom_col: str = "denom"

    # Run rule parameters
    rule1_window: int = 8   # 8 consecutive points below expected
    rule2_window: int = 3   # 2 of last 3 with z < -2
    rule3_window: int = 6   # 6 points monotonic trend

    # SPC thresholds
    z_rule2_threshold: float = -2.0
    clip_rate_min: float = 0.0
    clip_rate_max: float = 1.0


# ============================================================
# CORE HELPERS
# ============================================================

def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise a clear error if any required column is missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_data(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """
    Basic preprocessing:
    - ensure datetime for week_end
    - sort rows
    - drop exact duplicates
    - sanity checks on denominators and vaccinated counts
    """
    df = df.copy()

    # Ensure datetime
    df[cfg.week_col] = pd.to_datetime(df[cfg.week_col])

    # Sort
    df = df.sort_values(
        [cfg.region_col, cfg.group_col, cfg.cohort_col, cfg.week_col]
    )

    # Drop exact duplicates for safety
    df = df.drop_duplicates(
        subset=[cfg.region_col, cfg.group_col, cfg.cohort_col, cfg.week_col]
    )

    # Basic sanity checks (non-fatal: we just warn via assertions)
    assert (df[cfg.denom_col] >= df[cfg.vacc_col]).all(), \
        "Found rows where vacc_alive > denom. Check input data."

    return df


# ============================================================
# STEP 1: WEEKLY RATE
# ============================================================

def compute_weekly_rate(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    df = df.copy()

    group_keys = [cfg.region_col, cfg.group_col, cfg.cohort_col]

    # new_vacc
    df["new_vacc"] = df.groupby(group_keys)[cfg.vacc_col].diff()
    df["new_vacc"] = df["new_vacc"].fillna(df[cfg.vacc_col])

    # Build composite key
    df["__grp_key"] = (
        df[cfg.region_col].astype(str) + "|" +
        df[cfg.group_col].astype(str)  + "|" +
        df[cfg.cohort_col].astype(str)
    )

    # remaining_prev
    df["remaining_prev"] = (
        (df[cfg.denom_col] - df[cfg.vacc_col])
        .groupby(df["__grp_key"])
        .shift(1)
    )

    # First row per group
    first_idx = df.groupby("__grp_key").head(1).index
    df.loc[first_idx, "remaining_prev"] = (
        df.loc[first_idx, cfg.denom_col] - df.loc[first_idx, cfg.vacc_col]
    )

    # weekly_rate
    df["weekly_rate"] = df["new_vacc"] / df["remaining_prev"]
    df["weekly_rate"] = df["weekly_rate"].clip(cfg.clip_rate_min, cfg.clip_rate_max)

    # week_num
    df["week_num"] = df.groupby("__grp_key").cumcount() + 1

    # cleanup
    df = df.drop(columns="__grp_key")

    return df



# ============================================================
# STEP 2: EXPECTED CURVE + SIGMA (ACROSS REGIONS)
# ============================================================

def compute_expected_and_sigma(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """
    For each cohort × group × week_num:
      - expected_rate = mean weekly_rate across regions
      - sigma_rate    = std deviation of weekly_rate across regions
    This builds a system-wide baseline + natural variation.
    """
    df = df.copy()

    keys = [cfg.group_col, "week_num"]

    expected = (
        df.groupby(keys)["weekly_rate"]
          .mean()
          .reset_index()
          .rename(columns={"weekly_rate": "expected_rate"})
    )

    variance = (
        df.groupby(keys)["weekly_rate"]
          .std()
          .reset_index()
          .rename(columns={"weekly_rate": "sigma_rate"})
    )

    df = df.merge(expected, on=keys, how="left")
    df = df.merge(variance, on=keys, how="left")

    # Avoid zero sigma (can't compute z in that case)
    df["sigma_rate"] = df["sigma_rate"].replace(0, np.nan)

    return df


# ============================================================
# STEP 3: SPC COMPONENTS (Z, UCL, LCL, STATUS)
# ============================================================

def compute_spc_components(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """
    Compute:
      - z_score
      - UCL / LCL (±3σ)
      - status (GREEN / YELLOW / RED)
    """
    df = df.copy()

    df["z_score"] = (df["weekly_rate"] - df["expected_rate"]) / df["sigma_rate"]

    df["UCL"] = df["expected_rate"] + 3 * df["sigma_rate"]
    df["LCL"] = df["expected_rate"] - 3 * df["sigma_rate"]

    def _status(row):
        if pd.isna(row["weekly_rate"]) or pd.isna(row["expected_rate"]) or pd.isna(row["sigma_rate"]):
            return "NA"
        if row["weekly_rate"] < row["LCL"]:
            return "RED"
        elif row["weekly_rate"] < row["expected_rate"]:
            return "YELLOW"
        else:
            return "GREEN"

    df["status"] = df.apply(_status, axis=1)

    return df


# ============================================================
# STEP 4: RUN RULES
# ============================================================

def _apply_run_rules_to_group(g: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """
    Apply SPC run rules within a single Region × Group × Cohort.
    Uses:
      - weekly_rate
      - expected_rate
      - z_score
    """
    g = g.sort_values("week_num").copy()
    n = len(g)

    g["rule1"] = False
    g["rule2"] = False
    g["rule3"] = False

    # Rule 1: N1 = cfg.rule1_window consecutive points below expected
    w1 = cfg.rule1_window
    if n >= w1:
        for i in range(w1 - 1, n):
            window = g.iloc[i - w1 + 1:i + 1]
            if (window["weekly_rate"] < window["expected_rate"]).all():
                g.loc[g.index[i], "rule1"] = True

    # Rule 2: 2 of 3 points with z < threshold
    w2 = cfg.rule2_window
    if n >= w2:
        for i in range(w2 - 1, n):
            window = g.iloc[i - w2 + 1:i + 1]["z_score"]
            if (window < cfg.z_rule2_threshold).sum() >= 2:
                g.loc[g.index[i], "rule2"] = True

    # Rule 3: cfg.rule3_window points with monotonic trend
    w3 = cfg.rule3_window
    if n >= w3:
        for i in range(w3 - 1, n):
            window = g.iloc[i - w3 + 1:i + 1]["weekly_rate"]
            diffs = window.diff().dropna()
            if len(diffs) == w3 - 1:
                if (diffs > 0).all() or (diffs < 0).all():
                    g.loc[g.index[i], "rule3"] = True

    g["run_rule_alert"] = g[["rule1", "rule2", "rule3"]].any(axis=1)
    return g


def apply_run_rules(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """Apply run rules across all Region × Group × Cohort combinations."""
    group_keys = [cfg.region_col, cfg.group_col, cfg.cohort_col]

    df = (
        df.groupby(group_keys, group_keys=False)
          .apply(lambda g: _apply_run_rules_to_group(g, cfg))
          .reset_index(drop=True)
    )
    return df


# ============================================================
# STEP 5: DASHBOARD SUMMARY
# ============================================================

def build_dashboard(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """
    Build an aggregated table showing, for each Region × Cohort × Group:
      - total weeks
      - number of RED / YELLOW / GREEN weeks
      - number of run_rule alerts
    """
    group_keys = [cfg.region_col, cfg.cohort_col, cfg.group_col]

    dashboard = (
        df.groupby(group_keys)
          .agg(
              total_weeks=("week_num", "count"),
              red_weeks=("status", lambda x: (x == "RED").sum()),
              yellow_weeks=("status", lambda x: (x == "YELLOW").sum()),
              green_weeks=("status", lambda x: (x == "GREEN").sum()),
              run_rule_alerts=("run_rule_alert", lambda x: x.sum())
          )
          .reset_index()
    )
    return dashboard


# ============================================================
# STEP 6: PLOTTING
# ============================================================

def plot_spc(df: pd.DataFrame, title: str = "SPC Chart") -> None:
    """
    Visual SPC chart for a single Region × Group × Cohort subset:
      - Expected weekly_rate (line)
      - ±1σ and ±2σ bands
      - UCL/LCL
      - Colour-coded actual weekly_rate points
      - Run rule alerts highlighted
    """
    df = df.sort_values("week_num").copy()

    # Build bands
    df["upper_1sigma"] = df["expected_rate"] + df["sigma_rate"]
    df["lower_1sigma"] = df["expected_rate"] - df["sigma_rate"]
    df["upper_2sigma"] = df["expected_rate"] + 2 * df["sigma_rate"]
    df["lower_2sigma"] = df["expected_rate"] - 2 * df["sigma_rate"]

    color_map = {"GREEN": "green", "YELLOW": "orange", "RED": "red"}
    colors = df["status"].map(color_map).fillna("grey")

    plt.figure(figsize=(12, 6))

    # ±2σ band
    plt.fill_between(
        df["week_num"], df["lower_2sigma"], df["upper_2sigma"],
        color="lightgrey", alpha=0.5, label="±2σ"
    )

    # ±1σ band
    plt.fill_between(
        df["week_num"], df["lower_1sigma"], df["upper_1sigma"],
        color="darkgrey", alpha=0.4, label="±1σ"
    )

    # Expected
    plt.plot(df["week_num"], df["expected_rate"],
             color="blue", linewidth=2, label="Expected")

    # UCL/LCL
    plt.plot(df["week_num"], df["UCL"], "r--", label="UCL (+3σ)")
    plt.plot(df["week_num"], df["LCL"], "r--", label="LCL (-3σ)")

    # Actual weekly_rate points
    plt.scatter(df["week_num"], df["weekly_rate"], c=colors, s=80, label="Actual")

    # Run rule alerts (circled)
    alerts = df[df["run_rule_alert"]]
    if not alerts.empty:
        plt.scatter(
            alerts["week_num"], alerts["weekly_rate"],
            s=200, facecolors="none", edgecolors="red", linewidths=2,
            label="Run rule alert"
        )

    plt.title(title)
    plt.xlabel("Week number")
    plt.ylabel("Weekly vaccination rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def build_wales_overall(df: pd.DataFrame, cfg: SPCConfig) -> pd.DataFrame:
    """
    Build a Wales-wide timeline by aggregating all regions + groups.
    weekly_rate_Wales = (sum new_vacc) / (sum remaining_prev)

    Returns a DataFrame with:
      - week_num
      - weekly_rate
      - expected_rate
      - sigma_rate
      - status
      - run-rule alerts
      - UCL / LCL
    """

    w = df.copy()

    # Aggregate all regions+groups together per cohort+week_num
    agg = (
        w.groupby(["cohort", "week_num"])
         .agg(
             new_vacc_sum=("new_vacc", "sum"),
             remaining_prev_sum=("remaining_prev", "sum")
         )
         .reset_index()
    )

    agg["weekly_rate"] = agg["new_vacc_sum"] / agg["remaining_prev_sum"]

    # Now compute expected & sigma across ALL REGIONS (for same cohort)
    exp = (
        w.groupby(["cohort", "week_num"])["weekly_rate"]
         .mean()
         .reset_index()
         .rename(columns={"weekly_rate": "expected_rate"})
    )

    sig = (
        w.groupby(["cohort", "week_num"])["weekly_rate"]
         .std()
         .reset_index()
         .rename(columns={"weekly_rate": "sigma_rate"})
    )

    agg = agg.merge(exp, on=["cohort", "week_num"], how="left")
    agg = agg.merge(sig, on=["cohort", "week_num"], how="left")

    # Compute SPC status for Wales-wide series
    agg["UCL"] = agg["expected_rate"] + 3 * agg["sigma_rate"]
    agg["LCL"] = agg["expected_rate"] - 3 * agg["sigma_rate"]

    def _status(row):
        if pd.isna(row["weekly_rate"]) or pd.isna(row["expected_rate"]) or pd.isna(row["sigma_rate"]):
            return "NA"
        if row["weekly_rate"] < row["LCL"]:
            return "RED"
        elif row["weekly_rate"] < row["expected_rate"]:
            return "YELLOW"
        else:
            return "GREEN"

    agg["status"] = agg.apply(_status, axis=1)

    return agg



# ============================================================
# HIGH-LEVEL PIPELINE
# ============================================================

def run_spc_pipeline(raw_df: pd.DataFrame, cfg: SPCConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level orchestration:
      Input: raw_df with Region, group, cohort, week_end, vacc_alive, denom.
      Output:
        - full_df: row-level data with SPC flags
        - dashboard: aggregated summary table per Region × Cohort × Group
    """
    required_cols = [cfg.region_col, cfg.group_col, cfg.cohort_col,
                     cfg.week_col, cfg.vacc_col, cfg.denom_col]
    validate_columns(raw_df, required_cols)

    df = preprocess_data(raw_df, cfg)
    df = compute_weekly_rate(df, cfg)
    df = compute_expected_and_sigma(df, cfg)
    df = compute_spc_components(df, cfg)
    df = apply_run_rules(df, cfg)
    dashboard = build_dashboard(df, cfg)

    return df, dashboard


def fit_baseline_model(history_df: pd.DataFrame,
                       cfg: SPCConfig,
                       by_cohort: bool = False) -> pd.DataFrame:
    """
    Train baseline expected_rate and sigma_rate from historical campaigns
    (e.g. Autumn 2024 + Spring 2025).

    If by_cohort=False:
        baseline is learned per group + week_num, pooling all cohorts.
    If by_cohort=True:
        baseline is cohort-specific (cohort + group + week_num).
    """
    # Reuse core steps: preprocess + weekly_rate
    hist = preprocess_data(history_df, cfg)
    hist = compute_weekly_rate(hist, cfg)

    if by_cohort:
        keys = [cfg.cohort_col, cfg.group_col, "week_num"]
    else:
        keys = [cfg.group_col, "week_num"]

    expected = (
        hist.groupby(keys)["weekly_rate"]
            .mean()
            .reset_index()
            .rename(columns={"weekly_rate": "expected_rate"})
    )

    variance = (
        hist.groupby(keys)["weekly_rate"]
            .std()
            .reset_index()
            .rename(columns={"weekly_rate": "sigma_rate"})
    )

    baseline = expected.merge(variance, on=keys, how="left")
    baseline["sigma_rate"] = baseline["sigma_rate"].replace(0, np.nan)

    return baseline

def run_spc_with_baseline(new_df: pd.DataFrame,
                          baseline_df: pd.DataFrame,
                          cfg: SPCConfig,
                          by_cohort: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply SPC to a NEW campaign using a pre-trained baseline
    (from Autumn 2024 + Spring 2025).

    - new_df: new campaign (e.g. Autumn 2025) with Region, group, cohort, week_end, vacc_alive, denom.
    - baseline_df: output from fit_baseline_model.
    Returns:
      - full_new: row-level SPC-annotated data for the new campaign
      - dashboard_new: Region×Cohort×Group summary for new campaign
      - trend_table: latest-week trend per Region×Group×Cohort
    """
    new_proc = preprocess_data(new_df, cfg)
    new_proc = compute_weekly_rate(new_proc, cfg)

    if by_cohort:
        merge_keys = [cfg.cohort_col, cfg.group_col, "week_num"]
    else:
        merge_keys = [cfg.group_col, "week_num"]

    new_proc = new_proc.merge(baseline_df, on=merge_keys, how="left")

    # Use the same SPC components + run rules
    new_proc = compute_spc_components(new_proc, cfg)
    new_proc = apply_run_rules(new_proc, cfg)
    new_proc["deviation_label"] = new_proc.apply(classify_deviation, axis=1)
    dashboard_new = build_dashboard(new_proc, cfg)
    trend_table = build_trend_table(new_proc, cfg)

    return new_proc, dashboard_new, trend_table

def build_trend_table(df: pd.DataFrame,
                      cfg: SPCConfig,
                      cohort_name: Optional[str] = None) -> pd.DataFrame:
    """
    Build a table for the NEW campaign showing, for each Region × Group × Cohort:
      - latest week_num
      - latest weekly_rate
      - change vs previous week (Δ_week)
      - change vs 4 weeks ago (Δ_4weeks), if available
      - latest status (GREEN/YELLOW/RED)
      - whether a run_rule_alert is active in the latest week
    """
    d = df.copy()
    if cohort_name is not None:
        d = d[d[cfg.cohort_col] == cohort_name]

    rows = []
    group_keys = [cfg.region_col, cfg.group_col, cfg.cohort_col]

    for (reg, grp, coh), g in d.groupby(group_keys):
        g = g.sort_values("week_num")
        last = g.iloc[-1]

        # Δ vs last week
        if len(g) >= 2:
            prev = g.iloc[-2]
            delta_week = last["weekly_rate"] - prev["weekly_rate"]
        else:
            delta_week = np.nan

        # Δ vs 4 weeks ago (rough "month")
        if len(g) >= 5:
            prev4 = g.iloc[-5]
            delta_4weeks = last["weekly_rate"] - prev4["weekly_rate"]
        else:
            delta_4weeks = np.nan

        rows.append({
            "Region": reg,
            "Group": grp,
            "Cohort": coh,
            "latest_week_num": last["week_num"],
            "latest_week_end": last[cfg.week_col],
            "latest_weekly_rate": last["weekly_rate"],
            "delta_vs_last_week": delta_week,
            "delta_vs_4weeks": delta_4weeks,
            "latest_status": last["status"],
            "latest_run_rule_alert": bool(last["run_rule_alert"]),
        })

    trend_table = pd.DataFrame(rows)
    return trend_table.sort_values(["Cohort", "Region", "Group"])



def classify_deviation(row):
    if pd.isna(row["sigma_rate"]) or pd.isna(row["expected_rate"]) or pd.isna(row["weekly_rate"]):
        return "Insufficient data"

    exp = row["expected_rate"]
    sig = row["sigma_rate"]
    val = row["weekly_rate"]
    ucl = row["UCL"]
    lcl = row["LCL"]

    # Extreme (3σ)
    if val < lcl:
        return "Alert - extremely lower than expected"
    if val > ucl:
        return "Alert - extremely higher than expected"

    # Moderate (2σ)
    if val < exp - 2*sig:
        return "Lower than expected"
    if val > exp + 2*sig:
        return "Higher than expected"

    # Slight (1σ)
    if val < exp - 1*sig:
        return "Slightly Lower than expected"
    if val > exp + 1*sig:
        return "Slightly Higher than expected"

    return "Within expected variation"


def build_deviation_table(df: pd.DataFrame, cfg: SPCConfig):
    df = df.copy()
    cols = ["Region", "group", "cohort", "week_end",
            "weekly_rate", "expected_rate", "sigma_rate",
            "UCL", "LCL", "deviation_label"]

    return df[cols].sort_values(["cohort", "Region", "group", "week_end"])

# ============================================================
# EXAMPLE USAGE (if run as a script)
# ============================================================

if __name__ == "__main__":
    cfg = SPCConfig()

    # 1. Load historical campaigns: Autumn 2024 + Spring 2025
    aut = pd.read_csv("uptake_trend_Autumn_2024.csv")
    aut[cfg.cohort_col] = "Autumn 2024"
    aut[cfg.week_col] = pd.to_datetime(aut[cfg.week_col], format="%d%b%Y")

    spr = pd.read_csv("uptake_trend_Spring_2025.csv")
    spr[cfg.cohort_col] = "Spring 2025"
    spr[cfg.week_col] = pd.to_datetime(spr[cfg.week_col], format="%d/%m/%Y")
    spr = spr.rename(columns={"vacc": cfg.vacc_col, "Group": cfg.group_col})

    history = pd.concat([aut, spr], ignore_index=True)

    # 2. Train baseline model from Autumn + Spring
    #    Here we ignore cohort and learn baseline per group + week_num
    baseline = fit_baseline_model(history, cfg, by_cohort=False)

    # 3. Load a NEW campaign CSV (e.g. Autumn 2025)
    #    Make sure it has the same columns: Region, group, cohort, week_end, vacc_alive, denom
    new_campaign = pd.read_csv("synthetic_autumn_2025.csv")
    new_campaign[cfg.cohort_col] = "Autumn 2025"
    new_campaign[cfg.week_col] = pd.to_datetime(new_campaign[cfg.week_col])  # adjust format if needed

    # 4. Apply SPC to new campaign using the trained baseline
    full_new, dash_new, trend_new = run_spc_with_baseline(
        new_campaign,
        baseline,
        cfg,
        by_cohort=False  # using group + week_num baseline
    )

    print("\n===== NEW CAMPAIGN DASHBOARD (REGION × COHORT × GROUP) =====\n")
    print(dash_new.head())

    print("\n===== TREND TABLE (LATEST WEEK CHANGES) =====\n")
    print(trend_new.head())

    # --- WALES-WIDE SPC CHART ---
    print("\n===== WALES-WIDE SPC CHART (ALL REGIONS + ALL GROUPS) =====\n")

    wales_df = build_wales_overall(full_new, cfg)

    # Build chart
    plt.figure(figsize=(12, 6))
    plt.plot(wales_df["week_num"], wales_df["expected_rate"], label="Expected", color="blue")

    plt.fill_between(wales_df["week_num"], wales_df["expected_rate"] - wales_df["sigma_rate"],
                     wales_df["expected_rate"] + wales_df["sigma_rate"],
                     alpha=0.3, color="grey", label="±1σ")

    plt.fill_between(wales_df["week_num"], wales_df["LCL"], wales_df["UCL"],
                     alpha=0.2, color="lightgrey", label="±3σ")

    color_map = {"GREEN": "green", "YELLOW": "orange", "RED": "red"}
    colors = wales_df["status"].map(color_map).fillna("black")

    plt.scatter(wales_df["week_num"], wales_df["weekly_rate"], c=colors, s=80)

    plt.title("Wales-wide Vaccination Rate — Autumn 2025")
    plt.xlabel("Week number")
    plt.ylabel("Weekly vaccination rate (all regions + groups)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n===== AUTUMN 2025: CURRENT WEEK VS LAST WEEK VS LAST MONTH =====\n")

    trend_display = trend_new[[
        "Region", "Group", "Cohort",
        "latest_week_num", "latest_week_end",
        "latest_weekly_rate",
        "delta_vs_last_week",
        "delta_vs_4weeks",
        "latest_status",
        "latest_run_rule_alert"
    ]].sort_values(["Region","Group"])

    print(trend_display.to_string(index=False))