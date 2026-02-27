import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
DATA_DIR = REPO / "data"

C_LOCAL = 0.90  # default gating threshold


def load_data():
    """Load per-sample results and unified dataset"""
    print("Loading data...")

    per_sample_path = RESULTS_DIR / "full_pipeline_per_sample.csv"
    per_sample_df = pd.read_csv(per_sample_path)
    print(f"  Loaded {len(per_sample_df)} samples from full_pipeline_per_sample.csv")

    sentinel_path = RESULTS_DIR / "sentinel_per_sample.csv"
    sentinel_df = pd.read_csv(sentinel_path)
    print(f"  Loaded {len(sentinel_df)} samples from sentinel_per_sample.csv")

    dataset_path = DATA_DIR / "unified_dataset.csv"
    dataset_df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(dataset_df)} samples from unified_dataset.csv")

    # Merge per_sample with sentinel (use sentinel_pred as Stage 4)
    merged = per_sample_df.merge(
        sentinel_df[['idx', 'sentinel_pred', 'sentinel_confidence']],
        on='idx',
        how='left',
        suffixes=('', '_sentinel')
    )

    # Merge source info from unified dataset
    merged = merged.merge(
        dataset_df[['text', 'source']],
        left_on='idx',
        right_index=True,
        how='left'
    )

    if 'source_x' in merged.columns:
        merged['source'] = merged['source_x']
        merged = merged.drop(columns=['source_x', 'source_y'])

    print(f"  Final merged dataset: {len(merged)} samples")
    print("\n  Source distribution:")
    print(merged['source'].value_counts())

    return merged


def bootstrap_ci(y_true, y_pred, metric_func, n_iterations=500, ci=95):
    """Compute bootstrap confidence interval for a metric"""
    n = len(y_true)
    metrics = []
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        metric_val = metric_func(y_true[indices], y_pred[indices])
        metrics.append(metric_val)
    metrics = np.array(metrics)
    lower = np.percentile(metrics, (100 - ci) / 2)
    upper = np.percentile(metrics, 100 - (100 - ci) / 2)
    return float(np.mean(metrics)), float(lower), float(upper)


def gated_prediction(row, stage4_col, c_local):
    """
    Gated aggregation: start with Stage 4, override with local
    (Stages 1-3) only when local is confident AND predicts attack.

    Logic:
      if stage4 == 1:  final = 1  (Stage 4 says attack)
      elif full_pred == 1 AND full_conf >= c_local:  final = 1  (local override)
      else:  final = stage4  (defer to Stage 4)
    """
    if row[stage4_col] == 1:
        return 1
    elif row['full_pred'] == 1 and row['full_conf'] >= c_local:
        return 1
    else:
        return int(row[stage4_col])


def compute_metrics(y_true, y_pred):
    """Compute F1, precision, recall, accuracy"""
    return {
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'accuracy': float((y_pred == y_true).sum() / len(y_true))
    }


def run_experiment(merged_df):
    """
    Cross-source seeding experiment.

    Three conditions, all using gated aggregation:
      1. Cold-start: Stage 4 only (no local overrides) — upper bound
      2. Cross-source: Stage 4 + Stage 1 overrides only (no Stage 2 memory
         from evaluation source) — simulates seeding from disjoint sources
      3. Same-source: Stage 4 + full local (S1+S2+S3) overrides at c_local
         — represents the actual pipeline where memory was seeded from the
         same source as the evaluation set

    If same-source contamination exists, Condition 3 should be noticeably
    worse than Condition 2. The delta quantifies the bias.
    """
    print("\n" + "=" * 80)
    print("CROSS-SOURCE SEEDING EXPERIMENT")
    print("=" * 80)
    print(f"""
METHODOLOGY (corrected):
  All three conditions use gated aggregation with c_local >= {C_LOCAL}.
  - Condition 1: Cold-start  = Stage 4 only (no local overrides)
  - Condition 2: Cross-source = Stage 4 + Stage 1 overrides at c >= {C_LOCAL}
                  (simulates memory seeded from disjoint sources)
  - Condition 3: Same-source  = Stage 4 + Stages 1-3 overrides at c >= {C_LOCAL}
                  (actual pipeline; memory seeded from same source)
""")

    # Use stage4_pred from the full_pipeline CSV (original GPT-4o-mini judge)
    # If sentinel is available, we could also test with that, but the paper's
    # cross-source analysis uses the original Stage 4.
    stage4_col = 'stage4_pred'

    # Filter to hf_safeguard samples
    safeguard_df = merged_df[merged_df['source'] == 'hf_safeguard_prompt_injection'].copy()
    print(f"Evaluating on {len(safeguard_df)} hf_safeguard samples")

    y_true = safeguard_df['label'].values

    # --- Condition 1: Cold-start (Stage 4 only) ---
    y_pred_cold = safeguard_df[stage4_col].values

    # --- Condition 2: Cross-source (S4 + S1-only overrides) ---
    # Stage 1 is purely heuristic, not seeded from any source, so it is
    # source-independent. We allow S1 overrides when stage1_conf >= c_local.
    safeguard_df['cross_pred'] = safeguard_df.apply(
        lambda r: 1 if r[stage4_col] == 1
        else (1 if r['stage1_pred'] == 1 and r['stage1_conf'] >= C_LOCAL
              else int(r[stage4_col])),
        axis=1
    )
    y_pred_cross = safeguard_df['cross_pred'].values
    n_overrides_cross = int(((y_pred_cross == 1) & (safeguard_df[stage4_col].values == 0)).sum())

    # --- Condition 3: Same-source (S4 + full local overrides) ---
    # This is the actual gated pipeline: local = Stages 1-3 combined.
    safeguard_df['same_pred'] = safeguard_df.apply(
        lambda r: gated_prediction(r, stage4_col, C_LOCAL), axis=1
    )
    y_pred_same = safeguard_df['same_pred'].values
    n_overrides_same = int(((y_pred_same == 1) & (safeguard_df[stage4_col].values == 0)).sum())

    # Compute metrics
    m_cold = compute_metrics(y_true, y_pred_cold)
    m_cross = compute_metrics(y_true, y_pred_cross)
    m_same = compute_metrics(y_true, y_pred_same)

    # Bootstrap CIs
    ci_cold_mean, ci_cold_lo, ci_cold_hi = bootstrap_ci(y_true, y_pred_cold, f1_score)
    ci_cross_mean, ci_cross_lo, ci_cross_hi = bootstrap_ci(y_true, y_pred_cross, f1_score)
    ci_same_mean, ci_same_lo, ci_same_hi = bootstrap_ci(y_true, y_pred_same, f1_score)

    # Delta: same-source vs cross-source
    delta_f1 = m_same['f1'] - m_cross['f1']

    # Bootstrap delta CI
    np.random.seed(42)
    n = len(y_true)
    deltas = []
    for _ in range(500):
        idx = np.random.choice(n, size=n, replace=True)
        f1_s = f1_score(y_true[idx], y_pred_same[idx], zero_division=0)
        f1_c = f1_score(y_true[idx], y_pred_cross[idx], zero_division=0)
        deltas.append(f1_s - f1_c)
    deltas = np.array(deltas)
    delta_ci_lo = float(np.percentile(deltas, 2.5))
    delta_ci_hi = float(np.percentile(deltas, 97.5))

    # Print results
    for label, m, ci_lo, ci_hi, overrides in [
        ("Cold-start (Stage 4 only)", m_cold, ci_cold_lo, ci_cold_hi, 0),
        ("Cross-source (S4 + S1 overrides)", m_cross, ci_cross_lo, ci_cross_hi, n_overrides_cross),
        ("Same-source (S4 + S1-S3 overrides)", m_same, ci_same_lo, ci_same_hi, n_overrides_same),
    ]:
        print(f"\n  {label}:")
        print(f"    F1={m['f1']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  overrides={overrides}")

    print(f"\n  Same vs Cross ΔF1 = {delta_f1:.4f}  [{delta_ci_lo:.4f}, {delta_ci_hi:.4f}]")
    ci_includes_zero = delta_ci_lo <= 0 <= delta_ci_hi
    print(f"  CI includes zero: {ci_includes_zero}")

    if abs(delta_f1) < 0.01:
        interpretation = (
            f"Negligible difference (ΔF1={delta_f1:.4f}) between same-source and "
            f"cross-source seeding. No evidence of contamination bias."
        )
    else:
        interpretation = (
            f"Measurable difference (ΔF1={delta_f1:.4f}) detected. "
            f"Further investigation of memory seeding recommended."
        )
    print(f"\n  Interpretation: {interpretation}")

    return {
        'experiment': 'cross_source_seeding',
        'eval_set': 'hf_safeguard',
        'n_samples': len(safeguard_df),
        'c_local': C_LOCAL,
        'conditions': {
            'cold_start': {
                'description': 'Stage 4 only (no seeding, no local overrides)',
                'f1': m_cold['f1'],
                'f1_ci': [ci_cold_lo, ci_cold_hi],
                'precision': m_cold['precision'],
                'recall': m_cold['recall'],
            },
            'cross_source': {
                'description': f'Stage 4 + Stage 1 overrides only (seed-independent) at c>={C_LOCAL}',
                'f1': m_cross['f1'],
                'f1_ci': [ci_cross_lo, ci_cross_hi],
                'precision': m_cross['precision'],
                'recall': m_cross['recall'],
                'overrides': n_overrides_cross,
            },
            'same_source': {
                'description': f'Full gated pipeline (S4 + S1+S2+S3 overrides at c_local>={C_LOCAL})',
                'f1': m_same['f1'],
                'f1_ci': [ci_same_lo, ci_same_hi],
                'precision': m_same['precision'],
                'recall': m_same['recall'],
                'overrides': n_overrides_same,
            },
        },
        'deltas': {
            'same_vs_cross_f1': float(delta_f1),
            'same_vs_cross_ci': [delta_ci_lo, delta_ci_hi],
            'ci_includes_zero': bool(ci_includes_zero),
            'cross_vs_cold_f1': float(m_cross['f1'] - m_cold['f1']),
        },
        'interpretation': interpretation,
    }


def main():
    print(f"Repository: {REPO}")
    merged_df = load_data()
    results = run_experiment(merged_df)

    output_path = RESULTS_DIR / "experiment_crosssource_seeding.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
