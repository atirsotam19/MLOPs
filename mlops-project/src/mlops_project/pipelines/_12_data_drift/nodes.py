import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import jensenshannon

# ----- PSI CALCULATION -----

def calculate_psi_for_all_features(current: pd.DataFrame, reference: pd.DataFrame, features: list[str], bins=10) -> pd.DataFrame:
    """Calculate PSI for all features."""
    def psi_feature(curr, ref, feature, bins):
        ref_values = ref[feature]
        bins = np.linspace(ref_values.min(), ref_values.max(), bins + 1)
        ref_percents = np.histogram(ref[feature], bins=bins)[0] / len(ref)
        curr_percents = np.histogram(curr[feature], bins=bins)[0] / len(curr)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        return np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))

    psi_vals = {f: psi_feature(current, reference, f, bins) for f in features}
    return pd.DataFrame.from_dict(psi_vals, orient='index', columns=['PSI']).sort_values('PSI', ascending=False)

def compute_psi(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], bins: int = 10) -> pd.DataFrame:
    """Node to compute PSI given reference and current datasets."""
    psi_df = calculate_psi_for_all_features(current=current, reference=reference, features=features, bins=bins)
    return psi_df

def plot_psi_bar(psi_df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    psi_df['PSI'].plot(kind='bar', color='orange')
    plt.title('PSI by Feature')
    plt.ylabel('PSI')
    plt.xlabel('Features')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "psi_bar_plot.png"))
    plt.close()

# ----- JENSEN-SHANNON DRIFT PLOTS -----

def plot_feature_drift_over_time(data_chunks: list[pd.DataFrame], 
                                  feature: str, 
                                  reference_chunk: pd.DataFrame, 
                                  output_dir: str,
                                  threshold: float = 0.2) -> None:
    js_divergences = []

    ref_vals = reference_chunk[feature].dropna()
    ref_hist, bin_edges = np.histogram(ref_vals, bins=20, density=True)
    ref_hist = np.where(ref_hist == 0, 1e-8, ref_hist)
    ref_hist = ref_hist / np.sum(ref_hist)

    for chunk in data_chunks:
        chunk_vals = chunk[feature].dropna()
        chunk_hist, _ = np.histogram(chunk_vals, bins=bin_edges, density=True)
        chunk_hist = np.where(chunk_hist == 0, 1e-8, chunk_hist)
        chunk_hist = chunk_hist / np.sum(chunk_hist)

        js_div = jensenshannon(ref_hist, chunk_hist)
        js_divergences.append(js_div)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(data_chunks)), js_divergences, marker='o', label='Jensen-Shannon Divergence', color='blue')
    for i, val in enumerate(js_divergences):
        if val > threshold:
            plt.scatter(i, val, color='red', label='Alert' if i == 0 else "", zorder=5)
    plt.title(f"Univariate Data Drift for {feature}")
    plt.xlabel("Chunk Index")
    plt.ylabel("Jensen-Shannon Divergence")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{feature}_drift.png"))
    plt.close()

def plot_drift_for_features(current: pd.DataFrame, reference: pd.DataFrame, features: list[str], chunk_size: int, output_dir: str, threshold: float = 0.2) -> None:
    """Wrapper to create drift plots for all features."""
    data_chunks = [current.iloc[i:i+chunk_size] for i in range(0, len(current), chunk_size)]
    for feature in features:
        plot_feature_drift_over_time(data_chunks, feature, reference, output_dir, threshold)
