import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
