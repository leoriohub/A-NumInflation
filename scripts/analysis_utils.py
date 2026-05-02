import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline

def load_grid_search(path: str) -> tuple:
    """Load a single grid_search JSON and return (metadata_dict, results_df).

    The results DataFrame contains all grid points including errors.
    Use filter_successful / filter_errors to separate.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data["results"])
    if "phi0" in df.columns:
        df["phi0"] = pd.to_numeric(df["phi0"], errors="coerce")
    return data, df


def filter_successful(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with status='success'."""
    return df[df["status"] == "success"].copy()


def filter_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with status='error'."""
    return df[df["status"] == "error"].copy()


def sweep_summary(grid_data: dict, results_df: pd.DataFrame) -> None:
    """Print a text summary of a completed sweep."""
    mg = grid_data["grid_parameters"]
    mp = grid_data["model_parameters"]
    successes = filter_successful(results_df)
    errors = filter_errors(results_df)

    print(f"Model:  {grid_data['model_parameters'].get('name', '?')}")
    xi = mp.get("xi", "?")
    lam = mp.get("lam", "?")
    print(f"Params: xi={xi}, lambda={lam}")
    print(f"Grid:   phi0 ∈ [{mg['phi0_min']}, {mg['phi0_max']}] × {mg['phi0_steps']} steps, "
          f"yi ∈ {mg.get('yi_values', '?')}")
    print(f"Run:    {mg['total_configurations_attempted']} points → "
          f"{len(successes)} ok, {len(errors)} errors")

    if not successes.empty:
        print(f"\nn_s range:      [{successes['ns'].min():.4f}, "
              f"{successes['ns'].max():.4f}]")
        print(f"r range:        [{successes['r'].min():.6f}, "
              f"{successes['r'].max():.6f}]")
        print(f"N_total range:  [{successes['N_total'].min():.1f}, "
              f"{successes['N_total'].max():.1f}]")

    if not errors.empty:
        print(f"\nFirst few errors:")
        for _, row in errors.head(5).iterrows():
            msg = str(row.get("message", ""))[:80]
            print(f"  phi0={row['phi0']:.4f} yi={row['yi']:.4f}: {msg}")


def scan_simulations(directory="../outputs"):
    """Scans the output directory and returns a sorted metadata dataframe."""
    sims = []
    if not os.path.exists(directory):
        return pd.DataFrame()
        
    for f in os.listdir(directory):
        if f.endswith(".json") and "grid_search" in f:
            path = os.path.join(directory, f)
            try:
                with open(path, 'r') as jf:
                    data = json.load(jf)
                    m, g, p = data['metadata'], data['grid_parameters'], data['model_parameters']
                    sims.append({
                        "label": f"{m['timestamp'][:16].replace('T', ' ')} | ξ={p['xi']} | x0: {g['phi0_min']}-{g['phi0_max']}",
                        "path": path,
                        "timestamp": m['timestamp']
                    })
            except: continue
    return pd.DataFrame(sims).sort_values(by="timestamp", ascending=False) if sims else pd.DataFrame()

def get_yi_slice(dataframe, target_yi=None):
    """Filters the dataframe for a specific yi slice and cleans the data."""
    if target_yi is None:
        target_yi = dataframe['yi'].unique()[0]
    
    # Filter and clean
    slice_data = dataframe[np.isclose(dataframe['yi'], target_yi, atol=1e-5)].copy()
    slice_data = slice_data.sort_values('phi0').drop_duplicates(subset=['phi0'])
    
    # Apply physical constraints (ns < 1)
    slice_data = slice_data[(slice_data['ns_SR'] <= 1.0) & (slice_data['ns'] <= 1.0)]
    return slice_data, float(target_yi)

def plot_ms_vs_sr_comparison(slice_df, yi_val):
    """Generates the professional 2-panel comparison plot."""
    if slice_df.empty:
        print(f"Warning: No valid data points found for yi = {yi_val}")
        return

    x_raw, y_ms_raw, y_sr_raw = slice_df['phi0'], slice_df['ns'], slice_df['ns_SR']
    y_diff_raw = y_ms_raw - y_sr_raw

    # Interpolation logic for the lower panel
    x_smooth = np.linspace(x_raw.min(), x_raw.max(), 300)
    k_spline = min(3, len(x_raw) - 1)
    y_diff_smooth = make_interp_spline(x_raw, y_diff_raw, k=k_spline)(x_smooth)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

    # Top Panel
    ax1 = plt.subplot(gs[0])
    ax1.scatter(x_raw, y_ms_raw, color='teal', s=45, alpha=0.8, label='Exact MS ($n_s$)', zorder=5)
    ax1.scatter(x_raw, y_sr_raw, color='crimson', marker='s', s=45, alpha=0.8, label='Slow-Roll Approx ($n_s^{(SR)}$)', zorder=5)
    ax1.set_ylabel(r"$n_s$ at $N_*=60$")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    ax1.tick_params(labelbottom=False) 

    # Bottom Panel
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(x_smooth, y_diff_smooth, linestyle='-', linewidth=2, color='indigo', zorder=4)
    ax2.fill_between(x_smooth, 0, y_diff_smooth, color='indigo', alpha=0.2, zorder=3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel(r"Initial Field $x_0$")
    ax2.set_ylabel(r"Residual $\Delta n_s$")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Save logic
    save_dir = "../images"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    filename = f"ns_MSvSR_yi_{yi_val:.2f}_phi0_{x_raw.min():.2f}to{x_raw.max():.2f}.png"
    plt.savefig(f"{save_dir}/{filename}", bbox_inches='tight', dpi=300)
    print(f"Professional plot saved as: {save_dir}/{filename}")
    plt.show()
