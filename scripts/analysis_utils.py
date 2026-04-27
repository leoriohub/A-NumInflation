import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline

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
