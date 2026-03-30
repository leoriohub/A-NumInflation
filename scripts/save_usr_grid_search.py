import json
import datetime
import uuid
import os
import numpy as np

def save_usr_grid_to_json(xi, lam, phi_grid, y_grid, criteria_usr, grid_dur, output_dir="../outputs"):
    """
    Saves the results of the USR duration grid search to a structured JSON file.
    
    Parameters:
    - xi: The Non-Minimal Coupling parameter
    - lam: The self-coupling parameter
    - criteria_usr: The criteria for the USR threshold
    - phi_grid: 1D numpy array of phi0 values
    - y_grid: 1D numpy array of yi values
    - grid_dur: 2D numpy array of USR durations (shape: len(y_grid), len(phi_grid))
    - output_dir: directory to save the JSON file (default is ../outputs relative to script)
    """
    N_phi = len(phi_grid)
    N_y = len(y_grid)
    
    phi_min, phi_max = np.min(phi_grid), np.max(phi_grid)
    y_min, y_max = np.min(y_grid), np.max(y_grid)
    
    results = []
    success_count = 0
    total_configs = N_phi * N_y
    
    # Flatten grid_dur into a list of dictionaries
    for i, y_val in enumerate(y_grid):
        for j, phi_val in enumerate(phi_grid):
            dur = grid_dur[i, j]
            
            results.append({
                "phi0": float(phi_val),
                "yi": float(y_val),
                "usr_duration": float(dur)
            })
            
            if dur >= 0:  # Assuming positive duration means a successful track
                success_count += 1
                
    run_id = str(uuid.uuid4())[:8]
    data = {
        "metadata": {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "description": "Grid search mapping USR duration to initial conditions (phi0 and yi) for Higgs Inflation."
        },
        "model_parameters": {
            "name": "Higgs Inflation",
            "xi": float(xi),
            "lambda": float(lam)
        },
        "grid_parameters": {
            "criteria_usr": float(criteria_usr),
            "phi0_min": float(phi_min),
            "phi0_max": float(phi_max),
            "phi0_steps": N_phi,
            "yi_min": float(y_min),
            "yi_max": float(y_max),
            "yi_steps": N_y,
            "total_configurations_attempted": total_configs,
            "successful_simulations": success_count
        },
        "numerical_settings": {
            "T_span_bg_max": float(max(100, xi/5.0)),
            "T_span_bg_steps": 3000
        },
        "results": results
    }
    
    # Ensure output_dir is processed correctly relative to the project root
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, output_dir))
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"USR_duration_heat_{phi_min:.2f}to{phi_max:.2f}_yi_{abs(y_min):.2f}to{abs(y_max):.2f}_crit_{criteria_usr:.2f}_{run_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Results successfully saved to {filepath}")
    return filepath
