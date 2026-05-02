import os
import sys
import json
import uuid
import datetime
import itertools
import numpy as np
from typing import List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models import HiggsModel, NonMinimalQuarticModel, FullHiggsModel, QuadraticModel, SmoothUSRTransitionModel
from numerical_observables_calculation import run_inflation_protocol

_MODEL_REGISTRY = {
    "HiggsModel": HiggsModel,
    "NonMinimalQuarticModel": NonMinimalQuarticModel,
    "FullHiggsModel": FullHiggsModel,
    "QuadraticModel": QuadraticModel,
    "SmoothUSRTransitionModel": SmoothUSRTransitionModel,
}

def _grid_worker(task: dict) -> dict:
    model_class_name = task["model_class"]
    model_kwargs = task["model_kwargs"]
    phi0 = task["phi0"]
    yi = task["yi"]
    delta = task.get("delta", 1e-4)
    N_star = task.get("N_star", 60.0)
    output_dir = task.get("output_dir", "outputs/results")

    bg_t_max = task.get("bg_t_max")
    bg_t_steps = task.get("bg_t_steps")
    T_span_bg = None
    if bg_t_max is not None and bg_t_steps is not None:
        T_span_bg = np.linspace(0, bg_t_max, bg_t_steps)

    try:
        model = _MODEL_REGISTRY[model_class_name](**model_kwargs)
        res = run_inflation_protocol(
            model, phi0, yi,
            delta=delta,
            N_star=N_star,
            output_dir=output_dir,
            T_span_bg=T_span_bg,
            save_to_file=True,
        )
        if res["status"] == "success":
            return {
                "phi0": phi0,
                "yi": yi,
                "status": "success",
                "ns": res["ns"],
                "r": res["r"],
                "ns_SR": res["ns_SR"],
                "r_SR": res["r_SR"],
                "N_total": res["N_total"],
                "P_S": res["P_S"],
            }
        return {
            "phi0": phi0,
            "yi": yi,
            "status": "error",
            "message": res.get("message", ""),
        }
    except Exception as e:
        return {
            "phi0": phi0,
            "yi": yi,
            "status": "error",
            "message": str(e),
        }


def _build_grid_filename(model_class_name: str, model_kwargs: dict) -> str:
    safe_name = model_class_name.replace(' ', '_').replace('(', '').replace(')', '')
    xi = model_kwargs.get("xi", "?")
    lam = model_kwargs.get("lam", "?")
    return f"grid_search_{safe_name}_xi{xi}_lambda{lam}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"


def _results_subdir(output_dir: str, subdir: str) -> str:
    path = os.path.join(output_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def run_parameter_sweep(
    model_class_name: str,
    model_kwargs: dict,
    phi0_range: Tuple[float, float, int],
    yi_values: List[float],
    delta: float = 1e-4,
    N_star: float = 60.0,
    n_workers: int = 1,
    output_dir: str = "outputs",
    results_subdir: str = "results",
    output_name: Optional[str] = None,
    bg_t_max: Optional[float] = None,
    bg_t_steps: Optional[int] = None,
    resume: bool = False,
    description: str = "",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results_dir = _results_subdir(output_dir, results_subdir)

    if output_name is None:
        output_name = _build_grid_filename(model_class_name, model_kwargs)
    grid_path = os.path.join(output_dir, output_name)

    phi0_vals = np.linspace(phi0_range[0], phi0_range[1], phi0_range[2])

    task_list = []
    for phi0, yi in itertools.product(phi0_vals, yi_values):
        task_list.append({
            "model_class": model_class_name,
            "model_kwargs": dict(model_kwargs),
            "phi0": round(float(phi0), 10),
            "yi": yi,
            "delta": delta,
            "N_star": N_star,
            "output_dir": results_dir,
            "bg_t_max": bg_t_max,
            "bg_t_steps": bg_t_steps,
        })

    already_computed = set()
    if resume and os.path.exists(grid_path):
        with open(grid_path, 'r') as f:
            existing = json.load(f)
        for r in existing.get("results", []):
            if r.get("status") == "success":
                already_computed.add((r["phi0"], r["yi"]))

    to_run = [t for t in task_list if (t["phi0"], t["yi"]) not in already_computed]
    skipped = len(task_list) - len(to_run)
    total = len(to_run)

    if total == 0:
        print(f"All {len(task_list)} grid points already computed. Nothing to do.")
        return grid_path

    print(f"Grid: {len(phi0_vals)} x {len(yi_values)} = {len(task_list)} points "
          f"(n_workers={n_workers})"
          + (f", {skipped} skipped (resume)" if skipped else ""))

    results = []

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_grid_worker, t): (t["phi0"], t["yi"]) for t in to_run}
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                results.append(r)
                phi0, yi = futures[future]
                _print_progress(i, total, r, phi0, yi)
    else:
        for i, t in enumerate(to_run, 1):
            r = _grid_worker(t)
            results.append(r)
            _print_progress(i, total, r, t["phi0"], t["yi"])

    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    if resume and os.path.exists(grid_path):
        with open(grid_path, 'r') as f:
            existing = json.load(f)
        existing_results = existing.get("results", [])
        merged_phi0_yi = {(r["phi0"], r["yi"]) for r in results}
        existing_results = [r for r in existing_results
                           if (r["phi0"], r["yi"]) not in merged_phi0_yi]
        results = existing_results + results

    grid_data = {
        "metadata": {
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().isoformat(),
            "description": description or (
                f"Grid search of ns/r over (phi0, yi) for {model_class_name} "
                f"with xi={model_kwargs.get('xi', '?')}, lambda={model_kwargs.get('lam', '?')}"
            ),
        },
        "model_parameters": {
            "name": model_class_name,
            **model_kwargs,
        },
        "grid_parameters": {
            "phi0_min": phi0_range[0],
            "phi0_max": phi0_range[1],
            "phi0_steps": phi0_range[2],
            "yi_values": yi_values,
            "total_configurations_attempted": len(task_list),
            "successful_simulations": len(successes),
        },
        "numerical_settings": {
            "delta_finite_difference": delta,
            "N_star": N_star,
        },
        "results": results,
    }

    with open(grid_path, 'w') as f:
        json.dump(grid_data, f, indent=4)

    print(f"\nDone — {len(successes)}/{len(task_list)} successful, "
          f"{len(errors)} errors, "
          f"{skipped} skipped (from resume).")
    print(f"Grid JSON: {grid_path}")
    return grid_path


def _print_progress(i: int, total: int, r: dict, phi0: float, yi: float):
    if r["status"] == "success":
        print(f"  [{i}/{total}] phi0={phi0:.4f} yi={yi:.4f} -> "
              f"ns={r['ns']:.4f} ns_SR={r['ns_SR']:.4f} "
              f"r={r['r']:.6f} P={r['P_S']:.2e}  N={r['N_total']:.1f}")
    else:
        msg = r.get("message", "")[:80]
        print(f"  [{i}/{total}] phi0={phi0:.4f} yi={yi:.4f} -> ERROR: {msg}")
