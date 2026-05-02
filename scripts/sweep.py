#!/usr/bin/env python3
"""CLI entry point for running parameter sweeps on a lab machine.

Usage examples:

  # Minimal — uses default ranges (phi0=[5.4,5.8] (200) x yi=[-0.2,-0.001] (50))
  python scripts/sweep.py \
    --model HiggsModel --xi 15000 --lambda 0.13 --n-workers 8

  # Custom ranges
  python scripts/sweep.py \
    --model HiggsModel --xi 17000 --lambda 0.13 \
    --phi0-min 5.4 --phi0-max 5.7 --phi0-steps 260 \
    --yi -0.06 --n-workers 4

  # Resume an interrupted run
  python scripts/sweep.py \
    --model HiggsModel --xi 17000 --lambda 0.13 \
    --output outputs/grid_search_...json --resume

  # Custom integration time span
  python scripts/sweep.py \
    --model HiggsModel --xi 17000 --lambda 0.13 \
    --phi0-min 5.4 --phi0-max 5.7 --phi0-steps 100 \
    --yi -0.06 --bg-t-max 1000 --bg-t-steps 50000
"""

import os
import sys
import argparse
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.sweep_engine import run_parameter_sweep, _MODEL_REGISTRY


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a parallel grid search of inflationary observables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g_model = parser.add_argument_group("Model parameters")
    g_model.add_argument(
        "--model", required=True,
        choices=sorted(_MODEL_REGISTRY.keys()),
        help="Inflation model class name",
    )
    g_model.add_argument("--xi", type=float, default=None, help="Non-minimal coupling")
    g_model.add_argument("--lambda", dest="lam", type=float, default=None,
                         help="Self-coupling")
    g_model.add_argument("--S", type=float, default=None, help="Model parameter S")

    g_grid = parser.add_argument_group("Grid parameters")
    g_grid.add_argument("--phi0-min", type=float, default=5.4,
                        help="phi0 range minimum (default: 5.4)")
    g_grid.add_argument("--phi0-max", type=float, default=5.8,
                        help="phi0 range maximum (default: 5.8)")
    g_grid.add_argument("--phi0-steps", type=int, default=200,
                        help="phi0 range number of steps (default: 200)")
    g_grid.add_argument("--yi", type=float, nargs="+", default=None,
                        help="Specific yi value(s) — use nargs='+' syntax")
    g_grid.add_argument("--yi-min", type=float, default=-0.2,
                        help="yi range minimum (default: -0.2)")
    g_grid.add_argument("--yi-max", type=float, default=-0.001,
                        help="yi range maximum (default: -0.001)")
    g_grid.add_argument("--yi-steps", type=int, default=50,
                        help="yi range number of steps (default: 50)")

    g_num = parser.add_argument_group("Numerical settings")
    g_num.add_argument("--delta", type=float, default=1e-4,
                       help="Finite-difference step for ns (default: 1e-4)")
    g_num.add_argument("--N-star", type=float, default=60.0,
                       help="E-folds from end to pivot scale (default: 60)")
    g_num.add_argument("--bg-t-max", type=float, default=None,
                       help="Background integration max time")
    g_num.add_argument("--bg-t-steps", type=int, default=None,
                       help="Background integration time steps")

    g_run = parser.add_argument_group("Execution")
    g_run.add_argument("--n-workers", type=int, default=1,
                       help="Number of parallel workers (default: 1 = serial)")
    g_run.add_argument("--output-dir", default="outputs",
                       help="Output directory (default: outputs)")
    g_run.add_argument("--output", dest="output_name", default=None,
                       help="Grid JSON filename (auto-generated if not given)")
    g_run.add_argument("--resume", action="store_true",
                       help="Skip already-computed grid points")
    g_run.add_argument("--description", default="",
                       help="Optional description for grid metadata")

    args = parser.parse_args(argv)

    # Resolve yi: use --yi (list) if given, else fall back to range defaults
    if args.yi is not None:
        yi_values = args.yi
    else:
        yi_values = np.linspace(args.yi_min, args.yi_max, args.yi_steps).tolist()

    # Build model kwargs from provided args
    model_kwargs = {}
    if args.xi is not None:
        model_kwargs["xi"] = args.xi
    if args.lam is not None:
        model_kwargs["lam"] = args.lam
    if args.S is not None:
        model_kwargs["S"] = args.S

    return args, yi_values, model_kwargs


def main():
    args, yi_values, model_kwargs = parse_args()

    phi0_range = (args.phi0_min, args.phi0_max, args.phi0_steps)

    run_parameter_sweep(
        model_class_name=args.model,
        model_kwargs=model_kwargs,
        phi0_range=phi0_range,
        yi_values=yi_values,
        delta=args.delta,
        N_star=args.N_star,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        output_name=args.output_name,
        bg_t_max=args.bg_t_max,
        bg_t_steps=args.bg_t_steps,
        resume=args.resume,
        description=args.description,
    )


if __name__ == "__main__":
    main()
