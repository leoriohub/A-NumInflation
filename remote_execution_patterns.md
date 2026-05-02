# Remote Execution Design Patterns

Reusable boilerplate for running Python research code on a lab machine
with parallelism, progress, and minimal friction. Copy the patterns,
swap in your own work function.

## 1. Opt-in parallel — never break existing code

Wrap a CPU-bound loop with `ProcessPoolExecutor` behind `n_workers`
defaulting to `1`. The serial branch is the original code verbatim.

```python
def run_sweep(..., n_workers=1):
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(arg1, arg2) for ...]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_worker_fn, tasks):
                store(result)
    else:
        # original serial loop
        for arg1, arg2 in ...:
            result = compute(arg1, arg2)
            store(result)
```

**Details:**
- `ProcessPoolExecutor`, not `ThreadPoolExecutor` (CPU-bound, GIL-limited)
- Worker function must be at module level (pickle requirement)
- Import `concurrent.futures` inside the `if` so serial mode needs nothing extra
- Never nest parallelism — pick the outer loop or the inner one

## 2. Flat progress — one line per item

Same format in both serial and parallel:

```python
n = len(items)
print(f"Processing {n} items (n_workers={n_workers})...")
for i, item in enumerate(items, 1):
    result = compute(item)
    print(f"  [{i}/{n}] {label(item)} -> {summarize(result)}")
```

For groups (e.g. a single k-mode run has many sub-items), print a
header with the total, then inline progress every 10% using `\r`.

No external dependencies — no tqdm.

## 3. Outer parallelism for parameter scans

When sweeping over independent grid points, parallelise at the grid
level (each point gets its own worker). Use `as_completed` for
real-time progress:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(compute, a, b): (a, b) for a, b in grid}
    for i, future in enumerate(as_completed(futures), 1):
        r = future.result()
        a, b = futures[future]
        print(f"[{i}/{total}] {a} {b} -> {r['metric']:.2f}")
```

## 4. Errors are data — never stop the whole run

Wrap every grid point's work in try/except and return a status dict:

```python
def worker_fn(a, b):
    try:
        result = compute(a, b)
        if result['status'] != 'ok':
            return {'a': a, 'b': b, 'status': 'error',
                    'message': result.get('message', '')}
        return {'a': a, 'b': b, 'status': 'ok', 'value': result['value']}
    except Exception as e:
        return {'a': a, 'b': b, 'status': 'error', 'message': str(e)}
```

Output JSON includes both successes and failures — filter at analysis
time.

```json
{
  "config": {"a_range": [1, 10], "b_values": [0.1, 0.5]},
  "results": [
    {"a": 1, "b": 0.1, "status": "error", "message": "convergence failed"},
    {"a": 2, "b": 0.5, "status": "ok", "value": 42.0}
  ]
}
```

## 5. Lab computes, local analyses

Heavy computation runs on the lab machine. Results are saved as JSON.
Analysis and visualisation happen locally.

```
Lab: python sweep.py --params ...  → outputs/sweep.json
Local: rsync -avz lab:~/project/outputs/ outputs/
       jupyter lab notebooks/analyse.ipynb
```

## 6. CLI design for remote runs

- Every parameter is a CLI argument with a sensible default
- `--n-workers` always optional, defaults to 1
- Output path is configurable (`--output-dir`)
- Use `nargs='+'` for lists of floats (argparse confuses bare `-` with flags)

```bash
python sweep.py \
  --param1-min 1 --param1-max 10 --param1-step 0.5 \
  --param2 0.1 0.5 1.0 \
  --n-workers 4
```

## 7. Inline post-processing — skip file I/O in parallel workers

If each grid point produces intermediate data that gets reduced (e.g.
a metric, a score), compute the reduction inline rather than writing
and re-reading files:

```python
# Avoid:
result = run_pipeline(...)        # saves to disk
score = analyse_from_file(path)   # loads from disk

# Prefer:
result = run_pipeline(save=False)
score = compute_metric(result["data_a"], result["data_b"])
```

## 8. Path setup (same in every script)

```python
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
```
