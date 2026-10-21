"""
run_notebooks.py: execute the production notebooks in order and save their outputs.

Usage (from the repository root):

    python run_notebooks.py                     # 01_volume, 02_price, 03_income
    python run_notebooks.py 02_price 03_income  # a subset, still in pipeline order

Each notebook runs top to bottom in a fresh kernel and is saved in place. The run
stops at the first notebook that raises, after saving the outputs produced up to
the failing cell. 01_volume downloads the Eurostat deliveries, so it needs an
internet connection; 02_price and 03_income only use files in data/.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "production_notebooks"
PIPELINE = ["01_volume", "02_price", "03_income"]

if sys.platform == "win32":
    # pyzmq needs a selector event loop on Windows.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_notebook(name: str, kernel_name: str) -> bool:
    """Execute one notebook in place. Returns True if every cell ran."""
    path = NOTEBOOK_DIR / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name=kernel_name,
        timeout=None,
        record_timing=False,
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    start = time.time()
    try:
        client.execute()
        ok = True
    except CellExecutionError as exc:
        print(f"  failed: {str(exc).strip().splitlines()[-1]}")
        ok = False
    finally:
        nbformat.write(nb, path)
    print(f"  {'done' if ok else 'stopped'} in {time.time() - start:.0f} s")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("notebooks", nargs="*",
                        help=f"notebooks to run, among {', '.join(PIPELINE)} (default: all)")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    args = parser.parse_args()
    unknown = set(args.notebooks) - set(PIPELINE)
    if unknown:
        parser.error(f"unknown notebook(s): {', '.join(sorted(unknown))}")

    selected = [n for n in PIPELINE if not args.notebooks or n in args.notebooks]
    for name in selected:
        print(f"Running {name}.ipynb")
        if not run_notebook(name, args.kernel):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
