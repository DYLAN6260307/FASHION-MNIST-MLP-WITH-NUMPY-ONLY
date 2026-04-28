from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mlp.reporting import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the homework PDF report from a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--github-url", default="TODO: replace with Public GitHub Repo URL")
    parser.add_argument("--weights-url", default="TODO: replace with Google Drive model weights URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_report(args.run_dir, args.output_pdf, args.github_url, args.weights_url)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()

