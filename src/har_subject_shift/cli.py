"""Command-line entry points; all real experiments explicitly gated."""
from __future__ import annotations
import argparse
from pathlib import Path
from .data import download, audit
from .experiment import train, final_test
from .visualize import generate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="har-shift", description="HAR UCI subject-independent benchmark")
    parser.add_argument("command", choices=("acquire", "audit", "train", "test", "figures", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("quick", "full"), default="quick", help="full=short grouped grid, quick=one config per model")
    parser.add_argument("--skip-hypotheses", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command in ("acquire", "all"):
        print("Official ZIP:", download(root))
    if args.command in ("audit", "all"):
        result = audit(root)
        print("AUDIT:", result["status"], "| subjects:", [result["splits"][k]["n_subjects"] for k in ("train", "test")])
    if args.command in ("train", "all"):
        lock = train(root, mode=args.mode, run_hypotheses=not args.skip_hypotheses)
        print("LOCKED:", lock["selected_model"], "| no holdout consulted")
    if args.command in ("test", "all"):
        scores = final_test(root)
        print("FINAL TEST:", scores["model"], "macro-F1", scores["macro_f1"], "n subjects", scores["n_subjects"])
    if args.command in ("figures", "all"):
        m = generate(root)
        print("FIGURES:", len(m))


if __name__ == "__main__":
    main()
