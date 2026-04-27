"""Smoke test for the extracted multi_quickPIV GUI pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package yet.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multi_quickpiv_gui.backend.export import save_batch_result, save_pair_result
from multi_quickpiv_gui.backend.io import load_stack
from multi_quickpiv_gui.workflow.params import (
    MedianDespikeParams,
    PIVRunParams,
    PostProcessParams,
    SNFilterParams,
    WorkflowParams,
)
from multi_quickpiv_gui.workflow.pipeline import run_batch_piv, run_piv_pair


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Smoke-test the extracted multi_quickPIV GUI pipeline."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input TIFF/TIFF or HDF5 stack.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "batch"),
        default="single",
        help="Run a single consecutive frame pair or all consecutive pairs.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Start frame index for single mode. Uses frame_index -> frame_index + 1.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional export base path. Writes both .npz and .h5.",
    )

    parser.add_argument("--inter-h", type=int, default=64)
    parser.add_argument("--inter-w", type=int, default=64)
    parser.add_argument("--search-h", type=int, default=128)
    parser.add_argument("--search-w", type=int, default=128)
    parser.add_argument("--step-h", type=int, default=32)
    parser.add_argument("--step-w", type=int, default=32)

    parser.add_argument(
        "--corr-alg",
        type=str,
        default="nsqecc",
        help="Correlation algorithm string passed through to multi_quickPIV.",
    )
    parser.add_argument(
        "--compute-sn",
        action="store_true",
        help="Enable signal-to-noise computation.",
    )
    parser.add_argument(
        "--median-despike",
        action="store_true",
        help="Enable median-despike post-processing.",
    )
    parser.add_argument(
        "--despike-ksize",
        type=int,
        default=3,
        help="Median-despike window size.",
    )
    parser.add_argument(
        "--despike-threshold",
        type=float,
        default=3.5,
        help="Median-despike MAD threshold.",
    )
    parser.add_argument(
        "--sn-filter",
        action="store_true",
        help="Enable SN-threshold post-processing.",
    )
    parser.add_argument(
        "--sn-min",
        type=float,
        default=1.0,
        help="Minimum SN threshold for SN-based filtering.",
    )

    return parser


def build_workflow_params(args: argparse.Namespace) -> WorkflowParams:
    """Build WorkflowParams from CLI arguments."""
    params = WorkflowParams(
        run=PIVRunParams(
            inter_size=(args.inter_h, args.inter_w),
            search_margin=(args.search_h, args.search_w),
            step=(args.step_h, args.step_w),
            compute_sn=bool(args.compute_sn),
            corr_alg=args.corr_alg,
        ),
        postprocess=PostProcessParams(
            median_despike=MedianDespikeParams(
                enabled=bool(args.median_despike),
                ksize=max(3, args.despike_ksize),
                threshold=args.despike_threshold,
            ),
            sn_filter=SNFilterParams(
                enabled=bool(args.sn_filter),
                minimum=args.sn_min if args.sn_filter else 1.0,
            ),
        ),
    )
    params.validate()
    return params


def main() -> None:
    """Run the smoke test."""
    parser = build_parser()
    args = parser.parse_args()

    loaded = load_stack(args.input_path)
    params = build_workflow_params(args)

    print(f"Loaded stack: {loaded.source_path.name}")
    print(f"Shape: {loaded.shape}")
    if loaded.dataset_name is not None:
        print(f"HDF5 dataset: {loaded.dataset_name}")

    if args.mode == "single":
        if args.frame_index < 0 or args.frame_index >= loaded.num_frames - 1:
            raise ValueError(
                f"frame-index must be between 0 and {loaded.num_frames - 2}."
            )

        result = run_piv_pair(
            loaded.data[args.frame_index],
            loaded.data[args.frame_index + 1],
            params=params,
        )

        print(
            f"Ran single pair: {args.frame_index} -> {args.frame_index + 1}"
        )
        print(f"U/V shape: {result.u.shape}")
        print(f"Grid shape: {result.xg.shape}")
        print(f"SN available: {result.sn is not None}")
        print(f"SN replacements: {result.sn_replaced}")

        if args.out is not None:
            paths = save_pair_result(args.out, result)
            print(f"Saved NPZ: {paths.npz_path}")
            print(f"Saved H5 : {paths.h5_path}")

    else:
        def progress(done: int, total: int, _result) -> None:
            print(f"Processed pair {done}/{total}")

        batch_result = run_batch_piv(
            loaded.data,
            params=params,
            progress_callback=progress,
        )

        print(f"Ran batch over {len(batch_result.pair_results)} pairs.")
        print(f"Grid shape: {batch_result.xg.shape if batch_result.xg is not None else None}")
        print(f"SN available: {batch_result.sn_list is not None}")

        if args.out is not None:
            paths = save_batch_result(args.out, batch_result)
            print(f"Saved NPZ: {paths.npz_path}")
            print(f"Saved H5 : {paths.h5_path}")


if __name__ == "__main__":
    main()