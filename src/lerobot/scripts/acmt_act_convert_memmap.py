"""CLI for building the exact-crop ACMT-ACT training memmap."""

from __future__ import annotations

import argparse

from lerobot.datasets.acmt_act_memmap import convert_h5_to_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="Compatibility option; conversion is lossless CPU I/O.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    manifest = convert_h5_to_memmap(
        args.data_dir,
        args.split_file,
        args.output_dir,
        chunk_frames=args.chunk_frames,
        resume=args.resume,
        progress=args.progress,
        device=args.device,
    )
    print(f"ACMT-ACT memmap ready: {manifest}")


if __name__ == "__main__":
    main()
