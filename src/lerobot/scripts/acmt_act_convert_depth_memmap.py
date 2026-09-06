"""Convert aligned H5 depth streams to the ACMT-ACTv2 sidecar."""

from __future__ import annotations

import argparse

from lerobot.datasets.acmt_act_depth_memmap import convert_h5_to_depth_memmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--rgb-memmap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    manifest = convert_h5_to_depth_memmap(
        args.data_dir,
        args.rgb_memmap_dir,
        args.output_dir,
        split_file=args.split_file,
        chunk_frames=args.chunk_frames,
        resume=args.resume,
        progress=args.progress,
    )
    print(manifest)


if __name__ == "__main__":
    main()
