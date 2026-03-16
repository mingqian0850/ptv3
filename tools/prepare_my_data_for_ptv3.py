#!/usr/bin/env python3
"""
Convert custom point cloud exports into Pointcept/ScanNet-like folder format
so they can be directly used by PTv3 test pipeline.

Expected input files (per frame):
  - pointcloud_<id>.npy            -> Nx3 xyz
  - pointcloud_rgb_<id>.npy        -> Nx3 or Nx4 color
  - pointcloud_normals_<id>.npy    -> Nx3 or Nx4 normals (optional)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare custom point cloud data for PTv3 inference."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("deploy_host/data/my_data"),
        help="Directory containing pointcloud_*.npy style files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("deploy_host/data/custom_scannet"),
        help="Output dataset root in ScanNet-like layout.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split name to write scenes into (default: val).",
    )
    parser.add_argument(
        "--scene-prefix",
        type=str,
        default="custom_scene",
        help="Scene folder prefix (final name: <prefix>_<id>).",
    )
    parser.add_argument(
        "--write-dummy-segment",
        action="store_true",
        help="Write segment20.npy filled with -1 for unlabeled inference.",
    )
    return parser.parse_args()


def to_uint8_rgb(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] != n:
        raise ValueError(f"RGB point count mismatch: {arr.shape[0]} vs {n}")
    if arr.shape[1] < 3:
        raise ValueError(f"RGB array must have >=3 channels, got {arr.shape}")
    rgb = arr[:, :3]
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32)
    if np.max(rgb) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


def to_float32_normal(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] != n:
        raise ValueError(f"Normal point count mismatch: {arr.shape[0]} vs {n}")
    if arr.shape[1] < 3:
        raise ValueError(f"Normal array must have >=3 channels, got {arr.shape}")
    normal = arr[:, :3].astype(np.float32)
    return normal


def find_frames(input_dir: Path) -> List[Tuple[str, Path]]:
    pattern = re.compile(r"^pointcloud_(\d+)\.npy$")
    out: List[Tuple[str, Path]] = []
    for p in sorted(input_dir.glob("pointcloud_*.npy")):
        m = pattern.match(p.name)
        if m is None:
            continue
        frame_id = m.group(1)
        out.append((frame_id, p))
    return out


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_root: Path = args.output_root
    split_dir = output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    frames = find_frames(input_dir)
    if not frames:
        raise FileNotFoundError(
            f"No frame file like pointcloud_<id>.npy found in: {input_dir}"
        )

    scene_rel_paths: List[str] = []
    summary = {"input_dir": str(input_dir), "output_root": str(output_root), "frames": []}

    for frame_id, coord_file in frames:
        rgb_file = input_dir / f"pointcloud_rgb_{frame_id}.npy"
        normal_file = input_dir / f"pointcloud_normals_{frame_id}.npy"

        coord = np.load(coord_file).astype(np.float32)
        if coord.ndim != 2 or coord.shape[1] < 3:
            raise ValueError(f"Invalid coord shape in {coord_file}: {coord.shape}")
        coord = coord[:, :3]
        n = coord.shape[0]

        if rgb_file.is_file():
            rgb_raw = np.load(rgb_file)
            color = to_uint8_rgb(rgb_raw, n)
        else:
            color = np.full((n, 3), 127, dtype=np.uint8)

        if normal_file.is_file():
            normal_raw = np.load(normal_file)
            normal = to_float32_normal(normal_raw, n)
        else:
            normal = np.zeros((n, 3), dtype=np.float32)

        scene_name = f"{args.scene_prefix}_{frame_id}"
        scene_dir = split_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "coord.npy", coord)
        np.save(scene_dir / "color.npy", color)
        np.save(scene_dir / "normal.npy", normal)

        if args.write_dummy_segment:
            segment20 = np.full((n,), -1, dtype=np.int64)
            np.save(scene_dir / "segment20.npy", segment20)

        rel = f"{args.split}/{scene_name}"
        scene_rel_paths.append(rel)
        summary["frames"].append(
            {
                "frame_id": frame_id,
                "scene_name": scene_name,
                "num_points": int(n),
                "coord_file": str(coord_file),
                "rgb_file": str(rgb_file) if rgb_file.is_file() else None,
                "normal_file": str(normal_file) if normal_file.is_file() else None,
            }
        )

    split_json = output_root / f"{args.split}.json"
    split_json.write_text(json.dumps(scene_rel_paths, indent=2), encoding="utf-8")

    summary_path = output_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] Custom data prepared for PTv3.")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Split folder: {split_dir}")
    print(f"[INFO] Scene count: {len(scene_rel_paths)}")
    print(f"[INFO] Split file: {split_json}")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
