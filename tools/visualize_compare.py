#!/usr/bin/env python3
"""
Visualize ScanNet input and PTv3 prediction by exporting colored point clouds.

Outputs (PLY):
1) input RGB point cloud
2) prediction semantic color point cloud
3) GT semantic color point cloud (if segment20 exists)
4) side-by-side input vs prediction
5) side-by-side GT vs prediction (if GT exists)
6) prediction error highlight (if GT exists)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


SCANNET20_NAMES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

VALID_CLASS_IDS_20 = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
    dtype=np.int64,
)

SCANNET_COLOR_MAP_20 = {
    1: (174, 199, 232),
    2: (152, 223, 138),
    3: (31, 119, 180),
    4: (255, 187, 120),
    5: (188, 189, 34),
    6: (140, 86, 75),
    7: (255, 152, 150),
    8: (214, 39, 40),
    9: (197, 176, 213),
    10: (148, 103, 189),
    11: (196, 156, 148),
    12: (23, 190, 207),
    14: (247, 182, 210),
    16: (219, 219, 141),
    24: (255, 127, 14),
    28: (158, 218, 229),
    33: (44, 160, 44),
    34: (112, 128, 144),
    36: (227, 119, 194),
    39: (82, 84, 163),
}

PALETTE_20 = np.array([SCANNET_COLOR_MAP_20[c] for c in VALID_CLASS_IDS_20], dtype=np.uint8)


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be [N,3], got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb size mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        data = np.concatenate([xyz, rgb.astype(np.float32)], axis=1)
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")


def labels_to_rgb_sem20(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < 20)
    if np.any(valid):
        rgb[valid] = PALETTE_20[labels[valid].astype(np.int64)]
    return rgb


def normalize_input_rgb(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color)
    if color.dtype == np.uint8:
        return color
    color_f = color.astype(np.float32)
    max_v = float(np.max(color_f))
    if max_v <= 1.0:
        color_f = color_f * 255.0
    return np.clip(color_f, 0, 255).astype(np.uint8)


def find_scene_dir(data_root: Path, scene: str) -> Tuple[Path, str]:
    for split in ("train", "val", "test"):
        p = data_root / split / scene
        if p.is_dir():
            return p, split
    candidates = list(data_root.glob(f"*/{scene}"))
    if candidates:
        split_name = candidates[0].parent.name
        return candidates[0], split_name
    raise FileNotFoundError(f"Scene '{scene}' not found under {data_root}")


def scene_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 20) -> Dict[str, object]:
    valid = gt >= 0
    if not np.any(valid):
        return {"available": False}

    pred_v = pred[valid]
    gt_v = gt[valid]
    acc = float(np.mean(pred_v == gt_v))

    per_class = {}
    ious = []
    for c in range(num_classes):
        p = pred_v == c
        g = gt_v == c
        inter = int(np.sum(p & g))
        union = int(np.sum(p | g))
        if union > 0:
            iou = inter / union
            ious.append(iou)
            per_class[c] = {
                "name": SCANNET20_NAMES[c],
                "iou": float(iou),
                "intersection": inter,
                "union": union,
            }
    miou = float(np.mean(ious)) if ious else 0.0
    return {
        "available": True,
        "all_acc": acc,
        "miou_scene": miou,
        "valid_points": int(valid.sum()),
        "per_class": per_class,
    }


def default_pred_path(result_dir: Path, scene: Optional[str]) -> Tuple[Path, str]:
    if scene is not None:
        p = result_dir / f"{scene}_pred.npy"
        if not p.is_file():
            raise FileNotFoundError(f"Prediction file not found: {p}")
        return p, scene

    candidates = sorted(result_dir.glob("*_pred.npy"))
    if not candidates:
        raise FileNotFoundError(f"No *_pred.npy found in {result_dir}")
    p = candidates[0]
    scene_name = p.stem.replace("_pred", "")
    return p, scene_name


def concat_side_by_side(
    xyz_left: np.ndarray,
    rgb_left: np.ndarray,
    xyz_right: np.ndarray,
    rgb_right: np.ndarray,
    offset_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    span_x = float(np.max(xyz_left[:, 0]) - np.min(xyz_left[:, 0]))
    if span_x <= 1e-6:
        span_x = 1.0
    offset = np.array([span_x * offset_scale, 0.0, 0.0], dtype=np.float32)
    xyz = np.concatenate([xyz_left, xyz_right + offset], axis=0)
    rgb = np.concatenate([rgb_left, rgb_right], axis=0)
    return xyz, rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ScanNet input/prediction/gt comparison point clouds to PLY."
    )
    parser.add_argument("--data-root", type=Path, default=Path("deploy_host/data"))
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("deploy_host/exp/scannet/ptv3-2080-smoke/result"),
    )
    parser.add_argument("--pred-path", type=Path, default=None)
    parser.add_argument("--scene", type=str, default=None, help="Scene name, e.g. scene0011_00")
    parser.add_argument("--output-dir", type=Path, default=Path("deploy_host/exp/vis"))
    parser.add_argument("--offset-scale", type=float, default=1.2)
    args = parser.parse_args()

    if args.pred_path is not None:
        pred_path = args.pred_path
        if not pred_path.is_file():
            raise FileNotFoundError(f"--pred-path not found: {pred_path}")
        scene = args.scene or pred_path.stem.replace("_pred", "")
    else:
        pred_path, scene = default_pred_path(args.result_dir, args.scene)

    scene_dir, split = find_scene_dir(args.data_root, scene)
    coord = np.load(scene_dir / "coord.npy").astype(np.float32)
    color = normalize_input_rgb(np.load(scene_dir / "color.npy"))
    pred = np.load(pred_path).reshape(-1).astype(np.int64)
    if coord.shape[0] != pred.shape[0]:
        raise ValueError(f"Point count mismatch: coord={coord.shape[0]}, pred={pred.shape[0]}")

    gt_path = scene_dir / "segment20.npy"
    gt = None
    if gt_path.is_file():
        gt = np.load(gt_path).reshape(-1).astype(np.int64)
        if gt.shape[0] != coord.shape[0]:
            raise ValueError(f"GT size mismatch: coord={coord.shape[0]}, gt={gt.shape[0]}")

    out_scene_dir = args.output_dir / scene
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    pred_rgb = labels_to_rgb_sem20(pred)
    input_rgb = color
    gt_rgb = labels_to_rgb_sem20(gt) if gt is not None else None

    # 1) input
    input_ply = out_scene_dir / f"{scene}_input_rgb.ply"
    write_ply_xyzrgb(input_ply, coord, input_rgb)

    # 2) pred
    pred_ply = out_scene_dir / f"{scene}_pred_sem20.ply"
    write_ply_xyzrgb(pred_ply, coord, pred_rgb)

    # 3) gt
    gt_ply = None
    if gt_rgb is not None:
        gt_ply = out_scene_dir / f"{scene}_gt_sem20.ply"
        write_ply_xyzrgb(gt_ply, coord, gt_rgb)

    # 4) side-by-side input vs pred
    xyz_ip, rgb_ip = concat_side_by_side(coord, input_rgb, coord, pred_rgb, args.offset_scale)
    compare_input_pred_ply = out_scene_dir / f"{scene}_compare_input_vs_pred.ply"
    write_ply_xyzrgb(compare_input_pred_ply, xyz_ip, rgb_ip)

    # 5) side-by-side gt vs pred
    compare_gt_pred_ply = None
    error_ply = None
    if gt_rgb is not None:
        xyz_gp, rgb_gp = concat_side_by_side(coord, gt_rgb, coord, pred_rgb, args.offset_scale)
        compare_gt_pred_ply = out_scene_dir / f"{scene}_compare_gt_vs_pred.ply"
        write_ply_xyzrgb(compare_gt_pred_ply, xyz_gp, rgb_gp)

        valid = gt >= 0
        err_rgb = np.zeros_like(pred_rgb)
        err_rgb[valid & (pred == gt)] = np.array([120, 120, 120], dtype=np.uint8)
        err_rgb[valid & (pred != gt)] = np.array([255, 0, 0], dtype=np.uint8)
        error_ply = out_scene_dir / f"{scene}_pred_error_highlight.ply"
        write_ply_xyzrgb(error_ply, coord, err_rgb)

    metrics = scene_metrics(pred, gt) if gt is not None else {"available": False}
    summary = {
        "scene": scene,
        "split": split,
        "num_points": int(coord.shape[0]),
        "data_root": str(args.data_root),
        "pred_path": str(pred_path),
        "outputs": {
            "input_rgb_ply": str(input_ply),
            "pred_sem20_ply": str(pred_ply),
            "gt_sem20_ply": str(gt_ply) if gt_ply else None,
            "compare_input_vs_pred_ply": str(compare_input_pred_ply),
            "compare_gt_vs_pred_ply": str(compare_gt_pred_ply) if compare_gt_pred_ply else None,
            "pred_error_highlight_ply": str(error_ply) if error_ply else None,
        },
        "metrics": metrics,
        "sem20_legend": {str(i): SCANNET20_NAMES[i] for i in range(20)},
    }
    summary_path = out_scene_dir / f"{scene}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] Visualization files exported.")
    print(f"[INFO] Scene: {scene} ({split})")
    print(f"[INFO] Output dir: {out_scene_dir}")
    if metrics.get("available"):
        print(
            f"[INFO] Scene metrics: all_acc={metrics['all_acc']:.4f}, "
            f"miou_scene={metrics['miou_scene']:.4f}"
        )
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
