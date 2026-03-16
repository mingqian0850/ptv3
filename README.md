# ptv3

# PTv3 Custom Data Cheatsheet

> Image (recommended): `ptv3:2080-fixed`  
> Project root: `/home/mingqian/ws_ptv3`

## 0) Host Directory Layout

- Data root: `/home/mingqian/ws_ptv3/deploy_host/data`
- Your raw custom data: `/home/mingqian/ws_ptv3/deploy_host/data/my_data`
- Converted PTv3-ready data: `/home/mingqian/ws_ptv3/deploy_host/data/custom_scannet`
- Weights: `/home/mingqian/ws_ptv3/deploy_host/weights`
- Inference outputs: `/home/mingqian/ws_ptv3/deploy_host/exp`
- Logs: `/home/mingqian/ws_ptv3/deploy_host/logs`

## 1) Container Management

```bash
cd /home/mingqian/ws_ptv3

# Start (create if not exists)
IMAGE=ptv3:2080-fixed ./deploy_scripts/start_ptv3.sh

# Enter container shell
./deploy_scripts/shell_ptv3.sh

# Stop container
./deploy_scripts/stop_ptv3.sh
```

## 2) Prepare `my_data` for PTv3

```bash
cd /home/mingqian/ws_ptv3
./deploy_scripts/prepare_my_data.sh
```

This converts:
- `deploy_host/data/my_data/pointcloud_*.npy`
- `deploy_host/data/my_data/pointcloud_rgb_*.npy`
- `deploy_host/data/my_data/pointcloud_normals_*.npy`

into:
- `deploy_host/data/custom_scannet/val/custom_scene_xxxx/{coord,color,normal,segment20}.npy`
- `deploy_host/data/custom_scannet/val.json`

## 2.1) Scene Data Structure (What each file means)

A single scene folder usually looks like:

```text
sceneXXXX_YY/
  coord.npy
  color.npy
  normal.npy
  segment20.npy
  instance.npy      # optional
  segment200.npy    # optional
```

Meaning of each file:

- `coord.npy`  
  Point coordinates, usually shape `N x 3`, dtype `float32`.
- `color.npy`  
  RGB per point, usually shape `N x 3`, dtype `uint8` in range `0~255`.
- `normal.npy`  
  Surface normal per point, usually shape `N x 3`, dtype `float32` in range around `[-1, 1]`.
- `segment20.npy`  
  ScanNet20 semantic label per point, shape `N`, integer labels (`0~19`), `-1` means ignore/unlabeled.
- `instance.npy` (optional)  
  Instance ID per point, shape `N`, integer labels, `-1` means ignore/no instance.
- `segment200.npy` (optional)  
  ScanNet200 semantic label per point, shape `N`, integer labels, `-1` means ignore/unlabeled.

Important:

- All per-point files must have the same first dimension `N`.
- For your current custom inference pipeline, the required minimum is:
  - `coord.npy`
  - `color.npy`
  - `normal.npy`
  - `segment20.npy` (can be dummy all `-1` for unlabeled data)

## 3) Download Pretrained Weight (No GitHub Upload Needed)

The checkpoint `scannet-pt-v3m1-0-base-model_best.pth` is publicly downloadable from Hugging Face, so you do **not** need to upload this binary to your GitHub repo.

```bash
cd /home/mingqian/ws_ptv3/ptv3
curl -L \
  -o deploy_host/weights/scannet-pt-v3m1-0-base-model_best.pth \
  "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"

# verify
ls -lh deploy_host/weights/scannet-pt-v3m1-0-base-model_best.pth
```

Reference:
- https://huggingface.co/Pointcept/PointTransformerV3

## 4) Run Inference on Custom Data

```bash
cd /home/mingqian/ws_ptv3

IMAGE=ptv3:2080-fixed ./deploy_scripts/infer_custom_data_2080.sh \
  scannet-pt-v3m1-0-base-model_best.pth \
  /datasets/custom_scannet \
  val \
  /workspace/exp/custom/ptv3-2080-custom
```

Output files:
- `deploy_host/exp/custom/ptv3-2080-custom/result/*_pred.npy`
- `deploy_host/exp/custom/ptv3-2080-custom/result/submit/*.txt`

## 5) Export Comparison PLY (Input vs Prediction)

```bash
cd /home/mingqian/ws_ptv3
./deploy_scripts/visualize_custom_data.sh custom_scene_0001
```

Output folder:
- `deploy_host/exp/vis_custom/custom_scene_0001`

Generated files include:
- `*_input_rgb.ply`
- `*_pred_sem20.ply`
- `*_gt_sem20.ply` (dummy unlabeled baseline if no GT)
- `*_compare_input_vs_pred.ply`
- `*_compare_gt_vs_pred.ply`
- `*_pred_error_highlight.ply`
- `*_summary.json`

## 6) Popup Visualization (Open3D)

```bash
cd /home/mingqian/ws_ptv3

# mode: input | pred | gt | compare_input_pred | compare_gt_pred | error | all
./deploy_scripts/popup_custom_data_vis.sh custom_scene_0001 compare_input_pred
```

If popup does not show (headless or remote session), open the generated PLY files with CloudCompare or MeshLab.

## 7) Script Usage Reference

### `deploy_scripts/start_ptv3.sh`
- Purpose: create/start deployment container with mounted host folders.
- Optional env vars:
  - `IMAGE` (default: `ptv3:2080`)
  - `CONTAINER_NAME` (default: `ptv3-deploy`)
  - `HOST_DATA_DIR`, `HOST_WEIGHT_DIR`, `HOST_EXP_DIR`, `HOST_LOG_DIR`

### `deploy_scripts/shell_ptv3.sh`
- Purpose: open interactive shell in container.
- Optional env var: `CONTAINER_NAME`

### `deploy_scripts/stop_ptv3.sh`
- Purpose: stop running container.
- Optional env var: `CONTAINER_NAME`

### `deploy_scripts/prepare_my_data.sh`
- Purpose: convert raw custom data to PTv3-readable dataset layout.
- Args:
  - `$1` input dir (default: `deploy_host/data/my_data`)
  - `$2` output root (default: `deploy_host/data/custom_scannet`)
  - `$3` split (default: `val`)
  - `$4` scene prefix (default: `custom_scene`)

### `deploy_scripts/infer_custom_data_2080.sh`
- Purpose: run PTv3 inference on converted custom data.
- Args:
  - `$1` weight filename under `deploy_host/weights`
  - `$2` custom data root in container (default: `/datasets/custom_scannet`)
  - `$3` split/json name (default: `val`)
  - `$4` save path in container (default: `/workspace/exp/custom/ptv3-2080-custom`)
- Optional env var: `CONTAINER_NAME`

### `deploy_scripts/visualize_custom_data.sh`
- Purpose: export comparison PLY files.
- Args:
  - `$1` scene (default: `custom_scene_0001`)
  - `$2` result dir (default: custom result folder)
  - `$3` data root (default: `deploy_host/data/custom_scannet`)
  - `$4` output dir (default: `deploy_host/exp/vis_custom`)

### `deploy_scripts/popup_custom_data_vis.sh`
- Purpose: open Open3D popup viewer.
- Args:
  - `$1` scene (default: `custom_scene_0001`)
  - `$2` mode (default: `compare_input_pred`)
  - `$3` result dir (default: custom result folder)

### `tools/prepare_my_data_for_ptv3.py` (advanced)
```bash
python3 tools/prepare_my_data_for_ptv3.py --help
```

### `tools/visualize_compare.py` (advanced)
```bash
python3 tools/visualize_compare.py --help
```

### `tools/show_popup.py` (advanced)
```bash
python3 tools/show_popup.py --help
```

---

## Security Notes

- Do not put `HF_TOKEN` into README/scripts/git.
- Revoke and recreate tokens immediately if exposed.
