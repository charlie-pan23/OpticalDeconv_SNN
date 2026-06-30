#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Freeze post-training artifacts for HIPSA evaluation.
# Run from repository root:
#   ./scripts/freeze_eval_artifacts.sh
#
# Optional:
#   ./scripts/freeze_eval_artifacts.sh eval_v2_20260630
# ============================================================

TAG="${1:-eval_v2_$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="$(pwd)"
OUT_BASE="${ROOT_DIR}/frozen_artifacts"
OUT_DIR="${OUT_BASE}/${TAG}"
TAR_PATH="${OUT_BASE}/${TAG}.tar.gz"

mkdir -p "${OUT_DIR}"

echo "[INFO] Freezing artifacts into: ${OUT_DIR}"

# -----------------------------
# 1. Basic directory structure
# -----------------------------
mkdir -p \
  "${OUT_DIR}/checkpoints/cifar10dvs" \
  "${OUT_DIR}/checkpoints/dvsgesture" \
  "${OUT_DIR}/configs/cifar10dvs" \
  "${OUT_DIR}/configs/dvsgesture" \
  "${OUT_DIR}/train_runs/cifar10dvs" \
  "${OUT_DIR}/train_runs/dvsgesture" \
  "${OUT_DIR}/test_runs/cifar10dvs" \
  "${OUT_DIR}/test_runs/dvsgesture" \
  "${OUT_DIR}/results_json_csv" \
  "${OUT_DIR}/eval_scripts_snapshot" \
  "${OUT_DIR}/model_code_snapshot" \
  "${OUT_DIR}/repo_metadata"

# -----------------------------
# 2. Helper copy functions
# -----------------------------
copy_if_exists() {
  local src="$1"
  local dst="$2"

  if [ -e "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    echo "[COPY] ${src} -> ${dst}"
  else
    echo "[WARN] Missing: ${src}"
  fi
}

copy_dir_if_exists() {
  local src="$1"
  local dst="$2"

  if [ -d "${src}" ]; then
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    echo "[COPY_DIR] ${src} -> ${dst}"
  else
    echo "[WARN] Missing dir: ${src}"
  fi
}

# -----------------------------
# 3. Main checkpoints
# -----------------------------
copy_if_exists \
  "results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth" \
  "${OUT_DIR}/checkpoints/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth"

copy_if_exists \
  "results/dvsgesture/best_dvsgesture_acc88p54.pth" \
  "${OUT_DIR}/checkpoints/dvsgesture/best_dvsgesture_acc88p54.pth"

# -----------------------------
# 4. Main configs
# -----------------------------
copy_if_exists \
  "configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml" \
  "${OUT_DIR}/configs/cifar10dvs/config_cifar10dvs_clip3_b96_wd001_do03.yaml"

copy_if_exists \
  "results/dvsgesture/config_dvsgesture_acc88p54.yaml" \
  "${OUT_DIR}/configs/dvsgesture/config_dvsgesture_acc88p54.yaml"

copy_if_exists "configs/hardware_hipsa.yaml" "${OUT_DIR}/configs/hardware_hipsa.yaml"
copy_if_exists "configs/device_params.yaml" "${OUT_DIR}/configs/device_params.yaml"

# -----------------------------
# 5. Original train and formal test runs
# -----------------------------
copy_dir_if_exists \
  "results/cifar10dvs/cifar10dvs/run_20260629_194703_cifar10dvs" \
  "${OUT_DIR}/train_runs/cifar10dvs/run_20260629_194703_cifar10dvs"

copy_dir_if_exists \
  "results/cifar10dvs/run_20260629_223733_cifar10dvs" \
  "${OUT_DIR}/test_runs/cifar10dvs/run_20260629_223733_cifar10dvs"

copy_dir_if_exists \
  "results/dvsgesture/run_20260629_191844_dvsgesture" \
  "${OUT_DIR}/train_runs/dvsgesture/run_20260629_191844_dvsgesture"

copy_if_exists \
  "results/dvsgesture/final_test_acc88p54.json" \
  "${OUT_DIR}/test_runs/dvsgesture/final_test_acc88p54.json"

# -----------------------------
# 6. Existing JSON / CSV / YAML / TXT / MD / LOG results
#    Avoid copying extra .pth files except the two main checkpoints above.
# -----------------------------
echo "[INFO] Collecting lightweight result files..."

while IFS= read -r f; do
  rel="${f#./}"
  dst="${OUT_DIR}/results_json_csv/${rel}"
  mkdir -p "$(dirname "${dst}")"
  cp -a "${f}" "${dst}"
done < <(
  find ./results ./logs ./configs \
    -type f \( \
      -name "*.json" -o \
      -name "*.csv" -o \
      -name "*.yaml" -o \
      -name "*.yml" -o \
      -name "*.txt" -o \
      -name "*.md" -o \
      -name "*.log" \
    \) 2>/dev/null | sort
)

# -----------------------------
# 7. Snapshot current scripts and model code
# -----------------------------
copy_dir_if_exists "eval" "${OUT_DIR}/eval_scripts_snapshot/eval"
copy_dir_if_exists "models" "${OUT_DIR}/model_code_snapshot/models"
copy_dir_if_exists "train" "${OUT_DIR}/model_code_snapshot/train"
copy_dir_if_exists "utils" "${OUT_DIR}/model_code_snapshot/utils"
copy_dir_if_exists "hardware" "${OUT_DIR}/model_code_snapshot/hardware"

copy_if_exists "README.md" "${OUT_DIR}/repo_metadata/README.md"

# -----------------------------
# 8. Git and environment metadata
# -----------------------------
{
  echo "TAG=${TAG}"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "DATE_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "HOSTNAME=$(hostname || true)"
  echo "UNAME=$(uname -a || true)"
  echo "PWD=$(pwd)"
} > "${OUT_DIR}/repo_metadata/freeze_info.txt"

git rev-parse HEAD > "${OUT_DIR}/repo_metadata/git_commit.txt" 2>/dev/null || true
git branch --show-current > "${OUT_DIR}/repo_metadata/git_branch.txt" 2>/dev/null || true
git status --short > "${OUT_DIR}/repo_metadata/git_status.txt" 2>/dev/null || true
git diff > "${OUT_DIR}/repo_metadata/git_diff.patch" 2>/dev/null || true

python - <<'PY' > "${OUT_DIR}/repo_metadata/python_env.txt"
import sys, platform, subprocess
print("python:", sys.version)
print("platform:", platform.platform())
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch_info_error:", repr(e))
try:
    import snntorch
    print("snntorch:", snntorch.__version__)
except Exception as e:
    print("snntorch_info_error:", repr(e))
PY

pip freeze > "${OUT_DIR}/repo_metadata/pip_freeze.txt" 2>/dev/null || true

# -----------------------------
# 9. Human-readable manifest
# -----------------------------
cat > "${OUT_DIR}/MANIFEST.md" <<EOF
# HIPSA Frozen Evaluation Package

Tag: ${TAG}

## Main frozen checkpoints

### CIFAR10-DVS
- Model: SpikingVGGGAP
- Timestep: T=10
- Encoding: clipped_count max=3
- Best Val: 73.30%
- Test Acc: 76.40%
- Test Loss: 0.9317
- Checkpoint: checkpoints/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth
- Config: configs/cifar10dvs/config_cifar10dvs_clip3_b96_wd001_do03.yaml

### DVS Gesture
- Model: SpikingGestureCNN
- Timestep: T=10
- Encoding: binary
- Best Val: 91.67%
- Test Acc: 88.54%
- Test Loss: 0.4111
- Train / Val / Test samples: 1056 / 120 / 288
- Best checkpoint epoch: 46
- Checkpoint: checkpoints/dvsgesture/best_dvsgesture_acc88p54.pth
- Config: configs/dvsgesture/config_dvsgesture_acc88p54.yaml

## Notes

This package freezes training outputs before re-running or rewriting evaluation.
Evaluation scripts inside eval_scripts_snapshot are snapshots only and should be audited before final paper results.
EOF

# -----------------------------
# 10. Machine-readable manifest
# -----------------------------
python - <<PY > "${OUT_DIR}/manifest.json"
import json, os, subprocess, datetime, platform

def safe(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return None

manifest = {
    "tag": "${TAG}",
    "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "root_dir": "${ROOT_DIR}",
    "git_commit": safe("git rev-parse HEAD"),
    "git_branch": safe("git branch --show-current"),
    "platform": platform.platform(),
    "datasets": {
        "cifar10dvs": {
            "model": "SpikingVGGGAP",
            "timestep": 10,
            "encoding": "clipped_count",
            "clipped_count_max": 3,
            "best_val_percent": 73.30,
            "test_acc_percent_user_record": 76.40,
            "test_loss_user_record": 0.9317,
            "checkpoint": "checkpoints/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth",
            "config": "configs/cifar10dvs/config_cifar10dvs_clip3_b96_wd001_do03.yaml",
            "original_train_run": "train_runs/cifar10dvs/run_20260629_194703_cifar10dvs",
            "formal_test_run": "test_runs/cifar10dvs/run_20260629_223733_cifar10dvs"
        },
        "dvsgesture": {
            "model": "SpikingGestureCNN",
            "timestep": 10,
            "encoding": "binary",
            "best_val_percent": 91.67,
            "test_acc_percent_user_record": 88.54,
            "test_loss_user_record": 0.4111,
            "train_samples": 1056,
            "val_samples": 120,
            "test_samples": 288,
            "best_checkpoint_epoch": 46,
            "checkpoint": "checkpoints/dvsgesture/best_dvsgesture_acc88p54.pth",
            "config": "configs/dvsgesture/config_dvsgesture_acc88p54.yaml",
            "final_test_result": "test_runs/dvsgesture/final_test_acc88p54.json",
            "original_train_run": "train_runs/dvsgesture/run_20260629_191844_dvsgesture",
            "formal_test_run": "train_runs/dvsgesture/run_20260629_191844_dvsgesture"
        }
    },
    "intended_next_steps": [
        "Audit eval scripts before trusting final paper numbers",
        "Re-run clean accuracy sanity check",
        "Generate activity traces",
        "Run threshold / HAPR / ADC pool sensitivity",
        "Run device-specific robustness",
        "Generate paper figures and tables"
    ]
}
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

# -----------------------------
# 11. Checksums
# -----------------------------
echo "[INFO] Computing SHA256 checksums..."
(
  cd "${OUT_DIR}"
  find . -type f | sort | while read -r f; do
    sha256sum "$f"
  done
) > "${OUT_DIR}/sha256sums.txt"

# -----------------------------
# 12. File tree
# -----------------------------
if command -v tree >/dev/null 2>&1; then
  tree -ah "${OUT_DIR}" > "${OUT_DIR}/repo_metadata/file_tree.txt"
else
  find "${OUT_DIR}" -print | sort > "${OUT_DIR}/repo_metadata/file_tree.txt"
fi

# -----------------------------
# 13. Tarball
# -----------------------------
echo "[INFO] Creating tarball: ${TAR_PATH}"
mkdir -p "${OUT_BASE}"
tar -czf "${TAR_PATH}" -C "${OUT_BASE}" "${TAG}"

echo
echo "[DONE] Frozen package created:"
echo "  Directory: ${OUT_DIR}"
echo "  Tarball:   ${TAR_PATH}"
echo
echo "[NEXT] Verify checksum:"
echo "  cd ${OUT_DIR} && sha256sum -c sha256sums.txt"