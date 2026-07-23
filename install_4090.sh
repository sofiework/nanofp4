#!/usr/bin/env bash
# One-shot install for the nano-vllm `quant` branch on the 4090 (CUDA 13.0, Python 3.12).
#
# Usage:
#   conda create -n nanovllm python=3.12 -y && conda activate nanovllm
#   cd ~/nano-vllm && bash install_4090.sh
#
# Everything except flash-attn is a fast wheel download. flash-attn 2.8.3 likely has no prebuilt
# wheel for cu13/torch2.9/py312, so it may COMPILE from source (needs nvcc + ~30-60 min). If you
# don't have the CUDA toolkit, skip the build and copy the compiled .so from the source machine
# (see the "FALLBACK" note the script prints).
set -e
cd "$(dirname "$0")"

echo "==> [1/3] core deps (torch cu130 / triton / transformers ...)"
pip install -r requirements-4090.txt

echo "==> [2/3] flash-attn (torch is now present, so --no-build-isolation works)"
if ! pip install flash-attn==2.8.3 --no-build-isolation; then
  cat <<'EOF'

!!! flash-attn install failed (no matching wheel and/or no CUDA toolkit to build).
    FALLBACK — copy the compiled package from the source machine instead of building.
    On the source machine (fill in your ssh port/host), run:

      SP=~/miniconda3/envs/nanovllm/lib/python3.12/site-packages
      DST=$(python -c "import site;print(site.getsitepackages()[0])")   # <- run this line HERE first
      rsync -avP -e "ssh -p 2xxxx" \
        $SP/flash_attn $SP/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so \
        $SP/flash_attn-2.8.3.dist-info  linux@host:$DST/

    (The copied .so links against torch; it matches because torch==2.9.1+cu130 is pinned.)
EOF
  exit 1
fi

echo "==> [3/3] nano-vllm (editable)"
pip install -e . --no-deps

echo "==> verify"
python -c "import torch, triton, flash_attn, nanovllm; \
print('OK:', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
