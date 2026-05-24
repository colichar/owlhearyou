# Installation

Requires Python 3.12 and a CUDA-capable GPU.

## 1. Create the virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 2. Install onnxruntime-gpu

`sherpa-onnx` requires a specific version of ONNX Runtime. Install it pinned to CUDA 12:

**CUDA 12:**
```bash
pip install onnxruntime-gpu
```

**CUDA 13+ (e.g. CachyOS/Arch):** The standard PyPI wheel links against system CUDA 13 libraries, which are ABI-incompatible with `sherpa-onnx`. Install the pinned CUDA 12 wheel from the Microsoft index instead:
```bash
pip install onnxruntime-gpu==1.23.2 \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## 3. Install sherpa-onnx with CUDA support

The PyPI wheel is CPU-only. Install the CUDA wheel from the k2-fsa index:

```bash
pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

## 4. (CUDA 13+ only) Install portable CUDA 12 libraries

Install the CUDA 12 `.so` files directly into the venv so the system CUDA 13 libraries are never used at runtime:

```bash
pip install \
  nvidia-cublas-cu12 \
  nvidia-cufft-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cuda-runtime-cu12
```

## 5. (CUDA 13+ only) Create a symlink for the ONNX Runtime library

The dynamic linker looks for `libonnxruntime.so` (unversioned) but the pip package only ships `libonnxruntime.so.1.23.2`. Create the symlink manually:

```bash
cd .venv/lib/python3.12/site-packages/onnxruntime/capi/
ln -s libonnxruntime.so.1.23.2 libonnxruntime.so
cd -
```

## 6. Install project dependencies

```bash
pip install -e ".[dev]"

# Optional: add the Whisper backend
pip install -e ".[dev,whisper]"
```

## 7. Launch with scripts/run.sh

Always launch via `./scripts/run.sh`, never `uvicorn` directly. `run.sh` sets `LD_LIBRARY_PATH` to ensure the venv's CUDA libraries are loaded before any system CUDA libraries:

```bash
./scripts/run.sh
```
