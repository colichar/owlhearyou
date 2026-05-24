#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_PATH="$SCRIPT_DIR/../.venv"
PROJECT_ROOT="$SCRIPT_DIR/.."
SITE_PACKAGES="$VENV_PATH/lib/python3.12/site-packages"

ORT_PATH="$SITE_PACKAGES/onnxruntime/capi"

NV_PATHS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d | tr '\n' ':')

export LD_LIBRARY_PATH="$NV_PATHS$ORT_PATH:/opt/cuda/lib64:$LD_LIBRARY_PATH"

echo "Prioritizing pip NVIDIA libraries..."
echo "Launching from: $SCRIPT_DIR"

"$VENV_PATH/bin/python" "$PROJECT_ROOT/tests/manual/manual_transcription_test.py"
