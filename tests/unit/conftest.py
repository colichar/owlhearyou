import sys
from unittest.mock import MagicMock

# sherpa_onnx links against native ONNX/CUDA libraries that aren't available in CI.
# Register a stub before any test module imports src.transcriber so the import
# succeeds without the real shared library.
sys.modules.setdefault("sherpa_onnx", MagicMock())
