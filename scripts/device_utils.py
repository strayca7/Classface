"""
Shared device detection utilities for all scripts.

ONNX Runtime priority (InsightFace): CUDA > CoreML (Apple) > CPU
PyTorch priority                    : CUDA > MPS  (Apple) > CPU
"""

import logging

logger = logging.getLogger(__name__)


def get_ort_providers() -> list[str]:
    """Return ONNX Runtime execution providers in priority order.

    Priority:
      1. CUDAExecutionProvider   — NVIDIA GPU (e.g. RTX 4060)
      2. CoreMLExecutionProvider — Apple Silicon (M-series, uses GPU / Neural Engine)
      3. CPUExecutionProvider    — fallback

    Only providers that are actually installed/available are returned.
    """
    import onnxruntime

    available = set(onnxruntime.get_available_providers())
    priority = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [p for p in priority if p in available]
    logger.info("ORT providers selected: %s", selected)
    return selected


def get_device():
    """Return the best available torch.device: CUDA > MPS > CPU.

    Returns:
        torch.device
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("PyTorch device: CUDA — %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("PyTorch device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("PyTorch device: CPU (no GPU detected)")
    return device
