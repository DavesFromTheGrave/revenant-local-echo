import gc
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False
    logger.warning("torch not available — VRAMManager will report zero usage")


class VRAMManager:
    """Explicit VRAM lifecycle management for sequential GPU load/unload."""

    @staticmethod
    def check_cuda_available() -> bool:
        """Verify CUDA is available."""
        if not _TORCH_OK:
            return False
        available = torch.cuda.is_available()
        if available:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM total: {total:.1f} GB")
        else:
            logger.warning("CUDA not available — GPU operations will fail")
        return available

    @staticmethod
    def get_vram_total() -> float:
        """Total VRAM in MB (for compatibility with main.py)."""
        if _TORCH_OK and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e6
        return 0.0

    @staticmethod
    def get_vram_usage() -> float:
        """Get current VRAM usage in GB."""
        if _TORCH_OK and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    @staticmethod
    def get_vram_available() -> float:
        """Get available VRAM in GB."""
        if _TORCH_OK and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            used = torch.cuda.memory_allocated() / 1e9
            return total - used
        return 0.0

    @staticmethod
    def clear_cache(component: str = None):
        """Explicitly release VRAM and run garbage collection."""
        gc.collect()

        if _TORCH_OK and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        used = VRAMManager.get_vram_usage()
        available = VRAMManager.get_vram_available()

        label = f"[{component}] " if component else ""
        logger.debug(f"{label}VRAM cleared. Used: {used:.2f}GB, Available: {available:.2f}GB")

    @staticmethod
    def log_vram_snapshot(label: str):
        """Log current VRAM state with a label."""
        used = VRAMManager.get_vram_usage()
        available = VRAMManager.get_vram_available()
        logger.info(f"[{label}] VRAM: {used:.2f}GB used, {available:.2f}GB available")

    @staticmethod
    def model_to_device(model, device: str = "cuda"):
        """Move model to device with logging."""
        if not _TORCH_OK:
            return model
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            logger.debug(f"Model moved to CUDA. VRAM used: {VRAMManager.get_vram_usage():.2f}GB")
        else:
            model = model.to("cpu")
            logger.debug("Model moved to CPU")
        return model
