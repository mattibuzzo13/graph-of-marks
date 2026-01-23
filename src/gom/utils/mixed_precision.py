# igp/utils/mixed_precision.py
# Mixed Precision Training/Inference Utilities
# Uses FP16 for 2x speedup with minimal accuracy loss

from __future__ import annotations

import contextlib
from functools import wraps
from typing import Any, Callable, Optional

import torch


class MixedPrecisionManager:
    """
    Mixed precision inference manager.
    
    Benefits:
    - 1.5-2x faster inference on modern GPUs (Volta+)
    - ~50% less GPU memory
    - Minimal accuracy loss (<1%)
    
    Usage:
        >>> mp = MixedPrecisionManager(enabled=True)
        >>> with mp.autocast():
        ...     output = model(input)  # Runs in FP16
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: Optional[torch.dtype] = None,
        cache_enabled: bool = True,
    ):
        """
        Args:
            enabled: Enable mixed precision (auto-detect if GPU supports it)
            dtype: torch.float16 or torch.bfloat16 (auto-select if None)
            cache_enabled: Enable cudnn benchmark for repeated ops
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        # Auto-select dtype based on GPU capability
        if dtype is None:
            if self.enabled:
                # Check if GPU supports BF16 (better range than FP16)
                if torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                    print("[MixedPrecision] Using BF16 (GPU supports it)")
                else:
                    self.dtype = torch.float16
                    print("[MixedPrecision] Using FP16")
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype
        
        # Enable cudnn benchmark for faster inference
        if cache_enabled and self.enabled:
            torch.backends.cudnn.benchmark = True
            print("[MixedPrecision] Enabled cudnn benchmark")
    
    @contextlib.contextmanager
    def autocast(self):
        """
        Context manager for automatic mixed precision.
        
        Example:
            >>> with mp.autocast():
            ...     output = model(input)
        """
        if self.enabled:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            # Use new torch.amp.autocast API with explicit device_type/dtype
            with torch.amp.autocast(device_type=device_type, dtype=self.dtype):
                yield
        else:
            yield
    
    def convert_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Convert model to FP16/BF16.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model converted to lower precision
        """
        if not self.enabled:
            return model
        
        if self.dtype == torch.float16:
            model = model.half()
            print("[MixedPrecision] Converted model to FP16")
        elif self.dtype == torch.bfloat16:
            model = model.to(dtype=torch.bfloat16)
            print("[MixedPrecision] Converted model to BF16")
        
        return model
    
    def convert_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to appropriate dtype."""
        if not self.enabled:
            return tensor
        
        if tensor.dtype == torch.float32:
            return tensor.to(dtype=self.dtype)
        
        return tensor


def autocast_inference(func: Callable) -> Callable:
    """
    Decorator for automatic mixed precision inference.
    
    Usage:
        >>> @autocast_inference
        ... def detect(self, image):
        ...     return self.model(image)  # Runs in FP16
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if CUDA available
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    return wrapper


# Global instance for convenience
_global_mp_manager: Optional[MixedPrecisionManager] = None


def get_mp_manager(enabled: bool = True) -> MixedPrecisionManager:
    """Get or create global mixed precision manager."""
    global _global_mp_manager
    
    if _global_mp_manager is None:
        _global_mp_manager = MixedPrecisionManager(enabled=enabled)
    
    return _global_mp_manager


def enable_mixed_precision(dtype: Optional[torch.dtype] = None):
    """Enable mixed precision globally."""
    global _global_mp_manager
    _global_mp_manager = MixedPrecisionManager(enabled=True, dtype=dtype)
    print("[MixedPrecision] Enabled globally")


def disable_mixed_precision():
    """Disable mixed precision globally."""
    global _global_mp_manager
    _global_mp_manager = MixedPrecisionManager(enabled=False)
    print("[MixedPrecision] Disabled globally")


@contextlib.contextmanager
def autocast(enabled: bool = True):
    """
    Standalone autocast context manager.
    
    Usage:
        >>> with autocast():
        ...     output = model(input)  # FP16
    """
    if enabled and torch.cuda.is_available():
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


class OptimizedInferenceMixin:
    """
    Mixin class to add optimized inference to any detector/segmenter.
    
    Usage:
        >>> class MyDetector(OptimizedInferenceMixin, BaseDetector):
        ...     def detect(self, image):
        ...         return self._optimized_forward(image)
    """
    
    def __init__(self, *args, use_mixed_precision: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mp_manager = MixedPrecisionManager(enabled=use_mixed_precision)
        self._warmup_done = False
    
    def _optimized_forward(self, *args, **kwargs) -> Any:
        """Run forward pass with optimizations."""
        with self.mp_manager.autocast():
            return self._forward_impl(*args, **kwargs)
    
    def _forward_impl(self, *args, **kwargs) -> Any:
        """Override this in subclass."""
        raise NotImplementedError
    
    def warmup(self, input_shape: Optional[tuple] = None):
        """Warmup model to avoid cold start."""
        if self._warmup_done:
            return
        
        print("[Optimization] Warming up model...")
        
        # Create dummy input
        if input_shape is None:
            input_shape = (1, 3, 640, 640)  # Default
        
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Run dummy forward pass
        with torch.no_grad():
            with self.mp_manager.autocast():
                _ = self.model(dummy_input)
        
        self._warmup_done = True
        print("[Optimization] Warmup complete")
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        if hasattr(self, 'model'):
            return next(self.model.parameters()).device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply all inference optimizations to a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Optimized model
    """
    # 1. Set to eval mode
    model.eval()
    
    # 2. Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # 3. Convert to FP16 if on GPU
    if torch.cuda.is_available():
        mp = get_mp_manager()
        model = mp.convert_model(model)
    
    # 4. Enable cudnn benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 5. Compile model (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            print("[Optimization] Model compiled (PyTorch 2.0+)")
    except Exception as e:
        print(f"[Optimization] Compile failed (not critical): {e}")
    
    print("[Optimization] Model optimized for inference")
    return model


def benchmark_mixed_precision(
    model: torch.nn.Module,
    input_shape: tuple = (1, 3, 640, 640),
    num_runs: int = 100,
) -> dict:
    """
    Benchmark FP32 vs FP16 performance.
    
    Returns:
        Dict with timing and memory stats
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = torch.randn(input_shape, device=device)
    
    # FP32 benchmark
    print("[Benchmark] Testing FP32...")
    model = model.float()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end.record()
        
        torch.cuda.synchronize()
        fp32_time = start.elapsed_time(end) / num_runs
        fp32_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # FP16 benchmark
    print("[Benchmark] Testing FP16...")
    model = model.half()
    dummy_input = dummy_input.half()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end.record()
        
        torch.cuda.synchronize()
        fp16_time = start.elapsed_time(end) / num_runs
        fp16_memory = torch.cuda.max_memory_allocated() / 1e9
    
    results = {
        "fp32_time_ms": fp32_time,
        "fp16_time_ms": fp16_time,
        "speedup": fp32_time / fp16_time,
        "fp32_memory_gb": fp32_memory,
        "fp16_memory_gb": fp16_memory,
        "memory_savings": (fp32_memory - fp16_memory) / fp32_memory * 100,
    }
    
    print(f"[Benchmark] FP32: {fp32_time:.2f}ms, {fp32_memory:.2f}GB")
    print(f"[Benchmark] FP16: {fp16_time:.2f}ms, {fp16_memory:.2f}GB")
    print(f"[Benchmark] Speedup: {results['speedup']:.2f}x")
    print(f"[Benchmark] Memory savings: {results['memory_savings']:.1f}%")
    
    return results
