# igp/utils/gpu_memory.py
# GPU Memory Management and Optimization

from __future__ import annotations

import gc
import warnings
from contextlib import contextmanager
from typing import Any, List, Optional


class GPUMemoryManager:
    """
    Centralized GPU memory management with automatic cleanup.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._torch_available = False
        self._cuda_available = False
        
        try:
            import torch
            self.torch = torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            warnings.warn("PyTorch not available, GPU memory management disabled")
    
    def clear_cache(self, verbose: bool = False) -> None:
        """
        Clear PyTorch CUDA cache and run garbage collection.
        """
        if not self._cuda_available:
            return
        
        if verbose:
            before = self.get_memory_allocated()
        
        # Clear CUDA cache
        self.torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        if verbose:
            after = self.get_memory_allocated()
            freed = before - after
            print(f"[GPU] Freed {freed / 1024**2:.1f} MB, "
                  f"Now using {after / 1024**2:.1f} MB")
    
    def get_memory_allocated(self, device_id: int = 0) -> int:
        """Get currently allocated GPU memory in bytes."""
        if not self._cuda_available:
            return 0
        return self.torch.cuda.memory_allocated(device_id)
    
    def get_memory_reserved(self, device_id: int = 0) -> int:
        """Get reserved GPU memory in bytes."""
        if not self._cuda_available:
            return 0
        return self.torch.cuda.memory_reserved(device_id)
    
    def get_memory_summary(self, device_id: int = 0) -> dict:
        """Get detailed memory statistics."""
        if not self._cuda_available:
            return {}
        
        return {
            "allocated_mb": self.get_memory_allocated(device_id) / 1024**2,
            "reserved_mb": self.get_memory_reserved(device_id) / 1024**2,
            "max_allocated_mb": self.torch.cuda.max_memory_allocated(device_id) / 1024**2,
            "max_reserved_mb": self.torch.cuda.max_memory_reserved(device_id) / 1024**2,
        }
    
    def reset_peak_stats(self, device_id: int = 0) -> None:
        """Reset peak memory statistics."""
        if not self._cuda_available:
            return
        self.torch.cuda.reset_peak_memory_stats(device_id)
    
    @contextmanager
    def temporary_memory(self, clear_after: bool = True, verbose: bool = False):
        """
        Context manager for temporary GPU memory allocation.
        Automatically clears cache after exiting the context.
        
        Usage:
            with gpu_manager.temporary_memory():
                # GPU-intensive operations
                model(input_data)
            # Memory automatically cleared here
        """
        if verbose:
            print(f"[GPU] Before: {self.get_memory_allocated() / 1024**2:.1f} MB")
        
        try:
            yield self
        finally:
            if clear_after:
                self.clear_cache(verbose=verbose)
    
    def move_to_cpu_and_clear(self, *tensors, delete: bool = True) -> List[Any]:
        """
        Move tensors to CPU and clear GPU cache.
        
        Args:
            tensors: PyTorch tensors to move
            delete: If True, delete original tensors
            
        Returns:
            List of CPU tensors
        """
        if not self._torch_available:
            return list(tensors)
        
        cpu_tensors = []
        for tensor in tensors:
            if isinstance(tensor, self.torch.Tensor):
                cpu_tensor = tensor.detach().cpu()
                cpu_tensors.append(cpu_tensor)
                if delete:
                    del tensor
            else:
                cpu_tensors.append(tensor)
        
        if delete and self._cuda_available:
            self.torch.cuda.empty_cache()
            gc.collect()
        
        return cpu_tensors


# Global instance for easy access
_global_manager: Optional[GPUMemoryManager] = None


def get_gpu_manager(device: str = "cuda") -> GPUMemoryManager:
    """Get or create global GPU memory manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = GPUMemoryManager(device=device)
    return _global_manager


@contextmanager
def gpu_memory_context(clear_after: bool = True, verbose: bool = False):
    """
    Convenience context manager for GPU memory management.
    
    Usage:
        with gpu_memory_context():
            # Your GPU code here
            pass
    """
    manager = get_gpu_manager()
    with manager.temporary_memory(clear_after=clear_after, verbose=verbose):
        yield manager


def print_memory_stats(title: str = "GPU Memory", device_id: int = 0) -> None:
    """Print formatted memory statistics."""
    manager = get_gpu_manager()
    stats = manager.get_memory_summary(device_id)
    
    if not stats:
        print(f"{title}: CUDA not available")
        return
    
    print(f"\n{'='*60}")
    print(f"{title} Statistics")
    print(f"{'='*60}")
    print(f"  Allocated:     {stats['allocated_mb']:>8.1f} MB")
    print(f"  Reserved:      {stats['reserved_mb']:>8.1f} MB")
    print(f"  Peak Allocated:{stats['max_allocated_mb']:>8.1f} MB")
    print(f"  Peak Reserved: {stats['max_reserved_mb']:>8.1f} MB")
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = get_gpu_manager()
    
    # Print initial stats
    print_memory_stats("Initial")
    
    # Use context manager for temporary allocations
    try:
        import torch
        
        with gpu_memory_context(verbose=True) as mgr:
            # Allocate some memory
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = x @ y
            
            print(f"During allocation: {mgr.get_memory_allocated() / 1024**2:.1f} MB")
        
        # Memory automatically cleared
        print_memory_stats("After Context")
        
    except Exception as e:
        print(f"GPU test failed: {e}")
