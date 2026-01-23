# igp/utils/depth_v2.py
# Optimized Depth Estimator with support for multiple SOTA models:
# - Depth Anything V2 (2024) - SOTA monocular depth estimation
# - MiDaS v3.1 DPT-Large (fallback)
# - Optimized with: lazy loading, model caching, warmup, mixed precision

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    cv2 = None  # type: ignore

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    torch = None  # type: ignore


class DepthModel(str, Enum):
    """Supported depth estimation models."""
    DEPTH_ANYTHING_V2_SMALL = "depth_anything_v2_vits"
    DEPTH_ANYTHING_V2_BASE = "depth_anything_v2_vitb"  
    DEPTH_ANYTHING_V2_LARGE = "depth_anything_v2_vitl"  # Recommended: best accuracy
    MIDAS_DPT_LARGE = "DPT_Large"  # Fallback
    MIDAS_DPT_HYBRID = "DPT_Hybrid"


@dataclass
class DepthConfig:
    """
    Configuration for depth estimation.

    Depth Anything V2 models:
    - vits (Small): Fast, ~5 FPS on CPU, 50+ FPS on GPU
    - vitb (Base): Balanced, ~3 FPS on CPU, 40+ FPS on GPU
    - vitl (Large): Best quality, ~2 FPS on CPU, 30+ FPS on GPU

    MiDaS models:
    - DPT_Large: High quality, slower
    - DPT_Hybrid: Balanced quality/speed

    Optimization flags:
    - lazy_load: Load model only when first needed (saves ~2-5s startup)
    - enable_compile: Use torch.compile for 30% speedup (PyTorch 2.0+)
    - warmup: Run dummy inference for JIT compilation
    """
    model_name: DepthModel = DepthModel.DEPTH_ANYTHING_V2_LARGE
    device: Optional[str] = None
    fp16_on_cuda: bool = True  # Mixed precision for 2x speedup
    cache_maps: bool = True  # Cache depth maps per image hash
    max_cache_size: int = 100  # Maximum cached depth maps
    
    # Loading optimizations
    lazy_load: bool = True  # Defer model loading until first use
    enable_compile: bool = True  # torch.compile for speedup (PyTorch 2.0+)
    warmup: bool = False  # Run warmup inference (adds ~0.5s init time)
    download_timeout: int = 300  # HuggingFace download timeout (seconds)
    show_download_progress: bool = True  # Show progress bar during download


class DepthEstimatorV2:
    """
    Multi-model depth estimator supporting:
    - Depth Anything V2 (SOTA 2024)
    - MiDaS v3.1

    Features:
    - Lazy loading (load model only when first needed)
    - Intelligent caching (avoid recomputing same images)
    - Mixed precision FP16 (2x GPU speedup)
    - torch.compile support (30% speedup on PyTorch 2.0+)
    - Batch processing support
    - Normalized output [0, 1] where 1.0 = closer to camera
    """
    
    def __init__(self, config: Optional[DepthConfig] = None) -> None:
        self.config = config or DepthConfig()
        self._ok = bool(_HAS_TORCH)
        self.model = None
        self.transform = None
        self.device = "cpu"
        self._amp_enabled = False
        self._amp_dtype = None
        self._depth_cache: Dict[str, np.ndarray] = {}  # image_hash -> depth_map
        self._model_type = None
        self._model_loaded = False

        if not self._ok:
            return

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
        self._amp_enabled = (self.device == "cuda" and bool(self.config.fp16_on_cuda))
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32  # type: ignore[attr-defined]

        # Load model immediately or defer to first use
        if not self.config.lazy_load:
            self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded (lazy loading support)."""
        if self._model_loaded:
            return

        # Load model into instance variables
        self._load_model()
        self._model_loaded = True

        # Optional warmup
        if self.config.warmup:
            self._warmup()
    
    def _load_model(self) -> None:
        """Load the specified depth model."""
        model_name = self.config.model_name.value
        
        try:
            if "depth_anything_v2" in model_name:
                self._load_depth_anything_v2(model_name)
                self._model_type = "depth_anything_v2"
            elif model_name in ["DPT_Large", "DPT_Hybrid"]:
                self._load_midas(model_name)
                self._model_type = "midas"
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except Exception as e:
            print(f"[WARNING] Failed to load {model_name}: {e}")
            print("[WARNING] Falling back to MiDaS DPT_Large")
            self._load_midas("DPT_Large")
            self._model_type = "midas"
    
    def _warmup(self) -> None:
        """
        Warmup model with dummy inference for JIT compilation optimization.
        
        This runs a dummy forward pass to:
        - Trigger JIT compilation (torch.compile)
        - Optimize CUDA kernel selection
        - Pre-allocate GPU memory
        
        Recommended before batch processing for 10-20% speedup.
        """
        if not self.config.warmup:
            return
            
        print("[DEPTH] Warming up model...")
        
        try:
            # Create dummy input matching expected size
            dummy_size = (518, 518)  # Depth Anything V2 default
            dummy_input = torch.randn(
                1, 3, dummy_size[0], dummy_size[1],
                device=self.device,
                dtype=torch.float32
            )
            
            # Run dummy inference
            with torch.no_grad():
                if self.model_type == "depth_anything_v2":
                    _ = self.model(dummy_input)
                else:  # midas
                    _ = self.model(dummy_input)
            
            # Sync and clear
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            print("[DEPTH] Warmup complete")
            
        except Exception as e:
            print(f"[DEPTH] Warmup failed (skipping): {e}")

    def _load_depth_anything_v2(self, model_name: str) -> None:
        """
        Load Depth Anything V2 with optimized download and caching.
        
        Optimizations:
        - Progress bar during download
        - Retry logic for network issues
        - Local caching to avoid re-downloads
        - Automatic torch.compile for speedup
        
        Model variants:
        - depth_anything_v2_vits (Small, encoder='vits', features=64)
        - depth_anything_v2_vitb (Base, encoder='vitb', features=128)
        - depth_anything_v2_vitl (Large, encoder='vitl', features=256)
        """
        try:
            from huggingface_hub import hf_hub_download

            # Map model name to configuration
            model_configs = {
                "depth_anything_v2_vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                    "hf_repo": "depth-anything/Depth-Anything-V2-Small"
                },
                "depth_anything_v2_vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                    "hf_repo": "depth-anything/Depth-Anything-V2-Base"
                },
                "depth_anything_v2_vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                    "hf_repo": "depth-anything/Depth-Anything-V2-Large"
                },
            }
            
            config = model_configs.get(model_name, model_configs["depth_anything_v2_vitl"])
            encoder = config["encoder"]
            
            print(f"[DEPTH] Loading Depth Anything V2 ({encoder})...")
            
            # Download checkpoint with retry and cache support
            max_retries = 3
            checkpoint_path = None
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if offline mode is forced via environment
            hf_offline = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")
            
            for attempt in range(max_retries):
                try:
                    # Always try with local_files_only=False unless HF_HUB_OFFLINE is set
                    checkpoint_path = hf_hub_download(
                        repo_id=config["hf_repo"],
                        filename=f"depth_anything_v2_{encoder}.pth",
                        repo_type="model",
                        cache_dir=str(cache_dir),
                        local_files_only=hf_offline,  # Respect env var, allow downloads otherwise
                    )
                    
                    # Log cache hit vs download
                    cache_path = Path(checkpoint_path)
                    if cache_path.exists():
                        import time
                        age_seconds = time.time() - cache_path.stat().st_mtime
                        if age_seconds < 5:  # Modified in last 5 seconds
                            print(f"[DEPTH] Checkpoint downloaded: {cache_path.name}")
                        else:
                            print(f"[DEPTH] Using cached checkpoint: {cache_path.name}")
                    
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time as time_mod
                        print(f"[DEPTH] Download attempt {attempt + 1} failed, retrying in 2s...")
                        time_mod.sleep(2)  # Wait before retry
                    else:
                        raise RuntimeError(f"Failed to download after {max_retries} attempts: {e}")
            
            # Import Depth Anything V2 - Efficient approach
            # Strategy: Copy only necessary files to src/ instead of cloning full repo
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
                print(f"[DEPTH] Using installed depth_anything_v2 module")
            except ImportError:
                # Check if cached clone exists, and copy module to src/
                cache_dir = Path.home() / ".cache" / "depth_anything_v2"
                src_depth_v2_dir = Path(__file__).parent.parent.parent / "depth_anything_v2"
                
                if src_depth_v2_dir.exists():
                    # Module already in src/, just need to import
                    import sys
                    sys.path.insert(0, str(src_depth_v2_dir.parent))
                    try:
                        from depth_anything_v2.dpt import DepthAnythingV2
                        print(f"[DEPTH] Using depth_anything_v2 from {src_depth_v2_dir}")
                    except ImportError as e:
                        raise RuntimeError(f"Module exists but import failed: {e}")
                        
                elif cache_dir.exists():
                    # Copy from cache to src/ (one-time operation)
                    print(f"[DEPTH] Copying depth_anything_v2 module to src/ (one-time setup)...")
                    import shutil
                    source_module = cache_dir / "depth_anything_v2"
                    
                    if source_module.exists():
                        try:
                            shutil.copytree(source_module, src_depth_v2_dir)
                            print(f"[DEPTH] Module copied to {src_depth_v2_dir}")
                            
                            # Import after copy
                            import sys
                            sys.path.insert(0, str(src_depth_v2_dir.parent))
                            from depth_anything_v2.dpt import DepthAnythingV2
                            
                        except Exception as e:
                            raise RuntimeError(f"Failed to copy module: {e}")
                    else:
                        raise RuntimeError(f"Source module not found: {source_module}")
                else:
                    # Need to clone first (fallback for first-time setup)
                    print("[DEPTH] First-time setup: cloning Depth Anything V2...")
                    import subprocess
                    
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run([
                            "git", "clone", "--depth", "1",
                            "https://github.com/DepthAnything/Depth-Anything-V2.git",
                            str(cache_dir)
                        ], check=True, timeout=self.config.download_timeout)
                        
                        # Copy module to src/
                        source_module = cache_dir / "depth_anything_v2"
                        if source_module.exists():
                            import shutil
                            shutil.copytree(source_module, src_depth_v2_dir)
                            print(f"[DEPTH] Module setup complete at {src_depth_v2_dir}")
                            
                            # Import
                            import sys
                            sys.path.insert(0, str(src_depth_v2_dir.parent))
                            from depth_anything_v2.dpt import DepthAnythingV2
                        else:
                            raise RuntimeError(f"Module not found after clone: {source_module}")
                            
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(f"Clone timeout after {self.config.download_timeout}s")
                    except Exception as e:
                        raise RuntimeError(f"Setup failed: {e}")
            
            # Initialize model
            print(f"[DEPTH] Initializing Depth Anything V2 ({encoder})...")
            self.model = DepthAnythingV2(
                encoder=encoder,
                features=config["features"],
                out_channels=config["out_channels"]
            )
            
            # Load weights with memory optimization
            print(f"[DEPTH] Loading weights...")
            state_dict = torch.load(
                checkpoint_path, 
                map_location="cpu",  # Load to CPU first
                weights_only=True  # Security: only load weights
            )
            self.model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            
            # Free CPU memory
            del state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # No separate transform needed for Depth Anything V2
            self.transform = None
            
            # torch.compile for 30% speedup (PyTorch 2.0+)
            if self.config.enable_compile and hasattr(torch, "compile") and self.device == "cuda":
                try:
                    print("[DEPTH] Compiling model with torch.compile...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("[DEPTH] torch.compile enabled")
                except Exception as e:
                    print(f"[DEPTH] torch.compile failed (skipping): {e}")
            
            print(f"[DEPTH] Depth Anything V2 ({encoder}) loaded successfully")
                    
        except Exception as e:
            print(f"[WARNING] Failed to load Depth Anything V2: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Falling back to MiDaS")
            raise
    
    def _load_midas(self, model_name: str) -> None:
        """
        Load MiDaS model via torch.hub with explicit caching.
        
        torch.hub automatically caches models in ~/.cache/torch/hub
        """
        print(f"[DEPTH] Loading MiDaS: {model_name}")
        
        # Set torch hub cache directory explicitly
        torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
        
        # Load model (torch.hub automatically uses cache if available)
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self.model = torch.hub.load(
                    "intel-isl/MiDaS", 
                    model_name,
                    skip_validation=False,  # Validate cache
                    trust_repo=True,  # Suppress untrusted repo warning
                    verbose=True  # Show cache hits/misses
                ).to(self.device).eval()  # type: ignore[attr-defined]
                
                transforms = torch.hub.load(
                    "intel-isl/MiDaS", 
                    "transforms",
                    skip_validation=False,
                    trust_repo=True,  # Suppress untrusted repo warning
                    verbose=False  # Less verbose for transforms
                )  # type: ignore[attr-defined]
                
                if model_name.startswith("DPT"):
                    self.transform = transforms.dpt_transform
                else:
                    self.transform = transforms.small_transform
                    
                print(f"[DEPTH] MiDaS {model_name} loaded successfully")
                return  # Success, exit function
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time as time_mod
                    print(f"[DEPTH] MiDaS load attempt {attempt + 1} failed, retrying in 2s...")
                    time_mod.sleep(2)
        
        print(f"[ERROR] Failed to load MiDaS after {max_retries} attempts: {last_error}")
        raise last_error  # type: ignore[misc]
    
    def available(self) -> bool:
        """Check if depth estimation is available."""
        return self._ok and (self.model is not None)
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate hash for image caching."""
        if not self.config.cache_maps:
            return ""
        # Use image size + first 1000 bytes for quick hash
        img_bytes = image.tobytes()[:1000]
        return hashlib.md5(img_bytes + f"{image.size}".encode()).hexdigest()
    
    @torch.inference_mode()  # type: ignore[misc]
    def infer_map(self, image: Image.Image, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Compute normalized depth map in [0, 1] where 1.0 = closer to camera.
        
        Args:
            image: Input PIL Image
            use_cache: Use cached depth map if available
            
        Returns:
            Depth map as float32 numpy array, or None if unavailable
        """
        # Ensure model is loaded (lazy loading)
        self._ensure_model_loaded()
        
        if not self.available():
            return None
        
        # Check cache
        if use_cache and self.config.cache_maps:
            img_hash = self._get_image_hash(image)
            if img_hash in self._depth_cache:
                return self._depth_cache[img_hash].copy()
        
        # Compute depth based on model type
        if self._model_type == "depth_anything_v2":
            depth = self._infer_depth_anything_v2(image)
        else:
            depth = self._infer_midas(image)
        
        if depth is None:
            return None
        
        # Normalize to [0, 1] with 1.0 = closer
        depth = self._normalize_depth(depth)
        
        # Cache if enabled
        if use_cache and self.config.cache_maps:
            img_hash = self._get_image_hash(image)
            # LRU-style cache: remove oldest if full
            if len(self._depth_cache) >= self.config.max_cache_size:
                self._depth_cache.pop(next(iter(self._depth_cache)))
            self._depth_cache[img_hash] = depth.copy()
        
        return depth
    
    def _infer_depth_anything_v2(self, image: Image.Image) -> Optional[np.ndarray]:
        """Inference using Depth Anything V2."""
        try:
            # Ensure model is on the correct device
            try:
                param_device = next(self.model.parameters()).device
                if param_device.type != self.device and self.device != "cpu":
                    # If model is on CPU but we want GPU/MPS, move it
                    if param_device.type == "cpu":
                        print(f"[DEPTH] Moving model from {param_device} to {self.device}...")
                        self.model = self.model.to(self.device)
            except Exception:
                pass

            # Manual inference to ensure device consistency (avoids infer_image mismatch)
            # Preprocessing: Resize -> Normalize -> ToTensor
            w, h = image.size
            input_size = 518
            
            # Resize maintaining aspect ratio
            scale = input_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Ensure multiple of 14 (patch size)
            new_h = (new_h // 14) * 14
            new_w = (new_w // 14) * 14
            
            if new_h == 0 or new_w == 0:
                # Fallback for very small images
                new_h, new_w = 518, 518
                
            img_resized = image.resize((new_w, new_h), Image.BICUBIC)
            img_np = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Normalize (ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_np - mean) / std
            
            # To Tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                # Handle mixed precision
                device_type = "cuda" if self._amp_enabled else "cpu"
                # MPS doesn't support autocast fully yet, so we skip it for MPS or use default
                if self.device == "mps":
                    depth = self.model(img_tensor)
                else:
                    with torch.autocast(device_type=device_type, dtype=self._amp_dtype, enabled=self._amp_enabled):
                        depth = self.model(img_tensor)
                
            # Resize back to original resolution
            depth = F.interpolate(
                depth[:, None], 
                size=(h, w), 
                mode="bilinear", 
                align_corners=True
            )[0, 0]
            
            return depth.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"[ERROR] Depth Anything V2 inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _infer_midas(self, image: Image.Image) -> Optional[np.ndarray]:
        """Inference using MiDaS."""
        if not _HAS_CV2:
            img_np = np.array(image)
        else:
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        device_type = "cuda" if self._amp_enabled else "cpu"
        with torch.autocast(device_type=device_type, dtype=self._amp_dtype, enabled=self._amp_enabled):  # type: ignore[attr-defined]
            inp = self.transform(img_np).to(self.device)  # type: ignore[operator]
            pred = self.model(inp).squeeze().detach().cpu().numpy()  # type: ignore[operator]
        
        return pred.astype(np.float32)
    
    def _normalize_depth(self, depth: np.ndarray, invert: bool = True) -> np.ndarray:
        """
        Normalize depth to [0, 1] range.
        
        Args:
            depth: Raw depth map
            invert: If True, invert so 1.0 = closer (standard for this codebase)
        """
        depth = np.asarray(depth, dtype=np.float32)
        
        if not np.isfinite(depth).any():
            return np.full_like(depth, 0.5)
        
        # Robust percentile-based normalization (handles outliers)
        finite = depth[np.isfinite(depth)]
        pmin, pmax = np.percentile(finite, [2.0, 98.0])
        rng = max(1e-6, float(pmax - pmin))
        
        normalized = np.clip((depth - pmin) / rng, 0.0, 1.0)
        
        # Invert if needed (MiDaS: larger = farther, we want larger = closer)
        if invert:
            normalized = 1.0 - normalized
        
        return normalized
    
    def relative_depth_at(
        self, 
        image: Image.Image, 
        centers: Sequence[Tuple[float, float]],
        use_cache: bool = True
    ) -> List[float]:
        """
        Sample normalized depth values [0, 1] at given centers.
        
        Args:
            image: Input image
            centers: List of (x, y) coordinates
            use_cache: Use cached depth map if available
            
        Returns:
            List of depth values (1.0 = closer)
        """
        if not centers:
            return []
        
        dm = self.infer_map(image, use_cache=use_cache)
        if dm is None:
            return [0.5] * len(centers)
        
        H, W = dm.shape[:2]
        vals: List[float] = []
        
        for (cx, cy) in centers:
            x = int(np.clip(round(cx), 0, W - 1))
            y = int(np.clip(round(cy), 0, H - 1))
            vals.append(float(dm[y, x]))
        
        return vals
    
    def median_in_mask(
        self, 
        image: Image.Image, 
        mask: np.ndarray,
        use_cache: bool = True
    ) -> Optional[float]:
        """
        Compute median depth inside a binary mask.
        
        Args:
            image: Input image
            mask: Binary mask (bool or 0/1)
            use_cache: Use cached depth map if available
            
        Returns:
            Median depth value, or None if unavailable
        """
        dm = self.infer_map(image, use_cache=use_cache)
        if dm is None:
            return None
        
        m = mask.astype(bool)
        
        # Resize mask if needed
        if dm.shape != m.shape:
            if _HAS_CV2:
                m = cv2.resize(
                    m.astype(np.uint8), 
                    (dm.shape[1], dm.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                # Fallback: crop/pad
                H, W = dm.shape[:2]
                mh, mw = m.shape[:2]
                y0 = max(0, (mh - H) // 2)
                x0 = max(0, (mw - W) // 2)
                m = m[y0:y0 + H, x0:x0 + W]
                m = np.pad(
                    m, 
                    ((0, max(0, H - m.shape[0])), (0, max(0, W - m.shape[1]))), 
                    constant_values=False
                )
                m = m[:H, :W]
        
        vals = dm[m]
        vals = vals[np.isfinite(vals)]
        
        if vals.size == 0:
            return None
        
        return float(np.median(vals))
    
    def clear_cache(self) -> None:
        """Clear depth map cache."""
        self._depth_cache.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_maps": len(self._depth_cache),
            "max_size": self.config.max_cache_size,
        }


# Backward compatibility alias
DepthEstimator = DepthEstimatorV2
