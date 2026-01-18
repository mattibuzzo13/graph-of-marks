# igp/segmentation/samhq.py
# SAM-HQ wrapper con:
# - singolo embedding per immagine (set_image)
# - batching in chunk adattivi per box (predict_torch se disponibile)
# - autocast FP16 su CUDA
# - fallback robusti (punto al centro, sequenziale e shrinking del box)
# - postprocess maschere (chiudi fori, rimuovi componenti piccole)
# - cleanup memoria GPU tra chiamate

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig

# Checkpoint ufficiali SAM-HQ (usati se auto_download=True e non viene passato un ckpt locale).
_SAM_HQ_URLS = {
    "vit_b": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
    "vit_l": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
    "vit_h": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
}


class SamHQSegmenter(Segmenter):
    """
    SAM-HQ (SysCV). Richiede `segment_anything_hq`.
    Prompt principale: bounding box; fallback: punto positivo al centro del box.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: Optional[str] = None,
        *,
        config: Optional[SegmenterConfig] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Overwriting tiny_vit")
                from segment_anything_hq import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "segment_anything_hq non è installato. Installa da:\n"
                "pip install git+https://github.com/SysCV/sam-hq.git"
            ) from e

        ckpt_path = self._resolve_checkpoint(checkpoint, model_type, auto_download)
        self._SamPredictor = SamPredictor
        
        # Monkeypatch torch.load to handle CUDA checkpoints on CPU/MPS
        original_load = torch.load
        def safe_load(*args, **kwargs):
            if not torch.cuda.is_available() and "map_location" not in kwargs:
                kwargs["map_location"] = "cpu"
            return original_load(*args, **kwargs)
            
        try:
            torch.load = safe_load
            self._sam_model = sam_model_registry[model_type](checkpoint=str(ckpt_path)).to(self.device).eval()
        finally:
            torch.load = original_load
            
        self._predictor = SamPredictor(self._sam_model)

        # autocast preferito su CUDA
        self._amp_enabled = (self.device == "cuda")
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Segmenta ognuno dei box sull'immagine.
        Output per elemento:
          - 'segmentation': np.ndarray(bool, H, W)
          - 'bbox': [x, y, w, h] (xywh)
          - 'predicted_iou': float (score equivalente)
        """
        if not boxes:
            return []

        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        boxes_xyxy = [self.clamp_box_xyxy(b, W, H) for b in boxes]

        self._predictor.set_image(image_np)
        try:
            try:
                results = self._segment_batched(image_np, boxes_xyxy, H, W)
            except Exception as e:
                print(f"[SAM-HQ] Batch fallito, uso fallback sequenziale: {e}")
                results = self._segment_sequential(image_np, boxes_xyxy, H, W)

            final: List[Dict[str, Any]] = []
            for r in results:
                mask = self.postprocess_mask(r["segmentation"].astype(bool))
                final.append(
                    {
                        "segmentation": mask,
                        "bbox": self.bbox_from_mask(mask),
                        "predicted_iou": float(r.get("predicted_iou", 0.0)),
                    }
                )
            return final
        finally:
            self._release_predictor_memory()

    # ----------------- internals -----------------

    def _segment_batched(
        self,
        image_np: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Batching in chunk adattivi. Usa predict_torch se disponibile; altrimenti prevede per-box dentro il chunk.
        """
        results: List[Dict[str, Any]] = []
        chunk = self._adaptive_chunk_size(H, W, len(boxes_xyxy))
        has_predict_torch = hasattr(self._predictor, "predict_torch")
        transform = getattr(self._predictor, "transform", None)
        device_type = "cuda" if self._amp_enabled else "cpu"

        for start in range(0, len(boxes_xyxy), chunk):
            end = min(start + chunk, len(boxes_xyxy))
            current = boxes_xyxy[start:end]

            if has_predict_torch:
                try:
                    boxes_t = torch.as_tensor(current, dtype=torch.float32, device=self.device)
                    if transform is not None and hasattr(transform, "apply_boxes_torch"):
                        boxes_t = transform.apply_boxes_torch(boxes_t, image_np.shape[:2])

                    with torch.autocast(device_type=device_type, dtype=self._amp_dtype, enabled=self._amp_enabled), torch.inference_mode():
                        masks_t, scores_t, _ = self._predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=boxes_t,
                            multimask_output=True,
                        )

                    for i in range(masks_t.shape[0]):
                        m3 = masks_t[i]     # (3, H, W)
                        s3 = scores_t[i]    # (3,)
                        best = int(s3.argmax().item())
                        mask = m3[best].detach().to("cpu").numpy().astype(bool)
                        score = float(s3[best].item())

                        if mask.sum() < 50:
                            x1, y1, x2, y2 = current[i]
                            mask, score = self._fallback_point(mask, score, x1, y1, x2, y2)

                        results.append({"segmentation": mask, "predicted_iou": score})
                    continue
                except Exception as e:
                    print(f"[SAM-HQ] predict_torch non disponibile/errore: {e}; uso fallback per-box nel chunk.")

            # Fallback: per-box nel chunk (riduce overhead rispetto al tutto-sequenziale)
            for (x1, y1, x2, y2) in current:
                mask, score = self._predict_single_box(x1, y1, x2, y2, H, W)
                results.append({"segmentation": mask, "predicted_iou": score})

        return results

    def _predict_single_box(self, x1: int, y1: int, x2: int, y2: int, H: int, W: int) -> Tuple[np.ndarray, float]:
        """
        Predice una maschera per singolo box con shrinking progressivo e fallback al punto centrale.
        """
        device_type = "cuda" if self._amp_enabled else "cpu"
        mask_ok: Optional[np.ndarray] = None
        score_ok: float = 0.0

        for shrink in (0, 2, 4, 8, 12, 16):
            xs = max(0, x1 + shrink)
            ys = max(0, y1 + shrink)
            xe = max(xs + 1, x2 - shrink)
            ye = max(ys + 1, y2 - shrink)
            if xe <= xs or ye <= ys:
                continue

            box_arr = np.asarray([[xs, ys, xe, ye]], dtype=float)
            try:
                with torch.autocast(device_type=device_type, dtype=self._amp_dtype, enabled=self._amp_enabled), torch.inference_mode():
                    masks, scores, _ = self._predictor.predict(box=box_arr, multimask_output=True)
                best = int(scores.argmax()) if scores is not None else 0
                mask = masks[best].astype(bool)
                score = float(scores[best]) if scores is not None else 1.0

                if mask.sum() < 50:
                    mask, score = self._fallback_point(mask, score, xs, ys, xe, ye)

                mask_ok, score_ok = mask, score
                break
            except Exception:
                continue

        if mask_ok is None:
            mask_ok = np.zeros((H, W), dtype=bool)
            score_ok = 0.0

        return mask_ok, score_ok

    def _fallback_point(self, mask_cur: np.ndarray, score_cur: float, x1: int, y1: int, x2: int, y2: int) -> Tuple[np.ndarray, float]:
        """
        Fallback con singolo punto positivo al centro del box.
        """
        try:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            masks_pt, scores_pt, _ = self._predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            mask = masks_pt[0].astype(bool)
            score = float(scores_pt[0]) if scores_pt is not None else score_cur
            return mask, score
        except Exception:
            return mask_cur, score_cur

    def _segment_sequential(
        self,
        image_np: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback completamente sequenziale (solo se il batching fallisce).
        """
        out: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            mask, score = self._predict_single_box(x1, y1, x2, y2, H, W)
            out.append({"segmentation": mask, "predicted_iou": float(score)})
        return out

    def _adaptive_chunk_size(self, H: int, W: int, n_boxes: int) -> int:
        """
        Stima dimensione chunk sicura in base a VRAM e megapixel.
        """
        if not torch.cuda.is_available() or self.device != "cuda":
            return min(64, max(1, n_boxes))
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            gb = total_mem / (1024**3)
        except Exception:
            gb = 8.0

        if gb >= 40:
            base = 512
        elif gb >= 24:
            base = 384
        elif gb >= 16:
            base = 256
        elif gb >= 12:
            base = 192
        else:
            base = 128

        mp = (H * W) / 1_000_000.0
        if mp > 4:
            base //= 2
        if mp > 8:
            base //= 2

        return int(max(1, min(base, n_boxes)))

    def _release_predictor_memory(self) -> None:
        """
        Libera feature/embedding e cache CUDA tra chiamate.
        """
        try:
            self._predictor.reset_image()
        except Exception:
            pass
        for attr in ("features", "embedding", "image_embedding", "is_image_set"):
            if hasattr(self._predictor, attr):
                try:
                    delattr(self._predictor, attr)
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_checkpoint(self, checkpoint: Optional[str], model_type: str, auto_download: bool) -> Path:
        """
        Usa il checkpoint passato o risolve un default in ./checkpoints. Se mancante e auto_download=True, scarica dall'URL ufficiale.
        """
        if checkpoint:
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint SAM-HQ non trovato: {checkpoint}")
            return p

        fname = f"sam_hq_{model_type}.pth"
        p = Path("./checkpoints") / fname
        p.parent.mkdir(parents=True, exist_ok=True)

        if not p.exists():
            if not auto_download:
                raise FileNotFoundError(f"Checkpoint SAM-HQ assente: {p}")
            from torch.hub import download_url_to_file
            url = _SAM_HQ_URLS.get(model_type, _SAM_HQ_URLS["vit_h"])
            download_url_to_file(url, str(p), progress=True)
        return p