# igp/segmentation/sam2.py
# SAM 2.x wrapper con:
# - singolo embedding per immagine (set_image)
# - batching in chunk adattivi per box
# - autocast FP16 su CUDA
# - fallback robusti (punto al centro, sequenziale)
# - postprocess maschere (chiudi fori, rimuovi componenti piccole)

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


class Sam2Segmenter(Segmenter):
    """
    Segment Anything 2.x (Meta).
    Box-prompt primario; se la maschera è troppo piccola, fallback a un punto positivo al centro del box.
    """

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt",
        *,
        config: Optional[SegmenterConfig] = None,
        precision: Optional[str] = None,  # 'fp16' su CUDA, altrimenti 'fp32'
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._amp_enabled = (self.device == "cuda")
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32
        self._precision = precision or ("fp16" if self._amp_enabled else "fp32")

        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "Moduli SAM2 non trovati. Installa il repo ufficiale SAM2 (facebookresearch/sam2)."
            ) from e

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint SAM-2 non trovato: {ckpt_path}")
        # Il file YAML può essere risolto dal repo installato; non forziamo il download.
        self._sam2_model = build_sam2(model_cfg, str(ckpt_path), device=self.device, precision=self._precision).eval()
        self._predictor = SAM2ImagePredictor(self._sam2_model)

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Segmenta ogni box sull'immagine.
        Output per elemento:
          - 'segmentation': np.ndarray(bool, H, W)
          - 'bbox': [x, y, w, h] (xywh)
          - 'predicted_iou': float (o score equivalente)
        """
        if not boxes:
            return []

        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        boxes_xyxy = [self.clamp_box_xyxy(b, W, H) for b in boxes]

        self._predictor.set_image(image_np)

        try:
            results = self._segment_batched(image_np, boxes_xyxy, H, W)
        except Exception as e:
            print(f"[SAM2] Batch fallito, uso fallback sequenziale: {e}")
            results = self._segment_sequential(image_np, boxes_xyxy, H, W)

        # Postprocess per qualità e coerenza API
        final: List[Dict[str, Any]] = []
        for r in results:
            mask = r["segmentation"].astype(bool)
            # Postprocessing configurabile: solo se abilitato (ottimizzazione)
            if self.config.close_holes or self.config.remove_small_components:
                mask = self.postprocess_mask(mask)
            final.append(
                {
                    "segmentation": mask,
                    "bbox": self.bbox_from_mask(mask),
                    "predicted_iou": float(r.get("predicted_iou", 0.0)),
                }
            )
        # Libera cache/feature
        self._release_predictor_memory()
        return final

    # --------------------- internals ---------------------

    def _segment_batched(
        self,
        image_np: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Batching in chunk per evitare OOM. Tenta predict_torch se disponibile, altrimenti per-box nel chunk.
        """
        results: List[Dict[str, Any]] = []
        chunk = self._adaptive_chunk_size(H, W, len(boxes_xyxy))

        # Verifica API disponibili
        has_predict_torch = hasattr(self._predictor, "predict_torch")
        transform = getattr(self._predictor, "transform", None)
        device_type = "cuda" if self._amp_enabled else "cpu"

        for start in range(0, len(boxes_xyxy), chunk):
            end = min(start + chunk, len(boxes_xyxy))
            current = boxes_xyxy[start:end]

            if has_predict_torch:
                # Prova percorso torch nativo
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

                        # Fallback su punto centrale se maschera è troppo piccola
                        if mask.sum() < 50:
                            x1, y1, x2, y2 = current[i]
                            mask, score = self._fallback_point(mask, score, x1, y1, x2, y2)

                        results.append({"segmentation": mask, "predicted_iou": score})
                    continue
                except Exception as e:
                    print(f"[SAM2] predict_torch non disponibile/errore: {e}; uso fallback per-box.")

            # Fallback: per-box nel chunk (meno overhead rispetto a tutto-sequenziale)
            for (x1, y1, x2, y2) in current:
                mask, score = self._predict_single_box(x1, y1, x2, y2, H, W)
                results.append({"segmentation": mask, "predicted_iou": score})

        return results

    def _predict_single_box(self, x1: int, y1: int, x2: int, y2: int, H: int, W: int) -> Tuple[np.ndarray, float]:
        """
        Predice una maschera per singolo box. Include shrink progressivo e fallback punto centrale.
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
        Fallback con punto centrale positivo; se fallisce, ritorna l'input.
        """
        try:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            masks_pt, scores_pt, _ = self._predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1], dtype=int),
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
        Fallback totalmente sequenziale (usato solo se il batching fallisce all'inizio).
        """
        out: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            mask, score = self._predict_single_box(x1, y1, x2, y2, H, W)
            out.append({"segmentation": mask, "predicted_iou": float(score)})
        return out

    def _adaptive_chunk_size(self, H: int, W: int, n_boxes: int) -> int:
        """
        Stima una dimensione di chunk sicura in base a VRAM e megapixel dell'immagine.
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
        Libera feature/attivazioni tra chiamate per ridurre l'uso di VRAM.
        Smart cache clear: solo se memoria > 80% utilizzata.
        """
        # Alcune versioni hanno attributi interni con cache; facciamo best-effort.
        for attr in ("features", "embedding", "image_embedding", "is_image_set"):
            if hasattr(self._predictor, attr):
                try:
                    delattr(self._predictor, attr)
                except Exception:
                    pass
        # Smart cache clear
        if self._should_clear_cache():
            torch.cuda.empty_cache()
    
    def _should_clear_cache(self) -> bool:
        """Clear cache solo se memoria GPU utilizzata > 80%."""
        if not torch.cuda.is_available() or self.device != "cuda":
            return False
        try:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            if reserved == 0:
                return False
            ratio = allocated / reserved
            return ratio > 0.80  # Soglia 80%
        except Exception:
            return False  # Fallback sicuro