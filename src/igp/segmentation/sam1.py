# igp/segmentation/sam1.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


# Official SAM v1 checkpoint URLs (Meta)
_SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


class Sam1Segmenter(Segmenter):
    """
    Segment Anything v1 (Meta).
    - Singolo embedding immagine (predictor.set_image) per tutte le box
    - Batch dei prompt box via predict_torch (multimask_output=True)
    - Autocast FP16 su CUDA, fallback a float32
    - Fallback compatibile a percorso sequenziale (predict) per versioni vecchie
    - Postprocess: chiusura fori e rimozione componenti piccole (via SegmenterConfig)
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        *,
        points_per_side: int = 32,                  # mantenuto per compatibilità API
        pred_iou_thresh: float = 0.8,               # non usato direttamente
        stability_score_thresh: float = 0.85,       # non usato direttamente
        min_mask_region_area: int = 300,            # non usato direttamente
        config: Optional[SegmenterConfig] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        try:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "segment_anything non è installato. Installa da:\n"
                "https://github.com/facebookresearch/segment-anything"
            ) from e

        ckpt_path = self._resolve_checkpoint(checkpoint, model_type, auto_download)
        self._SamPredictor = SamPredictor
        self._sam_model = sam_model_registry[model_type](checkpoint=str(ckpt_path)).to(self.device).eval()
        self._predictor = SamPredictor(self._sam_model)

        # dtype preferita su CUDA
        self._amp_enabled = (self.device == "cuda")
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32

    # ----------------- public API -----------------

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Segmenta gli oggetti con SAM 1.0, usando BATCH dei box per massima velocità.
        Ritorna liste di dict con:
          - 'segmentation': np.ndarray(bool, H, W)
          - 'bbox': [x, y, w, h] (xywh)
          - 'predicted_iou': float
        """
        if not boxes:
            return []

        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        # Pre-clamp per sicurezza
        boxes_clamped = [self.clamp_box_xyxy(b, W, H) for b in boxes]

        # Prepara predictor e embedding
        self._predictor.set_image(image_np)

        try:
            # Percorso preferito: batching su torch
            results = self._segment_boxes_batched(image_np, boxes_clamped, H, W)
        except Exception as e:
            print(f"[SAM1] Batch fallito, fallback a sequenziale: {e}")
            results = self._segment_boxes_sequential(image_np, boxes_clamped, H, W)

        # Post-process maschere e bounding box
        final: List[Dict[str, Any]] = []
        for res in results:
            mask = res["segmentation"].astype(bool)
            # Postprocessing configurabile: solo se abilitato (ottimizzazione)
            if self.config.close_holes or self.config.remove_small_components:
                mask = self.postprocess_mask(mask)
            bbox_xywh = self.bbox_from_mask(mask)
            final.append({
                "segmentation": mask,
                "bbox": bbox_xywh,
                "predicted_iou": float(res.get("predicted_iou", 0.0)),
            })

        # Libera embedding
        self._predictor.reset_image()
        if hasattr(self._predictor, "features"):
            try:
                delattr(self._predictor, "features")
            except Exception:
                pass
        # Smart cache clear: solo se memoria > 80% utilizzata
        if self._should_clear_cache():
            torch.cuda.empty_cache()

        return final
    
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

    # ----------------- internals -----------------

    def _segment_boxes_batched(
        self,
        image_np: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Batch inference con predict_torch e chunking adattivo per evitare OOM.
        """
        device = self.device
        results: List[Dict[str, Any]] = []

        # Trasforma box in coordinate del modello SAM
        boxes_tensor = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=device)
        boxes_trans = self._predictor.transform.apply_boxes_torch(boxes_tensor, image_np.shape[:2])

        # Chunking adattivo
        chunk = self._adaptive_chunk_size(H, W, len(boxes_xyxy))
        for start in range(0, len(boxes_xyxy), chunk):
            end = min(start + chunk, len(boxes_xyxy))
            bx = boxes_trans[start:end]

            with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._amp_enabled), torch.inference_mode():
                # predict_torch ritorna (B, 3, H, W) e (B, 3)
                masks_t, ious_t, _ = self._predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bx,
                    multimask_output=True,
                )

            # Per ciascun box scegli la mask con IoU predetto maggiore
            for i in range(masks_t.shape[0]):
                m3 = masks_t[i]          # (3, H, W)
                s3 = ious_t[i]           # (3,)
                best_idx = int(s3.argmax().item())
                best_mask = m3[best_idx].detach().to("cpu").numpy().astype(bool)
                best_score = float(s3[best_idx].item())

                # Fallback: se maschera quasi vuota, prova punto centrale
                if best_mask.sum() < 50:
                    x1, y1, x2, y2 = boxes_xyxy[start + i]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    try:
                        masks_pt, scores_pt, _ = self._predictor.predict(
                            point_coords=np.array([[cx, cy]]),
                            point_labels=np.array([1]),
                            multimask_output=False,
                        )
                        best_mask = masks_pt[0].astype(bool)
                        best_score = float(scores_pt[0])
                    except Exception:
                        pass

                results.append({
                    "segmentation": best_mask,
                    "predicted_iou": best_score,
                })

        return results

    def _segment_boxes_sequential(
        self,
        image_np: np.ndarray,
        boxes_xyxy: Sequence[Sequence[float]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback compatibile: loop per box con predictor.predict.
        """
        out: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            mask_ok = None
            score_ok = 0.0

            # Prova box shrinking progressivo per robustezza
            for shrink in (0, 2, 4, 8, 12, 16):
                xs = max(0, x1 + shrink)
                ys = max(0, y1 + shrink)
                xe = max(xs + 1, x2 - shrink)
                ye = max(ys + 1, y2 - shrink)
                if xe <= xs or ye <= ys:
                    continue

                box_arr = np.asarray([[xs, ys, xe, ye]], dtype=float)
                masks_box, scores_box, _ = self._predictor.predict(
                    box=box_arr, multimask_output=True
                )
                best = int(scores_box.argmax())
                mask = masks_box[best].astype(bool)
                score = float(scores_box[best])

                # Fallback su punto centrale se maschera è troppo piccola
                if mask.sum() < 50:
                    cx, cy = (xs + xe) // 2, (ys + ye) // 2
                    try:
                        masks_pt, scores_pt, _ = self._predictor.predict(
                            point_coords=np.array([[cx, cy]]),
                            point_labels=np.array([1]),
                            multimask_output=False,
                        )
                        mask = masks_pt[0].astype(bool)
                        score = float(scores_pt[0])
                    except Exception:
                        pass

                mask_ok, score_ok = mask, score
                break

            if mask_ok is None:
                mask_ok = np.zeros((H, W), dtype=bool)
                score_ok = 0.0

            out.append({
                "segmentation": mask_ok,
                "predicted_iou": float(score_ok),
            })

        return out

    def _adaptive_chunk_size(self, H: int, W: int, n_boxes: int) -> int:
        """
        Stima una dimensione di chunk sicura in base alla VRAM e ai megapixel.
        """
        if not torch.cuda.is_available() or self.device != "cuda":
            return min(64, max(1, n_boxes))

        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            gb = total_mem / (1024**3)
        except Exception:
            gb = 8.0

        # base per VRAM
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

        # riduci per immagini molto grandi
        mp = (H * W) / 1_000_000.0
        if mp > 4:
            base //= 2
        if mp > 8:
            base //= 2

        return int(max(1, min(base, n_boxes)))

    def _resolve_checkpoint(self, checkpoint: Optional[str], model_type: str, auto_download: bool) -> Path:
        if checkpoint:
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"SAM-1 checkpoint non trovato: {checkpoint}")
            return p

        # Default filename per model type
        fname = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }.get(model_type, "sam_vit_h_4b8939.pth")
        p = Path("./checkpoints") / fname
        p.parent.mkdir(parents=True, exist_ok=True)

        if not p.exists():
            if not auto_download:
                raise FileNotFoundError(f"Checkpoint SAM-1 mancante: {p}")
            # Download dall’URL ufficiale
            url = _SAM_URLS.get(model_type, _SAM_URLS["vit_h"])
            from torch.hub import download_url_to_file
            download_url_to_file(url, str(p))
        return p