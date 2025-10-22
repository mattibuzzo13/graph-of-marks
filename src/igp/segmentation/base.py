from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np


@dataclass
class SegmenterConfig:
    device: Optional[str] = None
    close_holes: bool = False
    hole_kernel: int = 7
    min_hole_area: int = 100
    remove_small_components: bool = False
    min_component_area: int = 0  # rimuovi componenti < min_component_area px se >0


class Segmenter(ABC):
    """
    Base interface for SAM-like segmenters.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()

    # --------- Required API ---------
    @abstractmethod
    def segment(self, image_pil, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Compute a mask for each input box.

        Args:
            image_pil: input PIL image.
            boxes: list of [x1, y1, x2, y2] boxes in pixel coordinates.

        Returns:
            A list of dicts with keys:
              - 'segmentation': np.ndarray(bool, H, W)
              - 'bbox': [x, y, w, h] (xywh)
              - 'predicted_iou': float
        """

    # --------- Common utilities ---------
    @staticmethod
    def clamp_box_xyxy(box: Sequence[float], W: int, H: int) -> List[int]:
        """
        Clamp xyxy to image bounds, garantendo x2>x1 e y2>y1 anche per immagini molto piccole.
        """
        W = max(1, int(W))
        H = max(1, int(H))
        x1, y1, x2, y2 = box[:4]
        x1 = int(np.clip(round(x1), 0, W - 1))
        y1 = int(np.clip(round(y1), 0, H - 1))
        x2 = int(np.clip(round(x2), 0, W - 1))
        y2 = int(np.clip(round(y2), 0, H - 1))
        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)
        return [x1, y1, x2, y2]

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> List[int]:
        """
        Convert a boolean mask into an xywh bbox (in pixels). If empty, return [0, 0, 0, 0].
        """
        m = np.asarray(mask)
        if m.ndim > 2:
            m = m.squeeze()
        ys, xs = np.where(m)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    def close_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Chiude fori interni e piccoli gap:
          1) morphological closing
          2) riempie solo i fori < min_hole_area (con OpenCV)
        Fallback:
          - con SciPy: binary_closing + fill_holes con filtro per area
          - senza OpenCV/SciPy: ritorna la maschera (o solo closing NumPy leggero)
        """
        if not self.config.close_holes:
            return mask.astype(bool)

        m_bool = mask.astype(bool)

        # OpenCV path (più veloce/robusto)
        try:
            import cv2  # type: ignore

            k = max(1, int(self.config.hole_kernel))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            m_u8 = (m_bool.astype(np.uint8) * 255)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)

            # Trova i fori: floodfill sul background inverso
            inv = 255 - m_u8
            h, w = inv.shape[:2]
            flood = inv.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ff_mask, (0, 0), 0)
            holes = cv2.bitwise_and(inv, flood)

            # Rimuove solo fori piccoli via connectedComponentsWithStats
            num, labels, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), connectivity=4)
            out = m_u8.copy()
            min_area = int(self.config.min_hole_area)
            for lab_id in range(1, num):
                area = int(stats[lab_id, cv2.CC_STAT_AREA])
                if area < min_area:
                    out[labels == lab_id] = 255

            return (out > 0).astype(bool)

        except Exception:
            pass  # fallback SciPy/NumPy

        # SciPy fallback: closing + fill_holes + filtro per area
        try:
            from scipy.ndimage import binary_closing, binary_fill_holes, label  # type: ignore

            k = max(1, int(self.config.hole_kernel))
            structure = np.ones((k, k), dtype=bool)
            closed = binary_closing(m_bool, structure=structure)

            # Trova fori (componenti del background dentro all'oggetto)
            filled = binary_fill_holes(closed)
            holes = np.logical_and(filled, np.logical_not(closed))

            if self.config.min_hole_area > 0:
                lab, num = label(holes)
                out = closed.copy()
                min_area = int(self.config.min_hole_area)
                for lab_id in range(1, num + 1):
                    area = int((lab == lab_id).sum())
                    if area < min_area:
                        out[lab == lab_id] = True
                return out.astype(bool)
            else:
                return filled.astype(bool)

        except Exception:
            # Ultimo fallback: piccolo closing NumPy (dilatazione/erosione naive)
            return self._binary_closing_numpy(m_bool, radius=max(1, int(self.config.hole_kernel // 2)))

    @staticmethod
    def _binary_closing_numpy(mask: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Closing leggero senza dipendenze: ripete dilatazione ed erosione 3x3 per ~radius volte.
        Utile come fallback se OpenCV/SciPy non sono disponibili.
        """
        if radius <= 0:
            return mask.astype(bool)
        m = mask.astype(bool)
        for _ in range(radius):
            m = Segmenter._dilate_3x3_bool(m)
        for _ in range(radius):
            m = Segmenter._erode_3x3_bool(m)
        return m

    @staticmethod
    def _dilate_3x3_bool(m: np.ndarray) -> np.ndarray:
        s = [
            np.pad(m[1:, :], ((0, 1), (0, 0)), constant_values=False),
            np.pad(m[:-1, :], ((1, 0), (0, 0)), constant_values=False),
            np.pad(m[:, 1:], ((0, 0), (0, 1)), constant_values=False),
            np.pad(m[:, :-1], ((0, 0), (1, 0)), constant_values=False),
            np.pad(m[1:, 1:], ((0, 1), (0, 1)), constant_values=False),
            np.pad(m[1:, :-1], ((0, 1), (1, 0)), constant_values=False),
            np.pad(m[:-1, 1:], ((1, 0), (0, 1)), constant_values=False),
            np.pad(m[:-1, :-1], ((1, 0), (1, 0)), constant_values=False),
        ]
        return m | s[0] | s[1] | s[2] | s[3] | s[4] | s[5] | s[6] | s[7]

    @staticmethod
    def _erode_3x3_bool(m: np.ndarray) -> np.ndarray:
        s = [
            np.pad(m[1:, :], ((0, 1), (0, 0)), constant_values=True),
            np.pad(m[:-1, :], ((1, 0), (0, 0)), constant_values=True),
            np.pad(m[:, 1:], ((0, 0), (0, 1)), constant_values=True),
            np.pad(m[:, :-1], ((0, 0), (1, 0)), constant_values=True),
            np.pad(m[1:, 1:], ((0, 1), (0, 1)), constant_values=True),
            np.pad(m[1:, :-1], ((0, 1), (1, 0)), constant_values=True),
            np.pad(m[:-1, 1:], ((1, 0), (0, 1)), constant_values=True),
            np.pad(m[:-1, :-1], ((1, 0), (1, 0)), constant_values=True),
        ]
        return m & s[0] & s[1] & s[2] & s[3] & s[4] & s[5] & s[6] & s[7]

    def remove_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Rimuove componenti con area < min_area. Usa OpenCV se presente, altrimenti SciPy; fallback NumPy.
        """
        if min_area <= 0:
            return mask.astype(bool)

        m = mask.astype(bool)

        # OpenCV
        try:
            import cv2  # type: ignore
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
            keep = np.zeros_like(m, dtype=bool)
            for lab_id in range(1, num):
                area = int(stats[lab_id, cv2.CC_STAT_AREA])
                if area >= min_area:
                    keep |= (labels == lab_id)
            return keep
        except Exception:
            pass

        # SciPy
        try:
            from scipy.ndimage import label  # type: ignore
            lab, num = label(m)
            keep = np.zeros_like(m, dtype=bool)
            for lab_id in range(1, num + 1):
                area = int((lab == lab_id).sum())
                if area >= min_area:
                    keep |= (lab == lab_id)
            return keep
        except Exception:
            # Fallback: nessuna rimozione possibile
            return m

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Applica chiusura fori e rimozione componenti piccole secondo config.
        """
        m = mask.astype(bool)
        m = self.close_mask_holes(m)
        if self.config.remove_small_components and self.config.min_component_area > 0:
            m = self.remove_small_components(m, self.config.min_component_area)
        return m