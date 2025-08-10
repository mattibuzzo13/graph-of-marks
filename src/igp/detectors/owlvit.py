# igp/detectors/owlvit.py
# Wrapper around OWL-ViT (Owlv2) for open-vocabulary object detection.
# Provides single-image inference, optional horizontal TTA, and robust
# creation of `Detection` objects without altering core logic.

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from PIL import Image
import numpy as np

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from igp.detectors.base import Detector
from igp.types import Detection


# Default open-vocabulary queries (COCO-like plus a few extras).
_DEFAULT_QUERIES: Sequence[str] = (
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
    "vase","scissors","teddy bear","hair drier","toothbrush","fence","grass","table","house","plate",
    "lamp","street lamp","sign","glass","plant","hedge","sofa","light","window","curtain","candle","tree",
    "sky","cloud","road","hat","glove","helmet","mountain","snow","sunglasses","bow tie","picture",
    "printer","monitor","pillow","stone","glasses","wheel","building","bridge","tomato","phone"
)


class OwlViTDetector(Detector):
    """
    OWL-ViT (Owlv2) open-vocabulary object detector.

    - Uses Hugging Face Owlv2Processor / Owlv2ForObjectDetection.
    - Requires a list of text queries (defaults to a COCO-like set).
    - Returns List[Detection] with (xyxy box as float, label str, score float).
    """

# NOTE: The class is redeclared below and will override this stub definition at import time.


class OwlViTDetector(Detector):
    def __init__(
        self,
        *,
        model_name: str = "google/owlv2-base-patch16",
        queries: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.4,
        tta_hflip: bool = False,
        fp16: bool = True,
        low_cpu_mem_usage: bool = True,
    ) -> None:
        # Initialize base class (sets name, device selection, and score threshold).
        super().__init__(
            name="owlvit",
            device=device,
            score_threshold=score_threshold
        )
        
        self.model_name = model_name
        self.queries: Sequence[str] = list(queries) if queries is not None else list(_DEFAULT_QUERIES)
        # `self.device` and `self.score_threshold` are already defined by the base class.
        self.tta_hflip = bool(tta_hflip)

        # Load processor/model; pick dtype based on device and fp16 flag.
        self.processor = Owlv2Processor.from_pretrained(model_name)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(self.device)
        self.model.eval()

    # ----------------- Base API -----------------

    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Single-image inference.
        Note: `_ensure_rgb` and score filtering are handled by `run()` in the base class.
        """
        dets = self._detect_once(image)

        # Optional TTA: horizontal flip + box re-projection.
        if self.tta_hflip:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            dets_flip = self._detect_once(flipped)
            W = image.size[0]
            for d in dets_flip:
                x1, y1, x2, y2 = d.box
                new_box = (W - x2, y1, W - x1, y2)
                d = self._rebox(d, new_box)
                dets.append(d)

        return dets  # Do not threshold here; `run()` applies the global score filter.

    # ----------------- Internal helpers -----------------

    @torch.inference_mode()
    def _detect_once(self, image: Image.Image) -> List[Detection]:
        # OWLv2 expects batched inputs; wrap single image.
        encoding = self.processor(
            images=[image],
            text=self.queries,           # list of query strings
            return_tensors="pt",
        )
        # Move tensors to the selected device.
        encoding = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}

        outputs = self.model(**encoding)

        # Post-process to image size.
        w, h = image.size
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes)

        if not results:
            return []

        res = results[0]
        boxes_t = res.get("boxes", torch.empty(0, 4)).detach().cpu()
        scores_t = res.get("scores", torch.empty(0)).detach().cpu()
        labels_t = res.get("labels", torch.empty(0, dtype=torch.long)).detach().cpu()

        boxes = boxes_t.tolist()
        scores = scores_t.tolist()
        labels = labels_t.tolist()
        
        detections: List[Detection] = []
        for box, score, lab_idx in zip(boxes, scores, labels):
            # Thresholding is deferred; the base `run()` method will handle it.
            label = self._safe_label(lab_idx)
            detections.append(self._make_detection(box, label, float(score)))

        return detections

    def _safe_label(self, idx: int) -> str:
        # Resolve label from query list; fall back to the index as string.
        try:
            return str(self.queries[idx])
        except Exception:
            return str(idx)

    def _make_detection(self, box_xyxy: Sequence[float], label: str, score: float) -> Detection:
        # Create Detection robustly against variations of the dataclass signature.
        b = tuple(float(x) for x in box_xyxy[:4])
        try:
            return Detection(box=b, label=label, score=score, source="owlvit")
        except TypeError:
            try:
                return Detection(box=b, label=label, score=score)
            except TypeError:
                # Minimal compatibility fallback.
                return Detection(box=b, label=label)

    def _rebox(self, det: Detection, new_box_xyxy: Sequence[float]) -> Detection:
        # Return a copy of `det` with a new box; handles immutable dataclasses.
        b = tuple(float(x) for x in new_box_xyxy[:4])
        try:
            return Detection(
                box=b,
                label=getattr(det, "label", ""),
                score=getattr(det, "score", 1.0),
                source=getattr(det, "source", "owlvit"),
            )
        except TypeError:
            try:
                return Detection(
                    box=b,
                    label=getattr(det, "label", ""),
                    score=getattr(det, "score", 1.0),
                )
            except TypeError:
                return Detection(box=b, label=getattr(det, "label", ""))


__all__ = ["OwlViTDetector"]
