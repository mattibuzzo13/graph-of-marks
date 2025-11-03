# igp/detectors/owlvit.py
# Wrapper OWL-ViT (Owlv2) per open-vocabulary object detection.
# - Batch inference reale
# - FP16/autocast su GPU
# - TTA orizzontale opzionale (per detect singolo)
# - Creazione Detection robusta

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from PIL import Image

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from igp.detectors.base import Detector
from igp.types import Detection
from igp.utils.detector_utils import make_detection


# Queries di default (COCO-like + extra)
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

    - Usa Hugging Face Owlv2Processor / Owlv2ForObjectDetection.
    - Richiede una lista di query testuali (default COCO-like).
    - Ritorna List[Detection] con box (x1,y1,x2,y2), label (str), score (float).
    - Il filtraggio per score viene demandato alla logica della pipeline (base Detector).
    """

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
        super().__init__(name="owlvit", device=device, score_threshold=score_threshold)

        self.model_name = model_name
        self.queries: Sequence[str] = list(queries) if queries is not None else list(_DEFAULT_QUERIES)
        self.tta_hflip = bool(tta_hflip)

        # Processor + modello
        self.processor = Owlv2Processor.from_pretrained(model_name)

        # Dtype scelto in base al device e flag fp16
        dtype = torch.float16 if (fp16 and str(self.device).startswith("cuda") and torch.cuda.is_available()) else torch.float32
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(self.device)
        self.model.eval()

    def warmup(self, example_image=None, use_half: Optional[bool] = None) -> None:
        """Set fp16 preference and run a tiny forward pass to warmup memory (best-effort)."""
        # try to set dtype preference
        if use_half is not None:
            try:
                # only allow fp16 on CUDA
                if use_half and str(self.device).startswith("cuda"):
                    # no-op here; model was created with torch_dtype at init
                    pass
            except Exception:
                pass

        if example_image is None:
            return

        try:
            small = example_image.resize((min(512, example_image.width), min(512, example_image.height)))
            _ = self._detect_once(small)
        except Exception:
            # best-effort
            pass

    # ----------------- API -----------------

    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Inference su singola immagine.
        Il filtraggio per score viene applicato dalla pipeline a valle.
        """
        dets = self._detect_once(image)

        if self.tta_hflip:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            dets_flip = self._detect_once(flipped)
            W = image.size[0]
            remapped: List[Detection] = []
            for d in dets_flip:
                x1, y1, x2, y2 = d.box[:4]
                new_box = (W - x2, y1, W - x1, y2)
                remapped.append(self._rebox(d, new_box))
            dets.extend(remapped)

        return dets

    @property
    def supports_batch(self) -> bool:
        return True

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        if not images:
            return []

        # Prepara batch: ripeti le stesse query per ciascuna immagine
        batch_text = [self.queries] * len(images)

        encoding = self.processor(
            images=list(images),
            text=batch_text,
            return_tensors="pt",
        )
        # Move su device
        encoding = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}

        use_amp = (self.model.device.type == "cuda")
        # Use the new torch.amp.autocast API (device_type first) to avoid FutureWarning
        device_type = "cuda" if use_amp else "cpu"
        with torch.inference_mode(), torch.amp.autocast(device_type, enabled=use_amp):
            outputs = self.model(**encoding)

        # Post-process: target_sizes = (H, W) per immagine
        target_sizes = torch.tensor(
            [[img.height, img.width] for img in images],
            device=self.device,
        )
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
        )

        all_dets: List[List[Detection]] = []
        for res in results:
            boxes_t = res.get("boxes", torch.empty(0, 4)).detach().cpu()
            scores_t = res.get("scores", torch.empty(0)).detach().cpu()
            labels_t = res.get("labels", torch.empty(0, dtype=torch.long)).detach().cpu()

            dets: List[Detection] = []
            for box, score, lab_idx in zip(boxes_t.tolist(), scores_t.tolist(), labels_t.tolist()):
                label = self._safe_label(lab_idx)
                dets.append(self._make_detection(box, label, float(score)))
            all_dets.append(dets)

        return all_dets

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------- Interni -----------------

    @torch.inference_mode()
    def _detect_once(self, image: Image.Image) -> List[Detection]:
        encoding = self.processor(
            images=[image],
            text=self.queries,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}

        use_amp = (self.model.device.type == "cuda")
        device_type = "cuda" if use_amp else "cpu"
        with torch.amp.autocast(device_type, enabled=use_amp):
            outputs = self.model(**encoding)

        w, h = image.size
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
        )

        if not results:
            return []

        res = results[0]
        boxes_t = res.get("boxes", torch.empty(0, 4)).detach().cpu()
        scores_t = res.get("scores", torch.empty(0)).detach().cpu()
        labels_t = res.get("labels", torch.empty(0, dtype=torch.long)).detach().cpu()

        dets: List[Detection] = []
        for box, score, lab_idx in zip(boxes_t.tolist(), scores_t.tolist(), labels_t.tolist()):
            label = self._safe_label(lab_idx)
            dets.append(self._make_detection(box, label, float(score)))

        return dets

    def _safe_label(self, idx: int) -> str:
        try:
            return str(self.queries[idx])
        except Exception:
            return str(idx)

    def _make_detection(self, box_xyxy: Sequence[float], label: str, score: float) -> Detection:
        return make_detection(box_xyxy, label, score, source="owlvit")

    def _rebox(self, det: Detection, new_box_xyxy: Sequence[float]) -> Detection:
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