"""Coordinator per esecuzione e fusione di più detector.

Obiettivi principali:
- evitare preprocess ripetuti (convertire in RGB una volta)
- eseguire i detector con batching quando possibile
- eseguire i detector su CPU in parallelo e su GPU in sequenza per minimizzare contesa
- caching LRU per evitare ricalcoli su immagini identiche
- fusione dei risultati tramite `igp.fusion.wbf.fuse_detections_wbf`

Questo è un componente a basso rischio che usa le API esistenti dei Detector
definite in `igp.detectors.base.Detector` e la funzione WBF già presente nel
progetto.
"""
from __future__ import annotations

from typing import List, Sequence, Optional, Dict, Tuple
from collections import OrderedDict, defaultdict
import hashlib
import copy
import logging
import concurrent.futures

from PIL import Image

from igp.detectors.base import Detector
from igp.types import Detection
from igp.fusion.wbf import fuse_detections_wbf

# Advanced optimizations (optional, fallback to standard WBF if not available)
try:
    from igp.fusion.wbf_optimized import fuse_detections_wbf_spatial
    _HAVE_SPATIAL_WBF = True
except ImportError:
    _HAVE_SPATIAL_WBF = False

try:
    from igp.fusion.cascade import DetectorCascade
    _HAVE_CASCADE = True
except ImportError:
    _HAVE_CASCADE = False

logger = logging.getLogger(__name__)


class DetectorManager:
    """Gestisce e coordina più detectors.

    Usage (esempio):
        mgr = DetectorManager([det1, det2], cache_size=128)
        fused = mgr.detect_ensemble(images)

    Principali parametri:
        detectors: lista di oggetti che ereditano `Detector`.
        cache_size: dimensione LRU della cache delle immagini (default 128).
        weights_by_source: pass-through a WBF per pesare sorgenti diverse.
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        *,
        cache_size: int = 128,
        weights_by_source: Optional[Dict[str, object]] = None,
        hash_method: str = "full",
        enable_cross_class_suppression: bool = True,
        cross_class_iou_thr: Optional[float] = None,
        enable_mask_iou_suppression: bool = True,
        mask_iou_thr: Optional[float] = None,
        # Advanced optimizations
        use_spatial_fusion: bool = True,
        spatial_cell_size: int = 100,
        use_hierarchical_fusion: bool = True,
        use_cascade: bool = False,
        cascade_conf_threshold: float = 0.40,
    ) -> None:
        self.detectors = list(detectors)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, List[Detection]]" = OrderedDict()
        self.weights_by_source = weights_by_source or {}
        self.hash_method = str(hash_method).lower()
        # Cross-class suppression: when True, after fusion we greedily remove
        # lower-scored detections that highly overlap (IoU >= cross_class_iou_thr)
        # with a higher-scored detection. If cross_class_iou_thr is None we
        # use a conservative default computed per-call.
        self.enable_cross_class_suppression = bool(enable_cross_class_suppression)
        self.cross_class_iou_thr = float(cross_class_iou_thr) if cross_class_iou_thr is not None else None
        # Mask-based suppression: when enabled, after fusion we compare fused
        # segmentation masks and remove lower-scored detections that have high
        # mask IoU with a kept detection. This helps eliminate label duplicates
        # when masks are available.
        self.enable_mask_iou_suppression = bool(enable_mask_iou_suppression)
        self.mask_iou_thr = float(mask_iou_thr) if mask_iou_thr is not None else None
        # Group/merge overlapping detections into a single object when they
        # represent the same physical object. This will UNION masks (if
        # available) and compute a union bbox; label and score come from the
        # highest-scored detection in the group.
        self.enable_group_merge = True
        self.merge_mask_iou_thr = 0.6
        self.merge_box_iou_thr = 0.9
        
        # Keep low-score detections if no competing objects in region
        self.keep_non_competing_low_scores = True
        self.non_competing_iou_threshold = 0.30
        self.non_competing_min_score = 0.05
        
        # Advanced optimizations
        self.use_spatial_fusion = bool(use_spatial_fusion) and _HAVE_SPATIAL_WBF
        self.spatial_cell_size = int(spatial_cell_size)
        self.use_hierarchical_fusion = bool(use_hierarchical_fusion)
        self.use_cascade = bool(use_cascade) and _HAVE_CASCADE
        self.cascade_conf_threshold = float(cascade_conf_threshold)
        
        # Initialize cascade if enabled
        self._cascade = None
        if self.use_cascade and _HAVE_CASCADE:
            self._init_cascade()
        
        # Performance tracking
        self.last_run_stats = {}
        
        if self.use_spatial_fusion:
            logger.info("DetectorManager: Using spatial hash optimization (5-10x faster WBF)")
        if self.use_hierarchical_fusion:
            logger.info("DetectorManager: Using hierarchical fusion (2-3x faster)")
        if self.use_cascade:
            logger.info("DetectorManager: Using detector cascade (60-70% compute reduction)")
    
    def _init_cascade(self) -> None:
        """Initialize detector cascade with fast detector + heavy detectors."""
        if not _HAVE_CASCADE:
            return
        
        # Identify fast detector (YOLO) and heavy detectors (OWL-ViT, Detectron2)
        fast_detector = None
        heavy_detectors = []
        
        for det in self.detectors:
            det_name = det.__class__.__name__.lower()
            if "yolo" in det_name:
                fast_detector = det
            elif "owl" in det_name or "detectron" in det_name:
                heavy_detectors.append(det)
            else:
                # Unknown detector - add to heavy by default
                heavy_detectors.append(det)
        
        if fast_detector and heavy_detectors:
            self._cascade = DetectorCascade(
                fast_detector,
                heavy_detectors,
                cascade_conf_threshold=self.cascade_conf_threshold,
                roi_expansion=1.2,
                min_roi_size=100,
                max_roi_size=800,
            )
            logger.info(f"Cascade initialized: {fast_detector.__class__.__name__} → "
                       f"{[d.__class__.__name__ for d in heavy_detectors]}")
        else:
            logger.warning("Cascade requested but no suitable fast/heavy detector found")
            self.use_cascade = False

    # ----------------- cache helpers -----------------

    @staticmethod
    def _hash_image(img: Image.Image) -> str:
        """Hash deterministico di un'immagine PIL (mode+size+bytes).

        Non è criptografico critico, serve per deduplicare immagini identiche
        in memoria. Usa SHA1 su header (mode/size) + raw bytes.
        """
        h = hashlib.sha1()
        try:
            header = f"{img.mode}:{img.size}:{getattr(img, 'info', {})}".encode("utf-8")
            h.update(header)
            # Two hashing modes: full (original behaviour) or thumb (cheap)
            from io import BytesIO

            if getattr(DetectorManager, "hash_method", None) is not None:
                # instance attribute will be preferred; this staticmethod is called
                # via the instance wrapper which passes through. Keep backward
                # compatibility by checking instance attribute in callers.
                pass

            # We'll try a best-effort approach: if caller wanted a thumbnail
            # hash, create a small thumbnail and hash its bytes; otherwise fall
            # back to PNG bytes of the full image.
            # Note: callers should set instance.hash_method accordingly.
            # Default behaviour: full.
            # This function may be called without access to the instance; in
            # that rare case we use full mode.
            mode = getattr(img, "_hash_mode", None) or "full"
            if mode == "thumb":
                b = BytesIO()
                # create deterministic thumbnail (max side 160)
                thumb = img.copy()
                thumb.thumbnail((160, 160), resample=Image.LANCZOS)
                thumb.save(b, format="PNG")
                h.update(b.getvalue())
                return h.hexdigest()

            # full (original) mode
            b = BytesIO()
            img.save(b, format="PNG")
            h.update(b.getvalue())
            return h.hexdigest()
        except Exception:
            # fallback: use raw bytes if PNG encoding fails
            try:
                data = img.tobytes()
                h.update(data)
            except Exception:
                # last resort: use id
                h.update(str(id(img)).encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[List[Detection]]:
        val = self._cache.get(key)
        if val is None:
            return None
        # LRU: move to end
        self._cache.move_to_end(key)
        # return a deepcopy to avoid accidental mutation of cached objects
        return copy.deepcopy(val)

    def _cache_set(self, key: str, value: List[Detection]) -> None:
        self._cache[key] = copy.deepcopy(value)
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # ----------------- core API -----------------

    def detect_ensemble(
        self,
        images: Sequence[Image.Image],
        *,
        fuse: bool = True,
        iou_thr: float = 0.40,  # 🔧 Più aggressivo (era 0.55) per ridurre overlap
        skip_box_thr: float = 0.0,
        return_stats: bool = False,
    ) -> List[List[Detection]]:
        """Esegui tutti i detector sulle immagini e ritorna le detections per immagine.

        - Se `fuse` è True, applica WBF/NMS centralizzato per ogni immagine.
        - Usa la cache per immagini già elaborate.
        - Esegue i detector su CPU in parallelo e su GPU in sequenza.
        """
        if not images:
            return []

        import time
        t_start = time.monotonic()

        # Calcola hash immagini e trova quali sono in cache
        # Mark images with desired hash mode to allow _hash_image to pick a
        # cheaper thumbnail-based hash when requested.
        imgs_for_hash = []
        for img in images:
            im = self._ensure_rgb(img)
            try:
                # attach desired mode to the Image instance for the hasher
                setattr(im, "_hash_mode", self.hash_method)
            except Exception:
                pass
            imgs_for_hash.append(im)
        keys = [self._hash_image(im) for im in imgs_for_hash]
        results: List[Optional[List[Detection]]] = [None] * len(images)

        # Fill from cache
        to_compute_idx: List[int] = []
        for i, k in enumerate(keys):
            cached = self._cache_get(k)
            if cached is not None:
                results[i] = cached
            else:
                to_compute_idx.append(i)

        if not to_compute_idx:
            # All cached
            return results  # type: ignore[return-value]

        # Build list of images to compute
        compute_images = [self._ensure_rgb(images[i]) for i in to_compute_idx]

        # Group detectors by device (normalize string)
        devmap = defaultdict(list)
        for det in self.detectors:
            dev = (det.device or "cpu").lower()
            # normalize 'cuda:0' -> 'cuda'
            if dev.startswith("cuda"):
                dev = "cuda"
            devmap[dev].append(det)

        # reset last run stats
        self.last_run_stats = {}

    # For each detector, run detect_batch (real batching) when possible
        # Strategy: cpu detectors in parallel; cuda detectors sequentially
        all_detector_outputs: Dict[Detector, List[List[Detection]]] = {}

        # CPU detectors: parallel across detectors
        cpu_detectors = devmap.get("cpu", [])
        if cpu_detectors:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(cpu_detectors))) as ex:
                futs = {ex.submit(self._run_detector_batch, det, compute_images): det for det in cpu_detectors}
                for fut in concurrent.futures.as_completed(futs):
                    det = futs[fut]
                    try:
                        res = fut.result()
                        all_detector_outputs[det] = res
                        # record simple stats (duration and num_images) if set by worker
                        stats = getattr(det, "_last_run_stats", None)
                        if stats is not None:
                            self.last_run_stats[det.__class__.__name__] = stats
                    except Exception:
                        logger.exception("Detector %s failed on CPU; returning empty results", det)
                        all_detector_outputs[det] = [[] for _ in compute_images]

        # CUDA detectors: run one-by-one to avoid multiple models competing for GPU
        for det in devmap.get("cuda", []):
            try:
                out = self._run_detector_batch(det, compute_images)
                all_detector_outputs[det] = out
                stats = getattr(det, "_last_run_stats", None)
                if stats is not None:
                    self.last_run_stats[det.__class__.__name__] = stats
            except Exception:
                logger.exception("CUDA detector %s failed; skipping", det)
                all_detector_outputs[det] = [[] for _ in compute_images]

        # Other devices (e.g., 'cpu0' or custom) - run sequentially
        for dev, detectors in devmap.items():
            if dev in ("cpu", "cuda"):
                continue
            for det in detectors:
                try:
                    out = self._run_detector_batch(det, compute_images)
                    all_detector_outputs[det] = out
                    stats = getattr(det, "_last_run_stats", None)
                    if stats is not None:
                        self.last_run_stats[det.__class__.__name__] = stats
                except Exception:
                    logger.exception("Detector %s on device %s failed; skipping", det, dev)
                    all_detector_outputs[det] = [[] for _ in compute_images]

        # For each image to compute, gather detections from all detectors and fuse
        for out_idx, orig_idx in enumerate(to_compute_idx):
            per_image_dets: List[Detection] = []
            for det, det_outputs in all_detector_outputs.items():
                if out_idx < len(det_outputs):
                    # ensure source attribute is present
                    det_list = det_outputs[out_idx] or []
                    per_image_dets.extend(det_list)

            if fuse:
                W, H = images[orig_idx].size
                # 🚀 Save all original detections before fusion for recovery later
                all_original_detections = list(per_image_dets) if self.keep_non_competing_low_scores else []
                
                try:
                    # Use optimized spatial WBF if available and enabled
                    if self.use_spatial_fusion and _HAVE_SPATIAL_WBF:
                        fused = fuse_detections_wbf_spatial(
                            per_image_dets, 
                            (W, H), 
                            weights_by_source=self.weights_by_source, 
                            iou_thr=iou_thr, 
                            skip_box_thr=skip_box_thr,
                            cell_size=self.spatial_cell_size,
                            hierarchical=self.use_hierarchical_fusion,
                        )
                    else:
                        # Standard WBF
                        fused = fuse_detections_wbf(
                            per_image_dets, 
                            (W, H), 
                            weights_by_source=self.weights_by_source, 
                            iou_thr=iou_thr, 
                            skip_box_thr=skip_box_thr
                        )
                except Exception:
                    logger.exception("WBF failed; falling back to raw detections")
                    fused = per_image_dets

                # Post-fusion: enforce per-class NMS, then group/merge highly
                # overlapping detections into single objects when enabled.
                try:
                    # Prefer using the project's NMS utility for stable behaviour
                    from igp.fusion.nms import nms as _nms_list
                    # Run class-aware NMS first to remove intra-class duplicates
                    try:
                        fused = _nms_list(fused, class_aware=True, iou_thr=iou_thr, sort_desc=True)
                    except Exception:
                        # nms may accept different signatures; fall back silently
                        pass
                except Exception:
                    # If import fails, continue without labelwise NMS
                    logger.debug("DetectorManager: labelwise NMS not available; skipping")

                try:
                    # First: optional grouping/merge of overlapping detections.
                    if getattr(self, 'enable_group_merge', False) and fused:
                        try:
                            import numpy as _np
                            from igp.fusion.nms import iou as _box_iou_fn
                            from PIL import Image as _PILImage

                            N = len(fused)
                            boxes_np = _np.asarray([_np.asarray(d.box, dtype=_np.float32) for d in fused]) if N else _np.zeros((0,4), dtype=_np.float32)
                            masks = []
                            for d in fused:
                                m = None
                                if getattr(d, 'extra', None) and isinstance(d.extra, dict):
                                    seg = d.extra.get('segmentation', None)
                                    msk = d.extra.get('mask', None)
                                    m = seg if seg is not None else msk
                                masks.append(m)

                            # Build adjacency using mask IoU when available, else box IoU
                            adj = {i: set() for i in range(N)}
                            mask_thr = float(self.merge_mask_iou_thr or 0.6)
                            box_thr = float(self.merge_box_iou_thr or 0.9)
                            for i in range(N):
                                for j in range(i + 1, N):
                                    connected = False
                                    mi = masks[i]
                                    mj = masks[j]
                                    if mi is not None and mj is not None:
                                        try:
                                            mi_arr = _np.asarray(mi).astype(bool)
                                            mj_arr = _np.asarray(mj).astype(bool)
                                            # resize mj to mi shape if needed
                                            if mi_arr.shape != mj_arr.shape:
                                                try:
                                                    mj_img = _PILImage.fromarray(mj_arr.astype(_np.uint8) * 255)
                                                    mj_img = mj_img.resize((mi_arr.shape[1], mi_arr.shape[0]), resample=_PILImage.NEAREST)
                                                    mj_arr = _np.asarray(mj_img).astype(bool)
                                                except Exception:
                                                    # if resize fails, crop to intersection
                                                    h = min(mi_arr.shape[0], mj_arr.shape[0])
                                                    w = min(mi_arr.shape[1], mj_arr.shape[1])
                                                    mi_c = mi_arr[:h, :w]
                                                    mj_c = mj_arr[:h, :w]
                                                    inter = _np.logical_and(mi_c, mj_c).sum()
                                                    union = _np.logical_or(mi_c, mj_c).sum()
                                                    iou_val = float(inter) / float(union) if union > 0 else 0.0
                                                    if iou_val >= mask_thr:
                                                        connected = True
                                        except Exception:
                                            connected = False
                                        else:
                                            inter = _np.logical_and(mi_arr, mj_arr).sum()
                                            union = _np.logical_or(mi_arr, mj_arr).sum()
                                            iou_val = float(inter) / float(union) if union > 0 else 0.0
                                            if iou_val >= mask_thr:
                                                connected = True

                                    if not connected:
                                        # fallback to box IoU
                                        try:
                                            val = float(_box_iou_fn(boxes_np[[i]], boxes_np[[j]]).item())
                                        except Exception:
                                            # manual IoU
                                            bx1, by1, bx2, by2 = boxes_np[i]
                                            cx1, cy1, cx2, cy2 = boxes_np[j]
                                            ix1 = max(bx1, cx1)
                                            iy1 = max(by1, cy1)
                                            ix2 = min(bx2, cx2)
                                            iy2 = min(by2, cy2)
                                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                            area1 = max(1e-6, (bx2 - bx1) * (by2 - by1))
                                            area2 = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
                                            val = inter / (area1 + area2 - inter + 1e-7)
                                        if val >= box_thr:
                                            connected = True

                                    if connected:
                                        adj[i].add(j)
                                        adj[j].add(i)

                            # Connected components
                            visited = set()
                            groups = []
                            for i in range(N):
                                if i in visited:
                                    continue
                                stack = [i]
                                comp = []
                                while stack:
                                    u = stack.pop()
                                    if u in visited:
                                        continue
                                    visited.add(u)
                                    comp.append(u)
                                    for v in adj.get(u, []):
                                        if v not in visited:
                                            stack.append(v)
                                groups.append(sorted(comp))

                            # Merge groups into new fused list
                            new_fused = []
                            for comp in groups:
                                if len(comp) == 1:
                                    new_fused.append(fused[comp[0]])
                                    continue
                                # choose representative by highest score
                                best_idx = max(comp, key=lambda k: float(getattr(fused[k], 'score', 0.0)))
                                # union bbox
                                xs = [_np.asarray(fused[k].box)[0] for k in comp]
                                ys = [_np.asarray(fused[k].box)[1] for k in comp]
                                xs2 = [_np.asarray(fused[k].box)[2] for k in comp]
                                ys2 = [_np.asarray(fused[k].box)[3] for k in comp]
                                merged_box = (float(min(xs)), float(min(ys)), float(max(xs2)), float(max(ys2)))
                                rep = fused[best_idx]
                                merged_score = max(float(getattr(fused[k], 'score', 0.0)) for k in comp)
                                merged_label = getattr(rep, 'label', getattr(rep, 'label', ''))
                                merged_source = getattr(rep, 'source', 'fusion:merged')
                                # merge masks if present
                                merged_mask = None
                                any_mask = any(masks[k] is not None for k in comp)
                                if any_mask:
                                    try:
                                        H_img = H
                                        W_img = W
                                        import numpy as __np
                                        accum = __np.zeros((H_img, W_img), dtype=bool)
                                        for k in comp:
                                            mk = masks[k]
                                            if mk is None:
                                                continue
                                            mm = __np.asarray(mk).astype(bool)
                                            if mm.shape != (H_img, W_img):
                                                try:
                                                    mm_img = _PILImage.fromarray(mm.astype(__np.uint8) * 255)
                                                    mm_img = mm_img.resize((W_img, H_img), resample=_PILImage.NEAREST)
                                                    mm = __np.asarray(mm_img).astype(bool)
                                                except Exception:
                                                    # try crop/pad
                                                    h = min(mm.shape[0], H_img)
                                                    w = min(mm.shape[1], W_img)
                                                    tmp = __np.zeros((H_img, W_img), dtype=bool)
                                                    tmp[:h, :w] = mm[:h, :w]
                                                    mm = tmp
                                            accum = __np.logical_or(accum, mm)
                                        merged_mask = accum
                                    except Exception:
                                        merged_mask = None

                                # build new Detection
                                try:
                                    new_det = Detection(box=merged_box, label=merged_label, score=float(merged_score), source=merged_source, extra=( {'segmentation': merged_mask} if merged_mask is not None else None))
                                except TypeError:
                                    new_det = Detection(box=merged_box, label=merged_label)
                                    new_det.score = float(merged_score)
                                    new_det.source = merged_source
                                    if merged_mask is not None:
                                        new_det.extra = {'segmentation': merged_mask}
                                new_fused.append(new_det)

                            fused = new_fused
                        except Exception:
                            logger.exception('Group merge failed; continuing with fused list')

                    if self.enable_cross_class_suppression:
                        # 🔧 ULTRA-AGGRESSIVE cross-class suppression with:
                        # 1. Containment removal (remove boxes fully inside others)
                        # 2. Aggressive IoU-based removal
                        # 3. Semantic similarity check for label deduplication
                        from igp.fusion.nms import iou as _iou_fn
                        import numpy as _np

                        if self.cross_class_iou_thr is not None:
                            cross_class_iou_thr = float(self.cross_class_iou_thr)
                        else:
                            cross_class_iou_thr = max(0.75, float(iou_thr) + 0.1)  # conservative by default

                        if fused:
                            boxes = [_np.asarray(d.box, dtype=_np.float32) for d in fused]
                            labels = [getattr(d, 'label', '') for d in fused]
                            idxs = list(range(len(fused)))
                            # sort by score desc
                            idxs_sorted = sorted(idxs, key=lambda i: float(getattr(fused[i], 'score', 1.0)), reverse=True)
                            keep = []
                            removed = set()
                            
                            # Helper: check containment (box i fully inside box j)
                            def is_contained(bi, bj, threshold=0.95):
                                """Check if bi is contained in bj (>threshold% area overlap)"""
                                bx1, by1, bx2, by2 = bi
                                cx1, cy1, cx2, cy2 = bj
                                ix1 = max(bx1, cx1)
                                iy1 = max(by1, cy1)
                                ix2 = min(bx2, cx2)
                                iy2 = min(by2, cy2)
                                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                area_i = max(1e-6, (bx2 - bx1) * (by2 - by1))
                                return (inter / area_i) >= threshold
                            
                            # Helper: check semantic similarity
                            def are_semantically_similar(label1, label2):
                                """Check if two labels refer to same concept"""
                                l1 = str(label1).lower().strip()
                                l2 = str(label2).lower().strip()
                                if l1 == l2:
                                    return True
                                # Common synonyms/variants
                                synonyms = [
                                    {'sofa', 'couch', 'settee'},
                                    {'table', 'desk'},
                                    {'chair', 'seat'},
                                    {'lamp', 'light'},
                                    {'tv', 'television', 'monitor', 'screen'},
                                    {'bed', 'mattress'},
                                    {'cabinet', 'cupboard', 'wardrobe'},
                                    {'rug', 'carpet', 'mat'},
                                    {'picture', 'painting', 'frame'},
                                    {'pillow', 'cushion'},
                                    {'plant', 'flower', 'potted plant'},
                                    {'book', 'books'},
                                    {'window', 'windows'},
                                    {'door', 'doors'},
                                    {'shelf', 'shelves', 'bookshelf'},
                                ]
                                for syn_set in synonyms:
                                    if l1 in syn_set and l2 in syn_set:
                                        return True
                                # Check substring match (e.g., "dining table" contains "table")
                                if l1 in l2 or l2 in l1:
                                    return True
                                return False
                            
                            for i in idxs_sorted:
                                if i in removed:
                                    continue
                                keep.append(i)
                                bi = boxes[i]
                                label_i = labels[i]
                                
                                # compare with remaining lower-score candidates
                                for j in idxs_sorted:
                                    if j == i or j in removed:
                                        continue
                                    bj = boxes[j]
                                    label_j = labels[j]
                                    
                                    # 🔧 Strategy 1: Remove if fully contained
                                    if is_contained(bj, bi, threshold=0.90):
                                        removed.add(j)
                                        continue
                                    
                                    # 🔧 Strategy 2: Remove if semantically similar AND high overlap
                                    are_similar = are_semantically_similar(label_i, label_j)
                                    
                                    # Calculate IoU
                                    try:
                                        iou_mat = _iou_fn(bi[None, :], bj[None, :])
                                        val = float(iou_mat.item()) if hasattr(iou_mat, 'item') else float(iou_mat)
                                    except Exception:
                                        # fallback to simple IoU computation
                                        bx1, by1, bx2, by2 = bi
                                        cx1, cy1, cx2, cy2 = bj
                                        ix1 = max(bx1, cx1)
                                        iy1 = max(by1, cy1)
                                        ix2 = min(bx2, cx2)
                                        iy2 = min(by2, cy2)
                                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                        area1 = max(1e-6, (bx2 - bx1) * (by2 - by1))
                                        area2 = max(1e-6, (cx2 - cx1) * (cy2 - cy1))
                                        val = inter / (area1 + area2 - inter + 1e-7)
                                    
                                    # 🔧 Use more aggressive threshold for similar labels
                                    effective_threshold = 0.40 if are_similar else cross_class_iou_thr
                                    
                                    if val >= effective_threshold:
                                        removed.add(j)

                            fused = [fused[i] for i in keep]
                except Exception:
                    logger.exception("Cross-class suppression failed; returning fused results as-is")

                # Mask-IoU based suppression (optional): if fused detections
                # carry segmentation masks, remove lower-scored detections that
                # have high mask overlap with a kept detection.
                if self.enable_mask_iou_suppression:
                    try:
                        import numpy as _np
                        from PIL import Image as _PILImage

                        masks = []
                        for d in fused:
                            m = None
                            if getattr(d, 'extra', None) and isinstance(d.extra, dict):
                                seg = d.extra.get('segmentation', None)
                                msk = d.extra.get('mask', None)
                                m = seg if seg is not None else msk
                            masks.append(m)

                        if any(m is not None for m in masks):
                            thr = float(self.mask_iou_thr) if self.mask_iou_thr is not None else 0.6
                            idxs_sorted = sorted(range(len(fused)), key=lambda i: float(getattr(fused[i], 'score', 1.0)), reverse=True)
                            keep = []
                            removed = set()
                            for i in idxs_sorted:
                                if i in removed:
                                    continue
                                keep.append(i)
                                mi = masks[i]
                                if mi is None:
                                    continue
                                mi = _np.asarray(mi).astype(bool)
                                for j in idxs_sorted:
                                    if j == i or j in removed:
                                        continue
                                    mj = masks[j]
                                    if mj is None:
                                        continue
                                    mj = _np.asarray(mj).astype(bool)
                                    # resize mj to mi shape if needed
                                    if mj.shape != mi.shape:
                                        try:
                                            mj_img = _PILImage.fromarray(mj.astype(_np.uint8) * 255)
                                            mj_img = mj_img.resize((mi.shape[1], mi.shape[0]), resample=_PILImage.NEAREST)
                                            mj = _np.asarray(mj_img).astype(bool)
                                        except Exception:
                                            # fallback: compare cropped intersection
                                            h = min(mi.shape[0], mj.shape[0])
                                            w = min(mi.shape[1], mj.shape[1])
                                            mi_c = mi[:h, :w]
                                            mj_c = mj[:h, :w]
                                            inter = _np.logical_and(mi_c, mj_c).sum()
                                            union = _np.logical_or(mi_c, mj_c).sum()
                                            iou_val = float(inter) / float(union) if union > 0 else 0.0
                                            if iou_val >= thr:
                                                removed.add(j)
                                            continue

                                    inter = _np.logical_and(mi, mj).sum()
                                    union = _np.logical_or(mi, mj).sum()
                                    iou_val = float(inter) / float(union) if union > 0 else 0.0
                                    if iou_val >= thr:
                                        removed.add(j)

                            fused = [fused[i] for i in keep if i not in removed]
                    except Exception:
                        logger.exception("Mask IoU suppression failed; continuing")

                # 🚀 NEW: Recover low-score detections if no competing objects in region
                if self.keep_non_competing_low_scores and fuse and all_original_detections:
                    try:
                        fused = self._recover_non_competing_detections(
                            fused,
                            all_original_detections,
                            iou_threshold=self.non_competing_iou_threshold,
                            min_score=self.non_competing_min_score
                        )
                    except Exception:
                        logger.exception("Non-competing detection recovery failed; continuing")

                results[orig_idx] = fused
                # store in cache
                try:
                    self._cache_set(keys[orig_idx], fused)
                except Exception:
                    logger.exception("Failed to cache results for image %s", orig_idx)
            else:
                results[orig_idx] = per_image_dets
                try:
                    self._cache_set(keys[orig_idx], per_image_dets)
                except Exception:
                    logger.exception("Failed to cache results for image %s", orig_idx)

        # Now results is fully populated for all images
        # record total time and log structured stats
        try:
            total = time.monotonic() - t_start
            self.last_run_stats["total_s"] = total
            # structured log (many loggers accept `extra` or JSON formatting)
            logger.info("DetectorManager stats", extra={"detector_stats": self.last_run_stats, "n_images": len(images), "total_s": total})
        except Exception:
            pass

        # ✅ Aggiungi suffissi numerici univoci a ogni oggetto
        for img_idx, dets in enumerate(results):
            if dets:
                labels = [d.label for d in dets]
                unique_labels = self._add_unique_suffixes(labels)
                for det, new_label in zip(dets, unique_labels):
                    det.label = new_label

        if return_stats:
            return results, dict(self.last_run_stats)  # type: ignore[return-value]
        return results  # type: ignore[return-value]

    # ----------------- helpers -----------------

    @staticmethod
    def _ensure_rgb(image: Image.Image) -> Image.Image:
        if isinstance(image, Image.Image) and image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _run_detector_batch(self, det: Detector, images: Sequence[Image.Image]) -> List[List[Detection]]:
        """Run a detector on a list of PIL images, using `detect_batch` when supported.

        The detector is responsible for respecting its internal device and
        implementing batching efficiently when supported.
        """
        import time
        t0 = time.monotonic()
        try:
            if det.supports_batch:
                out = det.detect_batch(list(images))
            else:
                # fallback: call detect_batch from base which parallelizes internally
                out = det.detect_batch(list(images))
            return out
        except Exception:
            logger.exception("Detector %s failed in _run_detector_batch", det)
            return [[] for _ in images]
        finally:
            # record last run stats on the detector instance for external collection
            try:
                duration = time.monotonic() - t0
                setattr(det, "_last_run_stats", {"duration_s": duration, "num_images": len(images)})
                logger.info("Detector %s: ran on %d images in %.3fs", det.__class__.__name__, len(images), duration)
            except Exception:
                pass

    @staticmethod
    def _add_unique_suffixes(labels: list[str]) -> list[str]:
        """
        Aggiungi suffissi numerici univoci a ogni oggetto della stessa classe.
        
        Es: ["chair", "chair", "table", "chair"] → ["chair_1", "chair_2", "table_1", "chair_3"]
        
        Args:
            labels: Lista di label senza suffissi
            
        Returns:
            Lista di label con suffissi numerici univoci
        """
        label_counts = {}
        unique_labels = []
        
        for label in labels:
            # Rimuovi eventuali suffissi esistenti
            base_label = label.rsplit("_", 1)[0] if "_" in label and label.split("_")[-1].isdigit() else label
            
            # Incrementa il contatore per questa classe
            if base_label not in label_counts:
                label_counts[base_label] = 0
            label_counts[base_label] += 1
            
            # Crea label con suffisso univoco
            unique_label = f"{base_label}_{label_counts[base_label]}"
            unique_labels.append(unique_label)
        
        return unique_labels

    def _recover_non_competing_detections(
        self,
        fused_detections: List[Detection],
        all_original_detections: List[Detection],
        iou_threshold: float = 0.30,
        min_score: float = 0.05,
    ) -> List[Detection]:
        """
        🚀 Recover low-score detections that don't compete with higher-score ones.
        
        Strategy:
        1. For each original detection below the normal threshold
        2. Check if it has significant overlap (IoU > threshold) with ANY fused detection
        3. If NO overlap, it's a non-competing detection → keep it
        4. Only keep detections above min_score to filter noise
        
        Args:
            fused_detections: High-confidence detections after fusion/NMS
            all_original_detections: All original detections before filtering
            iou_threshold: IoU threshold to determine if regions compete
            min_score: Minimum score for recovery (avoid complete noise)
            
        Returns:
            Combined list of fused detections + recovered non-competing ones
        """
        if not all_original_detections or not fused_detections:
            return fused_detections
        
        try:
            import numpy as np
            from igp.fusion.nms import iou as compute_iou
            
            # Get boxes from fused detections
            fused_boxes = np.array([np.array(d.box, dtype=np.float32) for d in fused_detections])
            
            # Prepare to collect recovered detections
            recovered = []
            
            # Check each original detection
            for orig_det in all_original_detections:
                # Skip if already in fused (high score)
                orig_score = float(getattr(orig_det, 'score', 0.0))
                
                # Only consider detections above minimum threshold
                if orig_score < min_score:
                    continue
                
                # Check if this detection is already represented in fused results
                # by comparing boxes
                orig_box = np.array(orig_det.box, dtype=np.float32)
                
                # Compute IoU with all fused boxes
                ious = compute_iou(orig_box[None, :], fused_boxes)
                max_iou = float(np.max(ious)) if len(ious) > 0 else 0.0
                
                # If no significant overlap with any fused detection, this is non-competing
                if max_iou < iou_threshold:
                    # Check it's not a duplicate of an already recovered detection
                    if recovered:
                        recovered_boxes = np.array([np.array(d.box, dtype=np.float32) for d in recovered])
                        ious_rec = compute_iou(orig_box[None, :], recovered_boxes)
                        if float(np.max(ious_rec)) >= iou_threshold:
                            continue  # Already have similar recovered detection
                    
                    # This is a valid non-competing detection - recover it!
                    recovered.append(orig_det)
                    logger.debug(
                        f"Recovered non-competing detection: {getattr(orig_det, 'label', 'unknown')} "
                        f"with score {orig_score:.3f} (max_iou={max_iou:.3f})"
                    )
            
            if recovered:
                logger.info(f"Recovered {len(recovered)} non-competing low-score detections")
                # Combine fused + recovered
                return list(fused_detections) + recovered
            
            return fused_detections
            
        except Exception as e:
            logger.exception(f"Error in non-competing detection recovery: {e}")
            return fused_detections


__all__ = ["DetectorManager"]

