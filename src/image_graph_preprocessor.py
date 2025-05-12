import os
import sys
import math
import time
import json
import torch
import numpy as np
import cv2
import spacy
import nltk
from nltk.corpus import wordnet
from spacy.matcher import PhraseMatcher
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
from collections import defaultdict
import argparse
from pathlib import Path
from torch.hub import download_url_to_file
from spacy.matcher import PhraseMatcher
from adjustText import adjust_text


# YOLOv8
from ultralytics import YOLO

# Detectron2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

# OWL-ViT
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Segment Anything (SAM)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

# For optional Hugging Face dataset usage
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False


##############################################################################
# CLASS DEFINITION
##############################################################################
class ImageGraphPreprocessor:
    """
    A class-based Image Graph Preprocessor that can handle multiple detectors,
    run relationship inference, segment objects with SAM, filter detections
    by question or area, and visualize results.

    Now if the question yields zero recognized object terms or relationship terms,
    we skip question-based filtering for that category – effectively ignoring the question
    at that step (object or relationship).
    """

    def __init__(self, config: dict):
        """
        Initialize the ImageGraphPreprocessor with models and parameters.
        """
        # ---------------------------------------------------------
        # 1) Store config and parse relevant fields
        # ---------------------------------------------------------
        self.config = config

        # Basic paths / dataset fields
        self.input_path = config.get("input_path", None)
        self.output_folder = config.get("output_folder", "output_images")
        self.dataset_name = config.get("dataset", None)
        self.dataset_split = config.get("split", "train")
        self.image_column = config.get("image_column", "image")

        # Detectors & thresholds
        self.detectors_to_use = config.get("detectors_to_use", ["owlvit", "yolov8", "detectron2"])
        self.threshold_owl = config.get("threshold_owl", 0.5)
        self.threshold_yolo = config.get("threshold_yolo", 0.5)
        self.threshold_detectron = config.get("threshold_detectron", 0.5)

        # Relationship settings
        self.max_relations = config.get("max_relations", 10)
        self.proportional_relations = config.get("proportional_relations", False)
        self.relation_ratio = config.get("relation_ratio", 1.0)
        self.max_relations_per_object = config.get("max_relations_per_object", 1)
        self.min_relations_per_object = config.get("min_relations_per_object", 1)
        self.filter_relations_by_question = config.get("filter_relations_by_question", True)
        self.threshold_object_similarity   = config.get("threshold_object_similarity", 0.5)
        self.threshold_relation_similarity = config.get("threshold_relation_similarity", 0.5)

        # Label & visualization settings
        self.question = config.get("question", "")
        self.apply_question_filter = config.get("apply_question_filter", True)
        self.label_mode = config.get("label_mode", "original")  # "original","numeric","alphabetic"
        self.display_labels = config.get("display_labels", True)
        self.display_relationships = config.get("display_relationships", True)
        self.show_segmentation = config.get("show_segmentation", True)
        self.fill_segmentation = config.get("fill_segmentation", True)

        # Additional toggles
        self.show_confidence = config.get("show_confidence", False)
        self.display_relation_labels = config.get("display_relation_labels", False)

        # Area filtering
        self.enable_area_filter = config.get("enable_area_filter", False)
        self.min_area = config.get("min_area", 500)
        self.max_area = config.get("max_area", None)

        # Overlap & NMS thresholds
        self.label_nms_threshold = config.get("label_nms_threshold", 0.5)
        self.seg_iou_threshold = config.get("seg_iou_threshold", 0.5)

        # Relationship inference geometry
        self.margin = config.get("margin", 20)
        self.min_distance = config.get("min_distance", 90)
        self.max_distance = config.get("max_distance", 20000)

        # SAM parameters
        self.points_per_side = config.get("points_per_side", 32)
        self.pred_iou_thresh = config.get("pred_iou_thresh", 0.7)
        self.stability_score_thresh = config.get("stability_score_thresh", 0.85)
        self.min_mask_region_area = config.get("min_mask_region_area", 100)

        # Dataset indexing
        self.start_index = config.get("start_index", -1)
        self.end_index = config.get("end_index", -1)
        self.num_instances = config.get("num_instances", -1)

        # ---------------------------------------------------------
        # 2) Setup environment
        # ---------------------------------------------------------
        os.makedirs(self.output_folder, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Running on device: {self.device}")
        
        os.makedirs(self.output_folder, exist_ok=True)
        # se specificato in config, usiamo quel device (es. 'cpu')
        if self.config.get("preproc_device"):
            self.device = self.config["preproc_device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Running preprocessor on device: {self.device}")

        # Load SpaCy for question or text-based processing
        print("[INFO] Loading SpaCy 'en_core_web_md' model ...")
        self.nlp = spacy.load("en_core_web_md")
        self.question_doc = self.nlp(self.question) if self.question.strip() else None

        # Build matchers per relazioni
        self.relation_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._build_relation_matcher()

        # ---------------------------------------------------------
        # 3) Load Segment Anything (SAM) model
        # ---------------------------------------------------------
        self._load_sam_model()

        # ---------------------------------------------------------
        # 4) Initialize requested detectors
        # ---------------------------------------------------------
        self.owlvit_processor = None
        self.owlvit_model = None
        self.yolo_model = None
        self.d2_predictor = None
        self.d2_metadata = None
        self._init_detectors()

        owl_queries = [
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
                "printer","monitor","picture", "pillow","stone","glasses","wheel","building","bridge","tomato"
        ]

        self.detectors_label_list = []
        # Se OWL-ViT è stato caricato, aggiungo le sue label
        if self.owlvit_model:
            self.detectors_label_list.extend(owl_queries)

        # Se YOLOv8 è stato caricato, prendo le sue names
        if self.yolo_model:
            # results.names è dict[int,str], quindi ne estraggo i valori
            self.detectors_label_list.extend(list(self.yolo_model.names.values()))

        # Se Detectron2 è stato caricato, prendo thing_classes
        if self.d2_metadata:
            self.detectors_label_list.extend(self.d2_metadata.thing_classes)

        self.label_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp(lbl.lower()) for lbl in self.detectors_label_list]
        self.label_matcher.add("LABEL", None, *patterns)

        # —––––––––––––––––––––––––––––––––––––––––––––––—
        # 4) ORA posso ricostruire i termini della domanda
        # —––––––––––––––––––––––––––––––––––––––––––––––—

        # Reset dei set (utile se __init__ viene richiamato due volte)
        self._parsed_question_object_terms   = set()
        self._parsed_question_relation_terms = set()

        # Ed ecco la chiamata che ora funziona correttamente:
        self._build_question_semantics()

    ###########################################################################
    # QUESTION SEMANTICS
    ###########################################################################
    def _build_relation_matcher(self):
        base_map = {
            "above":   ["above"],
            "below":   ["below", "under"],
            "left_of": ["left of", "to the left of"],
            "right_of":["right of", "to the right of"]
        }
        for label, terms in base_map.items():
            patterns = [self.nlp(term) for term in terms]
            self.relation_matcher.add(label, None, *patterns)


    def _build_question_semantics(self):
        # reset
        self._parsed_question_object_terms = set()
        if not self.apply_question_filter or not self.question.strip():
            return

        doc = self.nlp(self.question)

        # a) exact phrase‐matches of detector labels
        for _, start, end in self.label_matcher(doc):
            self._parsed_question_object_terms.add(doc[start:end].text.lower())

        # b) fallback to nouns+WordNet if no labels matched
        if not self._parsed_question_object_terms:
            nouns = {
                tok.lemma_.lower()
                for tok in doc
                if tok.pos_ in {"NOUN","PROPN"} and not tok.is_stop
            }
            for term in list(nouns):
                for syn in wordnet.synsets(term):
                    nouns |= {lem.name().lower().replace("_"," ")
                              for lem in syn.lemmas()}
            self._parsed_question_object_terms = nouns




    def _extract_query_terms(self, text: str) -> list:
        """
        Extract candidate object query terms (nouns/proper nouns) from text via spaCy.
        """
        doc = self.nlp(text)
        candidates = []
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                candidates.append(token.lemma_.lower())
        return list(set(candidates))

    def _get_wordnet_synonyms(self, term: str) -> set:
        synonyms = set()
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace("_", " "))
        return synonyms

    def _extract_relation_terms(self, text: str) -> list:
        """
        Identify which of 'above/below/left_of/right_of' are in the question (including synonyms).
        """

        base_map = {
            "above":   ["above"],
            "below":   ["below", "under"],
            "left_of": ["left", "to the left of"],
            "right_of": ["right", "to the right of"]
        }

        # gather synonyms from WordNet
        all_relations = {}
        for key, base_list in base_map.items():
            syn_set = set()
            for base_word in base_list:
                for syn in wordnet.synsets(base_word):
                    for lemma in syn.lemmas():
                        syn_set.add(lemma.name().lower().replace("_", " "))
                syn_set.add(base_word.lower())
            all_relations[key] = syn_set

        text_lower = text.lower()
        found_rel_types = set()
        for rel_label, synonyms_for_rel in all_relations.items():
            for candidate in synonyms_for_rel:
                if candidate in text_lower:
                    found_rel_types.add(rel_label)
                    break

        return list(found_rel_types)

    ###########################################################################
    # MODEL LOADING
    ###########################################################################
    def _load_sam_model(self):
        print("[INFO] Loading SAM model ...")
        # Recupera il checkpoint dalla configurazione o usa il default
        sam_checkpoint = self.config.get("sam_checkpoint", "sam_vit_h_4b8939.pth")
        sam_model_type = self.config.get("sam_model_type", "vit_h")

        # Se il percorso non è assoluto, considera il file nella cartella di output
        checkpoint_path = (Path(self.output_folder) / sam_checkpoint) if not Path(sam_checkpoint).is_absolute() else Path(sam_checkpoint)

        # Controlla se il file esiste, altrimenti scaricalo
        SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not checkpoint_path.exists():
            print("[INFO] Scarico checkpoint SAM …")
            download_url_to_file(SAM_CKPT_URL, str(checkpoint_path))

        self.sam_model = sam_model_registry[sam_model_type](checkpoint=str(checkpoint_path))
        self.sam_model.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam_model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area
        )
        self.sam_predictor = SamPredictor(self.sam_model)

    def _init_detectors(self):
        if "owlvit" in self.detectors_to_use:
            print("[INFO] Loading OWL-ViT ...")
            self.owlvit_processor, self.owlvit_model = self._load_owlvit_detector()

        if "yolov8" in self.detectors_to_use:
            print("[INFO] Loading YOLOv8 ...")
            self.yolo_model = self._load_yolov8_detector()

        if "detectron2" in self.detectors_to_use:
            print("[INFO] Loading Detectron2 ...")
            self.d2_predictor, self.d2_metadata = self._load_detectron2_detector()

    def _load_owlvit_detector(self):
        processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-large-patch14-ensemble",
            torch_dtype=torch.float16,
            #low_cpu_mem_usage=False
        )
        model.to(self.device)
        model.eval()
        return processor, model

    def _load_yolov8_detector(self, model_path="yolov8x.pt"):
        model = YOLO(model_path)
        model.to(self.device)
        return model

    def _load_detectron2_detector(self, model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold_detectron
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        return predictor, metadata

    ###########################################################################
    # DETECTOR INFERENCE
    ###########################################################################
    @torch.inference_mode()
    def _run_owlvit_detection(self, image_pil: Image.Image, queries: list[str]) -> list[dict]:
        """
        Run OWL-ViT grounded object detection on a single PIL.Image
        using a list of text queries.
        """
        if not (self.owlvit_processor and self.owlvit_model):
            return []

        # 1) OWL-ViT expects a _batch_ of images, even if it's 1 item
        images = [image_pil]

        # 2) Build inputs: pass images first, then text
        encoding = self.owlvit_processor(
            images=images,
            text=queries,
            return_tensors="pt"
        )
        # 3) Move all tensors to the correct device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # 4) Forward pass
        outputs = self.owlvit_model(**encoding)

        # 5) Post-process at the original image size
        w, h = image_pil.size
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.owlvit_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes
        )

        # 6) If nothing was detected, return empty
        if not results or results[0] is None:
            return []
        res = results[0]

        # 7) Pull out boxes, scores, labels
        # boxes  = res["boxes"].cpu()
        # scores = res["scores"].cpu()
        # labels = res["labels"].cpu()
        
        boxes  = res["boxes"]
        scores = res["scores"]
        labels = res["labels"]

        # 8) Filter by your threshold and package
        detections = []
        for box, score, lab_idx in zip(boxes, scores, labels):
            if score >= self.threshold_owl:
                detections.append({
                    "box":   box.tolist(),
                    "label": queries[lab_idx],  # map back to your query
                    "score": float(score)
                })
        return detections



    @torch.inference_mode()
    def _run_yolov8_detection(self, image_pil: Image.Image) -> list:
        if not self.yolo_model:
            return []
        image_np = np.array(image_pil)
        results = self.yolo_model.predict(image_np, device=self.device)[0]
        threshold = self.threshold_yolo

        detections = []
        for box_xyxy, conf, cls_idx in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            score_val = float(conf.item())
            if score_val < threshold:
                continue
            x_min, y_min, x_max, y_max = box_xyxy.tolist()
            label_idx = int(cls_idx.item())
            label_name = results.names[label_idx]
            detections.append({
                "box": [x_min, y_min, x_max, y_max],
                "label": label_name,
                "score": score_val
            })
        return detections

    @torch.inference_mode()
    def _run_detectron2_detection(self, image_pil: Image.Image):
        if not (self.d2_predictor and self.d2_metadata):
            return []

        # 1) run the model
        image_np = np.array(image_pil)
        outputs  = self.d2_predictor(image_np)
        # instances = outputs["instances"].to("cpu")
        instances = outputs["instances"]

        # 2) extract all the fields
        boxes   = instances.pred_boxes.tensor.numpy().tolist()   # list of [x1,y1,x2,y2]
        scores  = instances.scores.numpy().tolist()             # list of floats
        classes = instances.pred_classes.numpy().tolist()               # list of ints
        masks   = instances.pred_masks.numpy()                  # array (N, H, W) of bool

        detections = []
        for box, sc, cls_idx, mask in zip(boxes, scores, classes, masks):
            if sc < self.threshold_detectron:
                continue

            label = self.d2_metadata.thing_classes[cls_idx]
            detections.append({
                "box":       box,
                "score":     float(sc),
                "label":     label,
                "det2_mask": mask.astype(bool)   # full‐size binary mask
            })

        return detections


    ###########################################################################
    # BOX / SEGMENTATION / RELATIONSHIP UTILS
    ###########################################################################
    def _mask_iou(self, m1: np.ndarray, m2: np.ndarray) -> float:
      inter = np.logical_and(m1, m2).sum()
      union = np.logical_or(m1, m2).sum()
      return inter / union if union > 0 else 0.0

    def _fuse_masks(self, sam_mask: np.ndarray, det2_mask: np.ndarray,
                    method="union", iou_thresh=0.5) -> np.ndarray:
        iou = self._mask_iou(sam_mask, det2_mask)
        if method == "iou_union" and iou >= iou_thresh:
            return np.logical_or(sam_mask, det2_mask)
        if method == "union":
            return np.logical_or(sam_mask, det2_mask)
        if method == "intersection":
            return np.logical_and(sam_mask, det2_mask)
        # fallback to SAM
        return sam_mask


    def _compute_iou(self, box1: list, box2: list) -> float:
        x1, y1, x2, y2 = box1
        xA, yA, xB, yB = box2
        inter_x1 = max(x1, xA)
        inter_y1 = max(y1, yA)
        inter_x2 = min(x2, xB)
        inter_y2 = min(y2, yB)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (xB - xA) * (yB - yA)
        return inter_area / float(area1 + area2 - inter_area)

    def _non_maximum_suppression(self, boxes: list, scores: list,
                                 iou_threshold: float) -> list:
        idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        keep = []
        while idxs:
            curr = idxs.pop(0)
            keep.append(curr)
            new_list = []
            for i in idxs:
                if self._compute_iou(boxes[curr], boxes[i]) < iou_threshold:
                    new_list.append(i)
            idxs = new_list
        return keep

    def _get_best_mask_for_box(self, detection_box: list, masks: list, iou_threshold: float = 0.3):
        best_mask = None
        best_iou = 0.0
        best_score = -1
        for m in masks:
            x, y, w, h = m['bbox']
            mask_box = [x, y, x + w, y + h]
            iou = self._compute_iou(detection_box, mask_box)
            if iou > best_iou:
                best_iou = iou
                best_score = m.get('predicted_iou', 0)
                best_mask = m
            elif math.isclose(iou, best_iou) and iou >= iou_threshold:
                score = m.get('predicted_iou', 0)
                if score > best_score:
                    best_mask = m
                    best_score = score
        return best_mask if best_iou >= iou_threshold else None

    def _bbox_from_contour(self, contour: np.ndarray) -> list:
        x, y, w, h = cv2.boundingRect(contour)
        return [x, y, x + w, y + h]

    def _filter_segmentation_duplicates(self, boxes, labels, scores, all_masks):
        """
        Refine SAM masks to detection bboxes and remove duplicates via NMS.

        Args:
            boxes: list of [x1, y1, x2, y2]
            labels: list of str
            scores: list of float
            all_masks: list of dict con chiavi 'segmentation' (H×W bool) e 'bbox' originale

        Returns:
            new_boxes, new_labels, new_scores, new_masks
              - new_masks è una list di dict identici a all_masks ma con 'bbox' aggiornato
        """

        refined_boxes   = []
        refined_labels  = []
        refined_scores  = []
        refined_masks   = []

        # 1) Per ogni detection, scegli la maschera migliore e ricalcola il bbox
        for box, lab, sc, m in zip(boxes, labels, scores, all_masks):
            best = self._get_best_mask_for_box(box, [m])
            if best is not None:
                # Estraggo la maschera e i contorni
                mask_uint8 = (best["segmentation"] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    refined_box = self._bbox_from_contour(c)
                    # aggiorno anche il campo 'bbox' della maschera
                    best["bbox"] = refined_box
                else:
                    refined_box = box
            else:
                refined_box = box
                best = m  # uso la maschera originale

            refined_boxes.append(refined_box)
            refined_labels.append(lab)
            refined_scores.append(sc)
            refined_masks.append(best)

        # 2) Non‐Maximum Suppression sui bbox rifiniti
        keep = self._non_maximum_suppression(refined_boxes,
                                            refined_scores,
                                            self.seg_iou_threshold)

        # 3) Ricostruisco le liste “filtrate”
        new_boxes  = [refined_boxes[i]   for i in keep]
        new_labels = [refined_labels[i]  for i in keep]
        new_scores = [refined_scores[i]  for i in keep]
        new_masks  = [refined_masks[i]   for i in keep]

        return new_boxes, new_labels, new_scores, new_masks



    ###########################################################################
    # RELATIONSHIP INFERENCE
    ###########################################################################
    def _infer_relationships_improved(self, boxes: list) -> list:
        n = len(boxes)
        centers = []
        for b in boxes:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            centers.append((cx, cy))

        rels = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                dx = cx2 - cx1
                dy = cy2 - cy1
                dist = math.hypot(dx, dy)
                if dist < self.min_distance or dist > self.max_distance:
                    continue

                # If vertically distinct
                if abs(dy) >= abs(dx) and abs(dy) > self.margin:
                    relation = "above" if dy > 0 else "below"
                # Else horizontally distinct
                elif abs(dx) > self.margin:
                    relation = "left_of" if dx > 0 else "right_of"
                else:
                    continue

                rels.append({
                    "src_idx": i,
                    "tgt_idx": j,
                    "relation": relation,
                    "distance": dist
                })

        return rels

    def _limit_relationships_per_object(self, relationships: list, boxes: list) -> list:
        rels_by_src = defaultdict(list)
        for r in relationships:
            rels_by_src[r["src_idx"]].append(r)

        n = len(boxes)
        centers = []
        for b in boxes:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            centers.append((cx, cy))

        for i in range(n):
            if len(rels_by_src[i]) < self.min_relations_per_object:
                best_j = None
                best_d = float('inf')
                for j in range(n):
                    if j == i:
                        continue
                    d = math.hypot(centers[i][0] - centers[j][0],
                                   centers[i][1] - centers[j][1])
                    if d < best_d and d >= self.min_distance:
                        best_j = j
                        best_d = d
                if best_j is not None:
                    rel_nearest = {
                        "src_idx": i,
                        "tgt_idx": best_j,
                        "relation": "nearest",
                        "distance": best_d
                    }
                    if all(rr["tgt_idx"] != best_j for rr in rels_by_src[i]):
                        rels_by_src[i].append(rel_nearest)

        final_list = []
        for src_idx, rlist in rels_by_src.items():
            sorted_rlist = sorted(rlist, key=lambda r: r["distance"])
            final_list.extend(sorted_rlist[: self.max_relations_per_object])
        return final_list

    def _unify_pair_relations(self, relationships: list) -> list:
        best_for_pair = {}
        for r in relationships:
            i = r["src_idx"]
            j = r["tgt_idx"]
            pair = tuple(sorted((i, j)))
            if pair not in best_for_pair:
                best_for_pair[pair] = r
            else:
                if r["distance"] < best_for_pair[pair]["distance"]:
                    best_for_pair[pair] = r
        return list(best_for_pair.values())

    ###########################################################################
    # QUESTION & AREA FILTERING
    ###########################################################################
    def _filter_by_area(self, boxes, labels, scores):
        if not self.enable_area_filter:
            return boxes, labels, scores

        new_boxes, new_labels, new_scores = [], [], []
        for b, lab, sc in zip(boxes, labels, scores):
            x1, y1, x2, y2 = b
            w = (x2 - x1)
            h = (y2 - y1)
            area = w * h

            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            new_boxes.append(b)
            new_labels.append(lab)
            new_scores.append(sc)
        return new_boxes, new_labels, new_scores

    def _filter_by_question(self, boxes, labels, scores):
        if not self.apply_question_filter:
            return boxes, labels, scores

        new_b, new_l, new_s = [], [], []
        for b, lab, sc in zip(boxes, labels, scores):
            lab_low = lab.lower()

            if lab_low in self._parsed_question_object_terms:
                keep = True
            else:
                sims = [
                    self.nlp(lab_low).similarity(self.nlp(term))
                    for term in self._parsed_question_object_terms
                ]
                keep = sims and max(sims) >= self.threshold_object_similarity

            if keep:
                new_b.append(b)
                new_l.append(lab)
                new_s.append(sc)

        return new_b, new_l, new_s


    def _filter_relationships_by_question(self, relationships):
        """
        Filters a list of relationship dicts based on question-specified relation terms
        and a spaCy similarity threshold.
        """
        # If filtering is disabled, return all relationships
        if not self.filter_relations_by_question:
            return relationships

        filtered = []
        question_terms = self._parsed_question_relation_terms

        for r in relationships:
            rel_label = r["relation"]

            # 1) Exact match against parsed question terms
            if rel_label in question_terms:
                filtered.append(r)
                continue

            # 2) If no question terms, skip similarity check
            if not question_terms:
                continue

            # 3) Compute spaCy similarity between the relation label and each question term
            rel_doc = self.nlp(rel_label.replace("_", " "))
            similarities = [
                rel_doc.similarity(self.nlp(term))
                for term in question_terms
            ]
            best_sim = max(similarities) if similarities else 0.0

            # 4) Keep the relation if similarity exceeds threshold
            if best_sim >= self.threshold_relation_similarity:
                filtered.append(r)

        return filtered



    ###########################################################################
    # VISUALIZATION
    ###########################################################################
    def _move_point_outside_contour(self, candidate: tuple, contour: np.ndarray,
                                    step: float = 5, max_iter: int = 20) -> tuple:
        new_candidate = np.array(candidate, dtype=float)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            return candidate
        direction = new_candidate - np.array([cx, cy])
        norm = np.linalg.norm(direction)
        if norm == 0:
            direction = np.array([1.0, 0.0])
        else:
            direction /= norm

        for _ in range(max_iter):
            if cv2.pointPolygonTest(contour, tuple(new_candidate), False) >= 0:
                new_candidate += step * direction
            else:
                break
        return tuple(new_candidate)

    def _adjust_position(self, candidate: tuple, placed_positions: list,
                         overlap_thresh: float, max_iterations: int = 10) -> tuple:
        new_candidate = np.array(candidate, dtype=float)
        eps = 1e-6
        for _ in range(max_iterations):
            displacement = np.zeros(2, dtype=float)
            for p in placed_positions:
                diff = new_candidate - np.array(p, dtype=float)
                dist = np.linalg.norm(diff)
                if dist < overlap_thresh:
                    push = (overlap_thresh - dist) * (diff / (dist + eps))
                    displacement += push
            if np.linalg.norm(displacement) < 1e-3:
                break
            new_candidate += displacement
        return tuple(new_candidate)

    def _visualize_detections_and_relationships_with_auto_masks(
        self,
        image: Image.Image,
        boxes: list,
        labels: list,
        scores: list,
        relationships: list,
        all_masks: list,
        save_path: str = None
    ):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.axis("off")

        placed_positions = []
        overlap_threshold = 30
        color_list = ['red', 'green', 'blue', 'magenta', 'cyan', 'orange', 'purple', 'brown',
              'yellow', 'pink', 'lime', 'teal', 'navy', 'indigo', 'olive', 'maroon', 'gold', 'turquoise', 'violet']
        obj_colors = [color_list[i % len(color_list)] for i in range(len(boxes))]

        # Decide label text style
        if self.label_mode == "numeric":
            vis_labels = [str(i + 1) for i in range(len(boxes))]
        elif self.label_mode == "alphabetic":
            vis_labels = list(string.ascii_uppercase[:len(boxes)])
        else:
            vis_labels = labels

        detection_labels_info = []
        arrow_counts = defaultdict(int)

        # Draw detections
        for i, box in enumerate(boxes):
            color = obj_colors[i]
            x_min, y_min, x_max, y_max = map(int, box)
            center_pt = ((x_min + x_max)//2, (y_min + y_max)//2)

            best_mask = self._get_best_mask_for_box(box, all_masks)
            if self.show_segmentation and best_mask is not None:
                mask_uint8 = (best_mask['segmentation'] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea).squeeze()
                    if largest_contour.ndim >= 2:
                        if self.fill_segmentation:
                            polygon = largest_contour.reshape(-1, 2)
                            ax.fill(
                                polygon[:,0], polygon[:,1],
                                color=color, alpha=0.3, zorder=1
                            )
                        ax.plot(
                            largest_contour[:,0], largest_contour[:,1],
                            color=color, linewidth=2, zorder=2
                        )
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            center_pt = (cx, cy)
                        center_pt = self._move_point_outside_contour(center_pt, largest_contour)
                    else:
                        rect = patches.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            linewidth=2,
                            edgecolor=color,
                            facecolor='none',
                            zorder=2
                        )
                        ax.add_patch(rect)
                else:
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none',
                        zorder=2
                    )
                    ax.add_patch(rect)
            else:
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                    zorder=2
                )
                ax.add_patch(rect)

            center_pt = self._adjust_position(center_pt, placed_positions, overlap_threshold)
            placed_positions.append(center_pt)

            if self.display_labels:
                if self.show_confidence:
                    text_label = f"{vis_labels[i]}: {scores[i]:.2f}"
                else:
                    text_label = vis_labels[i]
                detection_labels_info.append((center_pt, text_label, color))

        if self.display_relationships and len(relationships) > 0:
            centers = []
            for b in boxes:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                centers.append((cx, cy))

            for rel in relationships:
                s = rel["src_idx"]
                t = rel["tgt_idx"]
                # Salta la relazione se uno o entrambi gli indici non sono validi
                if s >= len(centers) or t >= len(centers):
                    continue
                rel_label = rel["relation"]
                color = obj_colors[s]
                start_pt = centers[s]
                end_pt = centers[t]

                arrow_counts[(s, t)] += 1
                arrow_index = arrow_counts[(s, t)]
                rad_offset = 0.2 + 0.1 * (arrow_index - 1)

                arrow = patches.FancyArrowPatch(
                    start_pt, end_pt,
                    arrowstyle='->',
                    color=color,
                    linewidth=2,
                    connectionstyle=f'arc3,rad={rad_offset}',
                    mutation_scale=12,
                    zorder=4
                )
                ax.add_patch(arrow)

                if self.display_relation_labels:
                    mx = (start_pt[0] + end_pt[0]) / 2
                    my = (start_pt[1] + end_pt[1]) / 2
                    dx = end_pt[0] - start_pt[0]
                    dy = end_pt[1] - start_pt[1]
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    if angle_deg < -90:
                        angle_deg += 180
                    elif angle_deg > 90:
                        angle_deg -= 180
                    mag = math.hypot(dx, dy)
                    base_offset = 10 if mag > 0 else 0
                    offset_x, offset_y = 0, 0
                    if mag > 0:
                        offset_x = -dy / mag * base_offset
                        offset_y = dx  / mag * base_offset
                    candidate = (mx + offset_x, my + offset_y)
                    candidate = self._adjust_position(candidate, placed_positions, overlap_threshold)
                    placed_positions.append(candidate)

                    ax.text(
                        candidate[0], candidate[1],
                        rel_label,
                        fontsize=8, color=color,
                        rotation=angle_deg,
                        rotation_mode='anchor',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=color),
                        zorder=5
                    )

        texts = []
        for i, (pt, text, color) in enumerate(detection_labels_info):
            t = ax.text(
                pt[0], pt[1], text,
                fontsize=10, color=color,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color),
                zorder=7
            )
            texts.append(t)

        adjust_text(
            texts,
            ax=ax,
            only_move={'points':'y', 'text':'xy'},
            expand_text=(1.1, 1.1),
            force_text=0.5,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5)
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[INFO] Saved output to {save_path}")
        else:
            plt.show()

    def _refine_mask_with_point(self, image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Given an initial binary mask, sample its centroid as a positive prompt
        and run SAM’s point‐prompt to get a tighter mask.
        """
        # 1) Compute centroid of the mask
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return mask  # empty or invalid
        cx, cy = int(xs.mean()), int(ys.mean())

        # 2) Prompt SAM
        self.sam_predictor.set_image(image_np)
        point_coords  = np.array([[cx, cy]])
        point_labels  = np.array([1])  # 1 = “foreground”
        masks_pts, scores_pts, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        # 3) Return the single refined mask
        return masks_pts[0]

    @torch.inference_mode()
    def _run_sam_combined_segmentation(self, image_pil: Image.Image, boxes: list[list[float]]):
        image_np = np.array(image_pil)

        # ——— Stage 1: Global masks ——————————————————————
        global_masks = self.mask_generator.generate(image_np)

        # Prepare SAM predictor once
        self.sam_predictor.set_image(image_np)

        refined_masks = []
        for box in boxes:
            # ——— Stage 2: Box‐prompt multimask ——————————————
            box_arr = np.array(box, dtype=float)[None, :]
            masks_box, scores_box, _ = self.sam_predictor.predict(
                box=box_arr,
                multimask_output=True
            )
            # pick the highest‐scoring box mask
            best_idx = int(np.argmax(scores_box))
            mask_box = masks_box[best_idx]

            # optionally compare to global: take whichever has better coverage
            # (here we just use the box mask as our “initial” mask)
            initial_mask = mask_box

            # ——— Stage 3: Point‐prompt refinement ————————————
            final_mask = self._refine_mask_with_point(image_np, initial_mask)

            # Post‐process (morphology, blur, etc.) if you already do that:
            # e.g., close → open → blur → binarize
            kernel = np.ones((7, 7), np.uint8)
            m_uint8 = (final_mask.astype(np.uint8) * 255)
            m_closed  = cv2.morphologyEx(m_uint8, cv2.MORPH_CLOSE,  kernel)
            m_opened  = cv2.morphologyEx(m_closed,  cv2.MORPH_OPEN,   kernel)
            m_blurred = cv2.GaussianBlur(m_opened, (7, 7), 0)
            mask_refined = (m_blurred > 127)

            refined_masks.append({
                "segmentation": mask_refined,
                "bbox":         box,
                # you can store score if you like, e.g. scores_box[best_idx]
                "predicted_iou": float(scores_box[best_idx])
            })

        return refined_masks



    ###########################################################################
    # PROCESS A SINGLE IMAGE
    ###########################################################################
    def process_single_image(self, image_pil: Image.Image, image_name: str):
        print(f"\n[PROCESS] {image_name}")
        start_t = time.time()

        # 1) Run all detectors
        all_detections = []
        if "owlvit" in self.detectors_to_use:
            owl_queries = [
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
                "printer","monitor","picture", "pillow","stone","glasses","wheel","building","bridge","tomato"
            ]
            all_detections.extend(self._run_owlvit_detection(image_pil, owl_queries))
        if "yolov8" in self.detectors_to_use:
            all_detections.extend(self._run_yolov8_detection(image_pil))
        if "detectron2" in self.detectors_to_use:
            all_detections.extend(self._run_detectron2_detection(image_pil))

        # If nothing found, just save original and return
        if not all_detections:
            print("[INFO] Nessuna detection trovata: salvo immagine originale.")
            out_path = os.path.join(self.output_folder, f"{image_name}_output.jpg")
            image_pil.save(out_path)
            return

        # Unzip detections
        boxes  = [d["box"]   for d in all_detections]
        labels = [d["label"] for d in all_detections]
        scores = [d["score"] for d in all_detections]

        # ————————————————
        # 2) QUESTION-DRIVEN FILTERING
        # ————————————————
        # First infer all possible relations over the unfiltered set
        rels_all = self._infer_relationships_improved(boxes)

        if self.apply_question_filter and self._parsed_question_object_terms:
            # 2a) Find the indices of detections whose label matches the question
            question_idxs = {
                i for i, lab in enumerate(labels)
                if lab.lower() in self._parsed_question_object_terms
            }

            if question_idxs:
                # 2b) Gather direct neighbours of those question-objects
                neighbor_idxs = set()
                for r in rels_all:
                    if r["src_idx"] in question_idxs:
                        neighbor_idxs.add(r["tgt_idx"])
                    if r["tgt_idx"] in question_idxs:
                        neighbor_idxs.add(r["src_idx"])

                # 2c) Our filtered objects are only the question ones + their immediate neighbours
                filtered_idxs = sorted(question_idxs | neighbor_idxs)

                # 2d) And the only relations we keep are those directly connecting a question object ↔ neighbour
                filtered_rels = [
                    {
                        "src_idx": filtered_idxs.index(r["src_idx"]),
                        "tgt_idx": filtered_idxs.index(r["tgt_idx"]),
                        "relation": r["relation"],
                        "distance": r["distance"],
                    }
                    for r in rels_all
                    if (r["src_idx"] in question_idxs and r["tgt_idx"] in neighbor_idxs)
                    or (r["tgt_idx"] in question_idxs and r["src_idx"] in neighbor_idxs)
                ]
                filtered_rels = self._limit_relationships_per_object(filtered_rels, boxes)
                # filtered_rels = sorted(filtered_rels, key=lambda r: r["distance"])[: self.max_relations]

            else:
                # nothing matched the question → fallback to no question-filtering
                print("[INFO] Matched terms but no seeds—falling back to full set.")
                filtered_idxs = list(range(len(boxes)))
                filtered_rels = []
        else:
            # question-filter disabled or no question terms → full set
            filtered_idxs = list(range(len(boxes)))
            filtered_rels = []

        # rebuild our detection lists to only include filtered objects
        boxes  = [boxes[i]   for i in filtered_idxs]
        labels = [labels[i]  for i in filtered_idxs]
        scores = [scores[i]  for i in filtered_idxs]

        # 7) Label-wise NMS
        by_label = defaultdict(list)
        for i, lab in enumerate(labels):
            by_label[lab.lower()].append(i)

        keep = []
        for idxs in by_label.values():
            lbs = [boxes[i] for i in idxs]
            scs = [scores[i] for i in idxs]
            kept = self._non_maximum_suppression(lbs, scs, self.label_nms_threshold)
            keep.extend(idxs[k] for k in kept)
        keep.sort()

        boxes  = [boxes[i]   for i in keep]
        labels = [labels[i]  for i in keep]
        scores = [scores[i]  for i in keep]

        image_np = np.array(image_pil)
        auto_masks = self.mask_generator.generate(image_np)

        # 8) SAM segmentation refinement
        refined_masks = self._run_sam_combined_segmentation(image_pil, boxes)

        # now align and fuse per-box:
        fused_masks = []
        for det, sam_dict in zip(all_detections, refined_masks):
            sam_mask = sam_dict["segmentation"]  # H×W bool
            det2_mask = det.get("det2_mask")     # might be None for non-Detectron2
            if det2_mask is not None:
                fused = self._fuse_masks(
                    sam_mask,
                    det2_mask,
                    method="iou_union",
                    iou_thresh=0.5
                )
                sam_dict["segmentation"] = fused
            fused_masks.append(sam_dict)

        all_masks_global = auto_masks + fused_masks

        #boxes, labels, scores, kept_masks = self._filter_segmentation_duplicates(
        #      boxes, labels, scores, all_masks_global
        #)

        # 9) Use our already‐computed filtered_rels (or fallback to full set if empty)
        rels = filtered_rels if filtered_rels else [
            {
                "src_idx": r["src_idx"],
                "tgt_idx": r["tgt_idx"],
                "relation": r["relation"],
                "distance": r["distance"],
            }
            for r in rels_all
            if r["src_idx"] in filtered_idxs and r["tgt_idx"] in filtered_idxs
        ]


        # —————————————————————————————————————————
        # 10) Enforce global cap on number of relations
        #    (and optionally re-apply per-object cap)
        # first, limit per object if you want:
        rels = self._limit_relationships_per_object(rels, boxes)
        # then, keep only the closest max_relations overall:
        #rels = sorted(rels, key=lambda r: r["distance"])[: self.max_relations]
        # —————————————————————————————————————————

        # 12) Visualize & save
        out_path = os.path.join(self.output_folder, f"{image_name}_output.jpg")
        self._visualize_detections_and_relationships_with_auto_masks(
            image=image_pil,
            boxes=boxes,
            labels=labels,
            scores=scores,
            relationships=rels,
            all_masks=all_masks_global,
            save_path=out_path
        )

        print(f"[DONE] {image_name} processed in {time.time()-start_t:.2f}s")



    def run(self):
        # JSON mode
        jf = self.config.get("json_file","")
        if jf:
            with open(jf, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if self.num_instances>0:
                rows = rows[: self.num_instances]
            for row in rows:
                img_p = row["image_path"]
                q = row.get("question","") if self.apply_question_filter else ""
                try:
                    img = Image.open(img_p).convert("RGB")
                except Exception as e:
                    print(f"[ERROR] Loading {img_p}: {e}")
                    continue
                name = os.path.splitext(os.path.basename(img_p))[0]
                self.process_single_image(img, name, q)
            return

        if self.dataset_name:
            if not HAVE_DATASETS:
                print("[ERROR] 'datasets' library not installed, cannot load a dataset.")
                return

            print(f"[INFO] Loading dataset '{self.dataset_name}' (split='{self.dataset_split}')...")
            dataset = load_dataset(self.dataset_name, split=self.dataset_split)
            print(f"[INFO] Dataset loaded with {len(dataset)} items")

            start_idx = 0 if self.start_index < 0 else self.start_index
            end_idx = len(dataset) - 1 if self.end_index < 0 else self.end_index

            if self.num_instances > 0:
                computed_end = start_idx + self.num_instances - 1
                end_idx = computed_end if self.end_index < 0 else min(end_idx, computed_end)
            end_idx = min(end_idx, len(dataset) - 1)

            print(f"[INFO] Will process dataset items from index {start_idx} to {end_idx}")

            for i in range(start_idx, end_idx + 1):
                if i >= len(dataset):
                    break
                example = dataset[i]
                if self.image_column not in example:
                    print(f"[ERROR] image_column='{self.image_column}' not found. Skipping.")
                    print("[INFO] Available columns:", list(example.keys()))
                    continue

                img_data = example[self.image_column]
                # Convert to PIL
                if isinstance(img_data, Image.Image):
                    img_pil = img_data
                elif isinstance(img_data, dict) and 'bytes' in img_data:
                    from io import BytesIO
                    img_pil = Image.open(BytesIO(img_data['bytes'])).convert("RGB")
                elif isinstance(img_data, np.ndarray):
                    img_pil = Image.fromarray(img_data).convert("RGB")
                else:
                    print(f"[WARNING] Index {i}: image not recognized. Skipping.")
                    continue

                image_name = str(example.get('id', f"img_{i}"))
                self.process_single_image(img_pil, image_name)
        else:
            if not self.input_path:
                print("[ERROR] No input_path provided and no dataset given.")
                return

            if os.path.isdir(self.input_path):
                valid_exts = (".jpg", ".jpeg", ".png")
                all_files = [
                    f for f in os.listdir(self.input_path)
                    if f.lower().endswith(valid_exts)
                ]
                image_paths = [os.path.join(self.input_path, f) for f in all_files]
            else:
                if not os.path.exists(self.input_path):
                    print(f"[ERROR] Input path '{self.input_path}' does not exist.")
                    return
                image_paths = [self.input_path]

            if not image_paths:
                print("[ERROR] No images found in the specified path.")
                return

            for ipath in image_paths:
                try:
                    img_pil = Image.open(ipath).convert("RGB")
                    base_name = os.path.splitext(os.path.basename(ipath))[0]
                    self.process_single_image(img_pil, base_name)
                except Exception as e:
                    print(f"[ERROR] Could not process '{ipath}': {e}")



def parse_preproc_args():
        parser = argparse.ArgumentParser(description="Image Graph Preprocessor (dict-based)")

        # I/O
        parser.add_argument("--input_path",    type=str, default=None,
                            help="Path to image or folder")
        parser.add_argument("--json_file",     type=str, default="",
                            help="Se non vuoto, processa batch JSON")
        parser.add_argument("--output_folder", type=str, default="output_images")
        parser.add_argument("--dataset",       type=str, default=None)
        parser.add_argument("--split",         type=str, default="train")
        parser.add_argument("--image_column",  type=str, default="image")

        # Batch limit
        parser.add_argument("--num_instances", type=int, default=-1,
                            help="Se >0, process only le prime N istanze")

        # Question filtering
        parser.add_argument("--question",                 type=str, default="")
        parser.add_argument("--disable_question_filter", action="store_true")

        # Detectors & relazioni
        parser.add_argument("--detectors",     type=str,
                            default="owlvit,yolov8,detectron2",
                            help="Lista separata da virgola")
        parser.add_argument("--max_relations", type=int, default=10)
        parser.add_argument(
            "--max_relations_per_object",
            type=int,
            default=1,
            help="Maximum number of relations to keep for each object"
        )
        parser.add_argument(
            "--min_relations_per_object",
            type=int,
            default=1,
            help="Minimum number of relations to guarantee for each object"
        )

        # Soglie detectors
        parser.add_argument("--owl_threshold",       type=float, default=0.5)
        parser.add_argument("--yolo_threshold",      type=float, default=0.5)
        parser.add_argument("--detectron_threshold", type=float, default=0.5)
        parser.add_argument("--fill_segmentation",   action="store_true")
        parser.add_argument(
            "--label_mode",
            type=str,
            choices=["original","numeric","alphabetic"],
            default="original",
            help="Label style: original names, numeric indexes, or alphabetic"
        )
        parser.add_argument("--display_labels", action="store_true")
        parser.add_argument("--display_relationships", action="store_true")
        parser.add_argument("--show_segmentation", action="store_true")

        # NMS e segmentazione
        parser.add_argument("--label_nms_threshold", type=float, default=0.5)
        parser.add_argument("--seg_iou_threshold",   type=float, default=0.5)

        # Relazioni geometriche
        parser.add_argument("--overlap_thresh", type=float, default=0.3)
        parser.add_argument("--margin",         type=int,   default=20)
        parser.add_argument("--min_distance",   type=float, default=90)
        parser.add_argument("--max_distance",   type=float, default=20000)

        # SAM
        parser.add_argument("--points_per_side",        type=int,   default=32)
        parser.add_argument("--pred_iou_thresh",        type=float, default=0.7)
        parser.add_argument("--stability_score_thresh", type=float, default=0.85)
        parser.add_argument("--min_mask_region_area",   type=int,   default=300)

        # Dispositivo
        parser.add_argument("--preproc_device", type=str, default=None)

        return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_preproc_args()

    # Mappatura su config dict
    config = {
        "input_path":         args.input_path,
        "json_file":          args.json_file,
        "output_folder":      args.output_folder,
        "dataset":            args.dataset,
        "split":              args.split,
        "image_column":       args.image_column,
        "num_instances":      args.num_instances,
        "question":           args.question,
        "apply_question_filter": not args.disable_question_filter,
        "detectors_to_use":   [d.strip() for d in args.detectors.split(",")],
        "max_relations": args.max_relations,
        "max_relations_per_object": args.max_relations_per_object,
        "min_relations_per_object": args.min_relations_per_object,
        "threshold_owl":      args.owl_threshold,
        "threshold_yolo":     args.yolo_threshold,
        "threshold_detectron":args.detectron_threshold,
        "label_mode":         args.label_mode,
        "display_labels":     args.display_labels,
        "display_relationships": args.display_relationships,
        "show_segmentation":  args.show_segmentation,
        "fill_segmentation":  args.fill_segmentation,
        "label_nms_threshold":args.label_nms_threshold,
        "seg_iou_threshold":  args.seg_iou_threshold,
        "overlap_thresh":     args.overlap_thresh,
        "margin":             args.margin,
        "min_distance":       args.min_distance,
        "max_distance":       args.max_distance,
        "points_per_side":    args.points_per_side,
        "pred_iou_thresh":    args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "min_mask_region_area":   args.min_mask_region_area,
        "preproc_device":     args.preproc_device,
    }

    preproc = ImageGraphPreprocessor(config)
    preproc.run()
