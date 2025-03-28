import os
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import spacy
from spacy.matcher import PhraseMatcher
import cv2
import numpy as np
import nltk
from nltk.corpus import wordnet
import string

from collections import defaultdict
from ultralytics import YOLO


import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# -----------------------------------------------------------------------------
# 1. spaCy-based extraction of query terms
# -----------------------------------------------------------------------------
nlp = spacy.load("en_core_web_md")

def get_wordnet_synonyms(term):
    """Returns a set of potential synonyms from WordNet for a given single word."""
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            lemma_text = lemma.name().lower().replace("_", " ")
            synonyms.add(lemma_text)
    return synonyms

def extract_query_terms(question):
    """Extracts candidate object query terms from a question by lemmatizing and filtering."""
    doc = nlp(question)
    ignore = {"type", "types"}
    candidates = [
        token.lemma_.lower() for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.lemma_.lower() not in ignore
    ]
    seen = set()
    result = []
    for token in candidates:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result

# -----------------------------------------------------------------------------
# 2. spaCy-based extraction of spatial relation terms
# -----------------------------------------------------------------------------
def build_relation_mapping():
    base_map = {
        "above":   ["above"],
        "below":   ["below"],
        "left_of": ["left"],
        "right_of":["right"]
    }
    manual_expansions = {
        "above":   ["on top of"],
        "below":   ["beneath", "under"],
        "left_of": ["to the left of"],
        "right_of":["to the right of"]
    }
    relation_mapping = {}
    for relation_label, base_words in base_map.items():
        synonyms = set()
        for bw in base_words:
            synonyms.update(get_wordnet_synonyms(bw))
        synonyms.update(manual_expansions.get(relation_label, []))
        relation_mapping[relation_label] = sorted(synonyms)
    return relation_mapping

def extract_relation_terms(question):
    """Uses spaCy’s PhraseMatcher to extract spatial relation terms from the question."""
    from spacy.matcher import PhraseMatcher  # ensure import is here
    relation_mapping = build_relation_mapping()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for relation_label, phrase_list in relation_mapping.items():
        patterns = [nlp(phrase) for phrase in phrase_list]
        matcher.add(relation_label.upper(), patterns)
    doc = nlp(question)
    matches = matcher(doc)
    desired = set()
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        relation_label = rule_id.lower()
        desired.add(relation_label)
    return list(desired)

# -----------------------------------------------------------------------------
# 3. Non-Maximum Suppression (NMS) Utilities
# -----------------------------------------------------------------------------
def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) of two boxes
    in [x_min, y_min, x_max, y_max] format.
    """
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
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xB - xA) * (yB - yA)
    return inter_area / float(box1_area + box2_area - inter_area)

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Standard NMS:
    - boxes: list of [x_min, y_min, x_max, y_max]
    - scores: list of confidence scores
    - iou_threshold: overlap threshold
    Returns indices of the boxes to keep.
    """
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        curr = idxs.pop(0)
        keep.append(curr)
        remain = []
        for i in idxs:
            if compute_iou(boxes[curr], boxes[i]) < iou_threshold:
                remain.append(i)
        idxs = remain
    return keep

# -----------------------------------------------------------------------------
# 4. Segmentation-related Functions (SAM and Duplicate Filtering)
# -----------------------------------------------------------------------------
def get_best_mask_for_box(detection_box, masks, iou_threshold=0.3):
    """
    Among the generated SAM masks, pick the one that best matches the bounding box
    by IoU (>= iou_threshold). On ties, pick the one with higher predicted_iou.
    """
    best_mask = None
    best_iou = 0.0
    best_score = -1
    for m in masks:
        x, y, w, h = m['bbox']
        mask_box = [x, y, x + w, y + h]
        iou = compute_iou(detection_box, mask_box)
        if iou > best_iou:
            best_iou = iou
            best_score = m.get('predicted_iou', 0)
            best_mask = m
        elif iou == best_iou and iou >= iou_threshold:
            score = m.get('predicted_iou', 0)
            if score > best_score:
                best_mask = m
                best_score = score
    if best_iou >= iou_threshold:
        return best_mask
    else:
        return None

def bbox_from_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]

def filter_segmentation_duplicates(boxes, labels, scores, all_masks, iou_threshold=0.8):
    """
    1) For each bounding box, find best SAM mask -> refine bounding box from the mask contour.
    2) Then do an NMS pass (with iou_threshold).
    3) Return the filtered set of boxes, labels, scores.
    """
    seg_boxes = []
    for box in boxes:
        mask = get_best_mask_for_box(box, all_masks)
        if mask is not None:
            mask_uint8 = (mask['segmentation'] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                refined_box = bbox_from_contour(largest_contour)
            else:
                refined_box = box
        else:
            refined_box = box
        seg_boxes.append(refined_box)

    keep_idx = non_maximum_suppression(seg_boxes, scores, iou_threshold)
    new_boxes  = [boxes[i]  for i in keep_idx]
    new_labels = [labels[i] for i in keep_idx]
    new_scores = [scores[i] for i in keep_idx]
    return new_boxes, new_labels, new_scores

# -----------------------------------------------------------------------------
# 5. Relationship Inference Functions
# -----------------------------------------------------------------------------
def compute_horizontal_overlap(b1, b2):
    """
    For checking vertical relationships, compute fraction of horizontal overlap
    relative to the min box width.
    """
    x1, y1, x2, y2 = b1
    xA, yA, xB, yB = b2
    overlap_min_x = max(x1, xA)
    overlap_max_x = min(x2, xB)
    overlap_width = max(0, overlap_max_x - overlap_min_x)
    w1 = x2 - x1
    w2 = xB - xA
    min_width = min(w1, w2)
    if min_width <= 0:
        return 0.0
    return overlap_width / float(min_width)

def infer_relationships_improved(
    boxes,
    labels,
    overlap_thresh=0.3,  # fraction of horizontal overlap to say "above"/"below"
    margin=20,           # margin in px for deciding "above"/"left_of", etc.
    min_distance=50,     # min distance between centers
    max_distance=20000,  # max distance between centers
    top_k=None           # limit total relationships by ascending distance
):
    n = len(boxes)
    centers = []
    for (x_min, y_min, x_max, y_max) in boxes:
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        centers.append((cx, cy))

    relationships = []
    seen_pairs = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (x1, y1) = centers[i]
            (x2, y2) = centers[j]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist < min_distance or dist > max_distance:
                continue
            overlap = compute_horizontal_overlap(boxes[i], boxes[j])
            # If horizontally overlapped, check above/below
            if overlap >= overlap_thresh:
                if y1 + margin < y2:
                    relation = "above"
                    cat = "vertical"
                elif y1 - margin > y2:
                    relation = "below"
                    cat = "vertical"
                else:
                    continue
            else:
                # else check left_of / right_of
                if x1 + margin < x2:
                    relation = "left_of"
                    cat = "horizontal"
                elif x1 - margin > x2:
                    relation = "right_of"
                    cat = "horizontal"
                else:
                    continue

            key = (min(i, j), max(i, j), cat)
            if key in seen_pairs:
                continue
            relationships.append({
                'src_idx': i,
                'tgt_idx': j,
                'relation': relation,
                'distance': dist
            })
            seen_pairs[key] = True

    if top_k is not None and len(relationships) > top_k:
        relationships = sorted(relationships, key=lambda r: r['distance'])[:top_k]
    return relationships

# -----------------------------------------------------------------------------
# 6. Visualization Helper Functions
# -----------------------------------------------------------------------------
def move_point_outside_contour(candidate, contour, step=5, max_iter=20):
    """
    Nudges the candidate point outward from a contour so that text is not inside the mask region.
    """
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

def adjust_position(candidate, placed_positions, overlap_thresh, max_iterations=10):
    """
    Avoid overlapping text labels by iteratively pushing candidate
    away from already placed text positions.
    """
    new_candidate = np.array(candidate, dtype=float)
    epsilon = 1e-6
    for _ in range(max_iterations):
        displacement = np.zeros(2, dtype=float)
        for p in placed_positions:
            diff = new_candidate - np.array(p, dtype=float)
            dist = np.linalg.norm(diff)
            if dist < overlap_thresh:
                push = (overlap_thresh - dist) * (diff / (dist + epsilon))
                displacement += push
        if np.linalg.norm(displacement) < 1e-3:
            break
        new_candidate += displacement
    return tuple(new_candidate)

# -----------------------------------------------------------------------------
# 7. Visualization Function
# -----------------------------------------------------------------------------
def visualize_detections_and_relationships_with_auto_masks(
    image,
    boxes,
    labels,
    scores,
    relationships,
    all_masks,
    view_relations_labels=False,
    label_mode="original",  # "original", "numeric", "alphabetic"
    show_confidence=True,
    draw_relationships=True,
    display_labels=True,
    save_path=None
):
    """
    Draw detections (SAM masks or bounding boxes),
    optionally draw relationships (arrows),
    and annotate with labels/scores.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")

    placed_label_positions = []
    overlap_threshold = 30
    color_list = ['red', 'green', 'blue', 'magenta', 'cyan', 'orange', 'purple', 'brown']
    obj_colors = [color_list[i % len(color_list)] for i in range(len(boxes))]

    # For label text
    if label_mode == "numeric":
        vis_labels = [str(i+1) for i in range(len(boxes))]
    elif label_mode == "alphabetic":
        vis_labels = list(string.ascii_uppercase[:len(boxes)])
    else:
        # "original"
        vis_labels = labels

    def place_label_neatly(candidate):
        return adjust_position(candidate, placed_label_positions, overlap_threshold)

    # Draw detections
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        color = obj_colors[i]
        best_mask = get_best_mask_for_box(box, all_masks)
        if best_mask is not None:
            mask_uint8 = (best_mask['segmentation'] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour = largest_contour.squeeze()
                if largest_contour.ndim >= 2:
                    ax.plot(largest_contour[:, 0], largest_contour[:, 1],
                            color=color, linewidth=2, zorder=2)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        candidate_pt = (cx, cy)
                    else:
                        candidate_pt = ((x_min + x_max)//2, (y_min + y_max)//2)
                    candidate_pt = move_point_outside_contour(candidate_pt, largest_contour)
                else:
                    candidate_pt = ((x_min + x_max)//2, (y_min + y_max)//2)
            else:
                candidate_pt = ((x_min + x_max)//2, (y_min + y_max)//2)
        else:
            # If no mask, just draw bounding box
            candidate_pt = ((x_min + x_max)//2, (y_min + y_max)//2)
            rect = patches.Rectangle((x_min, y_min),
                                     x_max - x_min,
                                     y_max - y_min,
                                     linewidth=2,
                                     edgecolor=color,
                                     facecolor='none',
                                     zorder=2)
            ax.add_patch(rect)

        candidate_pt = place_label_neatly(candidate_pt)
        placed_label_positions.append(candidate_pt)

        if display_labels:
            if show_confidence:
                text = f"{vis_labels[i]}: {scores[i]:.2f}"
            else:
                text = vis_labels[i]
            ax.text(candidate_pt[0], candidate_pt[1], text,
                    fontsize=10, color=color,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    zorder=3)

    # Optionally draw relationships
    centers = []
    for b in boxes:
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        centers.append((cx, cy))

    if draw_relationships:
        for rel in relationships:
            src_idx = rel['src_idx']
            tgt_idx = rel['tgt_idx']
            rel_label = rel['relation']
            src_color = obj_colors[src_idx]
            start_pt = centers[src_idx]
            end_pt   = centers[tgt_idx]

            arrow = patches.FancyArrowPatch(start_pt, end_pt,
                                            arrowstyle='->',
                                            color=src_color,
                                            linewidth=1.5,
                                            connectionstyle='arc3,rad=0.2',
                                            mutation_scale=12,
                                            zorder=4)
            ax.add_patch(arrow)

            mid_x = (start_pt[0] + end_pt[0]) / 2
            mid_y = (start_pt[1] + end_pt[1]) / 2
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            angle_deg = math.degrees(math.atan2(dy, dx))
            if angle_deg < -90:
                angle_deg += 180
            elif angle_deg > 90:
                angle_deg -= 180
            mag = math.sqrt(dx**2 + dy**2)
            if mag > 0:
                base_offset = 10
                offset_x = -dy / mag * base_offset
                offset_y = dx / mag * base_offset
            else:
                offset_x, offset_y = 0, 0

            candidate = (mid_x + offset_x, mid_y + offset_y)
            candidate = place_label_neatly(candidate)
            placed_label_positions.append(candidate)

            if view_relations_labels:
                ax.text(candidate[0], candidate[1], rel_label,
                        fontsize=8, color=src_color,
                        rotation=angle_deg,
                        rotation_mode='anchor',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                        zorder=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# -----------------------------------------------------------------------------
# 8. Detection Routines: OWL-ViT, YOLOv8, Detectron2
# -----------------------------------------------------------------------------
def load_owlvit_detector(device="cuda"):
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
    model.to(device)
    model.eval()
    return processor, model

@torch.inference_mode()
def run_owlvit_detection(image_pil, queries, processor, model, threshold=0.1, device="cuda"):
    width, height = image_pil.size
    inputs = processor(text=queries, images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.Tensor([[height, width]]).to(device)
    results_list = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes
    )
    if not results_list or results_list[0] is None:
        return []

    results = results_list[0]
    if any(results.get(k) is None for k in ["boxes", "scores", "labels"]):
        return []

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    detections = []
    for box, score, label_idx in zip(boxes, scores, labels):
        if score >= threshold:
            box_list = box.tolist()
            label_name = queries[label_idx]
            detections.append({
                "box": box_list,
                "label": label_name,
                "score": float(score)
            })

    del outputs, results_list, results, boxes, scores, labels
    torch.cuda.empty_cache()
    return detections

def load_yolov8_detector(model_path="yolov8x.pt", device="cuda"):
    if YOLO is None:
        raise ImportError("ultralytics is not installed or could not be imported.")
    model = YOLO(model_path)
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model

@torch.inference_mode()
def run_yolov8_detection(image_pil, yolo_model, threshold=0.1, device="cuda"):
    if yolo_model is None:
        return []

    image_np = np.array(image_pil)
    results = yolo_model.predict(image_np, device=device)[0]
    detections = []
    for box_xyxy, conf, cls_idx in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        score_val = float(conf.item())
        if score_val < threshold:
            continue
        label_idx = int(cls_idx.item())
        label_name = results.names[label_idx]
        x_min, y_min, x_max, y_max = box_xyxy.tolist()
        detections.append({
            "box": [x_min, y_min, x_max, y_max],
            "label": label_name,
            "score": score_val
        })
    del results
    torch.cuda.empty_cache()
    return detections

def load_detectron2_detector(model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                             score_thresh=0.1,
                             device="cuda"):
    if detectron2 is None:
        raise ImportError("Detectron2 is not installed or could not be imported.")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    return predictor, metadata

@torch.inference_mode()
def run_detectron2_detection(image_pil, detectron2_predictor, detectron2_metadata):
    image_np = np.array(image_pil)
    outputs = detectron2_predictor(image_np)
    instances = outputs["instances"].to("cpu")

    boxes   = instances.pred_boxes   if instances.has("pred_boxes") else None
    scores  = instances.scores       if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None

    detections = []
    if boxes is not None and scores is not None and classes is not None:
        for box_xyxy, score_val, cls_idx in zip(boxes, scores, classes):
            x_min, y_min, x_max, y_max = box_xyxy.tolist()
            label_name = detectron2_metadata.thing_classes[cls_idx]
            detections.append({
                "box": [x_min, y_min, x_max, y_max],
                "label": label_name,
                "score": float(score_val)
            })
    del outputs, instances, boxes, scores, classes
    torch.cuda.empty_cache()
    return detections

# -----------------------------------------------------------------------------
# 9. Main function: Single Image OR Folder
# -----------------------------------------------------------------------------
def main():
    input_path = "/content/gqa_subset_images"  # can be single image OR a folder
    output_folder = "/content/GQA_seg"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    DETECTORS_TO_USE = ["owlvit", "yolov8", "detectron2"]

    # Detector thresholds
    THRESH_OWL = 0.15
    THRESH_YOLO = 0.3
    THRESH_DETECTRON = 0.3

    # Relationship inference limit
    MAX_RELATIONS = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    question = (
        "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, "
        "traffic light, fire hydrant, sign, bench, bird, cat, "
        "dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, "
        "handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, "
        "baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, "
        "knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, "
        "donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, "
        "remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, "
        "vase, scissors, teddy bear, hair drier, toothbrush, wood, fence, lamp, weel, tree, building, street lamp, flower, "
        "shoe, hand, bridge, mountain, tomato, plant, flag, headphones, hat, shirt, plate, road sign, river, helmet, stone, "
        "monument, sea, doll, candle, coach, wardrobe, bread, cloud"
    )
    query_terms = extract_query_terms(question)
    desired_relation_labels = extract_relation_terms(question)
    print("Extracted query terms:", query_terms)
    print("Desired relation labels:", desired_relation_labels)
    if not query_terms:
        query_terms = ["chair"]

    coco_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "sign", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush", "wood", "fence", "lamp", "weel", "tree", "building", "street lamp",
        "flower", "shoe", "hand", "bridge", "mountain", "tomato", "plant", "flag", "headphones", "hat",
        "shirt", "plate", "road sign", "river", "helmet", "stone", "monument", "sea", "doll", "candle",
        "coach", "wardrobe", "bread", "cloud"
    ]

    # ----------------- Load detectors -----------------
    owlvit_processor = None
    owlvit_model = None
    yolo_model = None
    d2_predictor = None
    d2_metadata = None

    if "owlvit" in DETECTORS_TO_USE:
        print("Loading OWL‑ViT Detector...")
        owlvit_processor, owlvit_model = load_owlvit_detector(device=device)

    if "yolov8" in DETECTORS_TO_USE:
        print("Loading YOLOv8x Detector...")
        yolo_model = load_yolov8_detector("yolov8x.pt", device=device)

    if "detectron2" in DETECTORS_TO_USE:
        print("Loading Detectron2 Detector...")
        d2_predictor, d2_metadata = load_detectron2_detector(
            model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            score_thresh=THRESH_DETECTRON,
            device=device
        )

    # ----------------- Set up the SAM model -----------------
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.95,
        min_mask_region_area=100
    )

    # -----------------------------------------------------------------------------
    # Check if input_path is a directory or a single file
    # -----------------------------------------------------------------------------
    if os.path.isdir(input_path):
        image_files = [
            f for f in os.listdir(input_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        image_paths = [os.path.join(input_path, f) for f in image_files]
    else:
        if not os.path.exists(input_path):
            print(f"Error: {input_path} does not exist.")
            return
        image_paths = [input_path]

    # -----------------------------------------------------------------------------
    # Process each image path
    # -----------------------------------------------------------------------------
    for image_path in image_paths:
        print("\nProcessing:", image_path)
        image_pil = Image.open(image_path).convert("RGB")

        all_detections = []

        # 1) OWL‑ViT
        if "owlvit" in DETECTORS_TO_USE and owlvit_processor and owlvit_model:
            owl_dets = run_owlvit_detection(
                image_pil=image_pil,
                queries=coco_labels,
                processor=owlvit_processor,
                model=owlvit_model,
                threshold=THRESH_OWL,
                device=device
            )
            all_detections.extend(owl_dets)

        # 2) YOLOv8
        if "yolov8" in DETECTORS_TO_USE and yolo_model:
            yolo_dets = run_yolov8_detection(
                image_pil=image_pil,
                yolo_model=yolo_model,
                threshold=THRESH_YOLO,
                device=device
            )
            all_detections.extend(yolo_dets)

        # 3) Detectron2
        if "detectron2" in DETECTORS_TO_USE and d2_predictor and d2_metadata:
            d2_dets = run_detectron2_detection(
                image_pil=image_pil,
                detectron2_predictor=d2_predictor,
                detectron2_metadata=d2_metadata
            )
            all_detections.extend(d2_dets)

        if not all_detections:
            print("No detections found. Skipping.")
            torch.cuda.empty_cache()
            continue

        # Combine & run label-based NMS
        detections_by_label = defaultdict(list)
        for i, det in enumerate(all_detections):
            label = det['label']
            detections_by_label[label].append(i)

        final_indices = []
        label_nms_threshold = 0.5
        for label, idx_list in detections_by_label.items():
            label_boxes  = [all_detections[i]['box']   for i in idx_list]
            label_scores = [all_detections[i]['score'] for i in idx_list]
            keep_for_label = non_maximum_suppression(label_boxes, label_scores, label_nms_threshold)
            for k in keep_for_label:
                final_indices.append(idx_list[k])
        final_indices = sorted(final_indices)

        detected_boxes  = [all_detections[i]['box']   for i in final_indices]
        detected_labels = [all_detections[i]['label'] for i in final_indices]
        detected_scores = [all_detections[i]['score'] for i in final_indices]

        # Generate SAM masks
        with torch.inference_mode():
            image_np = np.array(image_pil)
            all_masks = mask_generator.generate(image_np)

        # Filter duplicates with segmentation
        detected_boxes, detected_labels, detected_scores = filter_segmentation_duplicates(
            detected_boxes,
            detected_labels,
            detected_scores,
            all_masks,
            iou_threshold=0.8
        )

        # Relationship inference
        final_relationships = infer_relationships_improved(
            boxes=detected_boxes,
            labels=detected_labels,
            overlap_thresh=0.3,
            margin=20,
            min_distance=90,
            max_distance=20000,
            top_k=MAX_RELATIONS
        )
        # Possibly filter relationships to only the ones we extracted
        desired_relation_labels = [
            r for r in extract_relation_terms(question)
        ]
        if desired_relation_labels:
            final_relationships = [
                r for r in final_relationships
                if r['relation'] in desired_relation_labels
            ]
            print("Filtered relationships:", final_relationships)

        # Visualization & saving
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = base_name + "_output.jpg"
        output_path = os.path.join(output_folder, output_filename)

        visualize_detections_and_relationships_with_auto_masks(
            image=image_pil,
            boxes=detected_boxes,
            labels=detected_labels,
            scores=detected_scores,
            relationships=final_relationships,
            all_masks=all_masks,
            view_relations_labels=False,
            label_mode="numeric",
            show_confidence=False,
            draw_relationships=False,
            display_labels=False,
            save_path=output_path
        )
        print("Saved output to:", output_path)

        # Clean up
        del image_pil, image_np, all_detections
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
