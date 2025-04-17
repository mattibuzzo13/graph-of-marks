#!/usr/bin/env python

import os
import sys
import math
import time
import argparse

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
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# For optional Hugging Face dataset usage
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False


###############################################################################
# ARGUMENT PARSING (combined/advanced version)
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Image Graph Preprocessor")

    # Input/output arguments
    parser.add_argument("--input_path", type=str,
                        help="Path to a single image or directory of images")
    parser.add_argument("--output_folder", type=str, default="output_images",
                        help="Output folder for processed images")
    parser.add_argument("--dataset", type=str,
                        help="Hugging Face dataset name to download (e.g. 'coco', 'imagefolder', etc.)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (e.g. 'train', 'validation', 'test')")
    parser.add_argument("--image_column", type=str, default="image",
                        help="Column name containing images in the dataset")

    # Processing control arguments
    parser.add_argument("--detectors", type=str, default="owlvit,yolov8,detectron2",
                        help="Comma-separated list of detectors to use (e.g. 'owlvit,yolov8,detectron2')")
    parser.add_argument("--relationship_type", type=str, default="all",
                        help="Types of relationships to extract (all, above, below, left_of, right_of)")
    parser.add_argument("--max_relations", type=int, default=8,
                        help="Maximum number of relationships to extract")
    parser.add_argument("--start_index", type=int, default=-1,
                        help="Start index (0-based) for processing dataset items")
    parser.add_argument("--end_index", type=int, default=-1,
                        help="End index (inclusive, 0-based) for processing dataset items")
    parser.add_argument("--num_instances", type=int, default=-1,
                        help="Absolute number of dataset items to process, if > 0")

    # Detection thresholds
    parser.add_argument("--owl_threshold", type=float, default=0.15,
                        help="Confidence threshold for OWL-ViT detector")
    parser.add_argument("--yolo_threshold", type=float, default=0.3,
                        help="Confidence threshold for YOLOv8 detector")
    parser.add_argument("--detectron_threshold", type=float, default=0.3,
                        help="Confidence threshold for Detectron2 detector")

    # NMS parameters
    parser.add_argument("--label_nms_threshold", type=float, default=0.5,
                        help="IoU threshold for label-based NMS")
    parser.add_argument("--seg_iou_threshold", type=float, default=0.8,
                        help="IoU threshold for segmentation duplicate filtering")

    # Relationship inference parameters
    parser.add_argument("--overlap_thresh", type=float, default=0.3,
                        help="Horizontal overlap threshold for relationship inference")
    parser.add_argument("--margin", type=int, default=20,
                        help="Margin in pixels for relationship inference")
    parser.add_argument("--min_distance", type=int, default=90,
                        help="Minimum distance between centers for relationship inference")
    parser.add_argument("--max_distance", type=int, default=20000,
                        help="Maximum distance between centers for relationship inference")

    # SAM parameters
    parser.add_argument("--points_per_side", type=int, default=32,
                        help="Points per side for SAM mask generator")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.9,
                        help="Predicted IoU threshold for SAM mask generator")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95,
                        help="Stability score threshold for SAM mask generator")
    parser.add_argument("--min_mask_region_area", type=int, default=100,
                        help="Minimum mask region area for SAM mask generator")

    return parser.parse_args()


###############################################################################
# SPACy and NLTK Setup
###############################################################################
nlp = spacy.load("en_core_web_md")
# nltk.download("wordnet")


###############################################################################
# HELPER FUNCTIONS (Synonyms, Relation Extraction, NMS, etc.)
###############################################################################
def get_wordnet_synonyms(term: str) -> set:
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

def extract_query_terms(question: str) -> list:
    """Extract candidate object query terms (nouns/props) from question via spaCy."""
    doc = nlp(question)
    ignore = {"type", "types"}
    candidates = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            lemma = token.lemma_.lower()
            if lemma not in ignore:
                candidates.append(lemma)
    # Deduplicate
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

def build_relation_mapping() -> dict:
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
    mapping = {}
    for rel_label, base_words in base_map.items():
        synonyms = set()
        for bw in base_words:
            synonyms.update(get_wordnet_synonyms(bw))
        synonyms.update(manual_expansions.get(rel_label, []))
        mapping[rel_label] = sorted(synonyms)
    return mapping

def extract_relation_terms(question: str) -> list:
    """Extract spatial relation terms from question using spaCy PhraseMatcher."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    rel_map = build_relation_mapping()
    for rel_label, phrases in rel_map.items():
        patterns = [nlp(text) for text in phrases]
        matcher.add(rel_label.upper(), patterns)
    doc = nlp(question)
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        label_str = nlp.vocab.strings[match_id]
        found.add(label_str.lower())
    return list(found)

def compute_iou(box1: list, box2: list) -> float:
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

def non_maximum_suppression(boxes: list, scores: list, iou_threshold: float = 0.5) -> list:
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        curr = idxs.pop(0)
        keep.append(curr)
        new_list = []
        for i in idxs:
            if compute_iou(boxes[curr], boxes[i]) < iou_threshold:
                new_list.append(i)
        idxs = new_list
    return keep


###############################################################################
# SAM + Segmentation
###############################################################################
def get_best_mask_for_box(detection_box: list, masks: list, iou_threshold: float = 0.3):
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
        elif math.isclose(iou, best_iou) and iou >= iou_threshold:
            score = m.get('predicted_iou', 0)
            if score > best_score:
                best_mask = m
                best_score = score
    return best_mask if best_iou >= iou_threshold else None

def bbox_from_contour(contour: np.ndarray) -> list:
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]

def filter_segmentation_duplicates(boxes: list, labels: list, scores: list,
                                  all_masks: list, iou_threshold: float = 0.8):
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


###############################################################################
# Relationship Inference
###############################################################################
def infer_relationships_improved(
    boxes: list,
    labels: list,
    overlap_thresh: float = 0.3,
    margin: float = 20,
    min_distance: float = 90,
    max_distance: float = 20000,
    top_k: int = None
) -> list:
    """
    Infers spatial relationships (above/below/left_of/right_of) between bounding boxes
    using center positions and thresholds.
    """
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
            (cx1, cy1) = centers[i]
            (cx2, cy2) = centers[j]
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = math.hypot(dx, dy)
            if dist < min_distance or dist > max_distance:
                continue

            # If vertically distinct
            if abs(dy) >= abs(dx) and abs(dy) > margin:
                relation = "above" if dy > 0 else "below"
            # Else horizontally distinct
            elif abs(dx) > margin:
                relation = "left_of" if dx > 0 else "right_of"
            else:
                continue

            rels.append({
                "src_idx": i,
                "tgt_idx": j,
                "relation": relation,
                "distance": dist
            })

    if top_k is not None and len(rels) > top_k:
        rels = sorted(rels, key=lambda r: r['distance'])[:top_k]
    return rels

def limit_relationships_per_object(
    relationships: list,
    boxes: list,
    max_per_object: int,
    min_per_object: int = None,
    min_relation_distance: float = 80
) -> list:
    """
    Limits how many relationships an object can have as a 'source'.
    Optionally ensures each object has at least a minimum number of relationships.
    """
    rels_by_src = defaultdict(list)
    for r in relationships:
        rels_by_src[r["src_idx"]].append(r)

    centers = []
    for b in boxes:
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        centers.append((cx, cy))

    n = len(boxes)
    # If min_per_object is set, ensure each object has at least that many
    if min_per_object is not None:
        for i in range(n):
            if len(rels_by_src[i]) < min_per_object:
                best_j = None
                best_d = float('inf')
                for j in range(n):
                    if j == i:
                        continue
                    d = math.hypot(centers[i][0] - centers[j][0],
                                   centers[i][1] - centers[j][1])
                    if d < best_d and d >= min_relation_distance:
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
        final_list.extend(sorted_rlist[:max_per_object])
    return final_list

def unify_pair_relations(relationships: list) -> list:
    """
    Ensures that for any pair of objects (i, j), we only keep ONE relationship.
    If multiple relationships exist for the pair (either direction),
    we keep the one with the smaller distance.
    """
    best_for_pair = {}
    for r in relationships:
        i = r["src_idx"]
        j = r["tgt_idx"]
        pair = tuple(sorted((i, j)))  # ignore direction for dedup

        if pair not in best_for_pair:
            best_for_pair[pair] = r
        else:
            # compare distances; keep the smaller
            if r["distance"] < best_for_pair[pair]["distance"]:
                best_for_pair[pair] = r

    return list(best_for_pair.values())


###############################################################################
# Visualization
###############################################################################
def move_point_outside_contour(candidate: tuple, contour: np.ndarray,
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
        # If inside contour, push outward
        if cv2.pointPolygonTest(contour, tuple(new_candidate), False) >= 0:
            new_candidate += step * direction
        else:
            break
    return tuple(new_candidate)

def adjust_position(candidate: tuple, placed_positions: list,
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

def visualize_detections_and_relationships_with_auto_masks(
    image,
    boxes: list,
    labels: list,
    scores: list,
    relationships: list,
    all_masks: list,
    view_relations_labels: bool = False,
    label_mode: str = "original",
    show_confidence: bool = True,
    draw_relationships: bool = True,
    display_labels: bool = True,
    show_segmentation: bool = True,
    fill_segmentation: bool = False,
    save_path: str = None
):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")

    placed_positions = []
    overlap_threshold = 30
    color_list = ['red','green','blue','magenta','cyan','orange','purple','brown']
    obj_colors = [color_list[i % len(color_list)] for i in range(len(boxes))]

    # Decide label text style
    if label_mode == "numeric":
        vis_labels = [str(i + 1) for i in range(len(boxes))]
    elif label_mode == "alphabetic":
        vis_labels = list(string.ascii_uppercase[:len(boxes)])
    else:
        vis_labels = labels

    def place_label_neatly(pt):
        return adjust_position(pt, placed_positions, overlap_threshold)

    detection_labels_info = []
    arrow_counts = defaultdict(int)

    # Draw each detection
    for i, box in enumerate(boxes):
        color = obj_colors[i]
        x_min, y_min, x_max, y_max = map(int, box)
        center_pt = ((x_min + x_max)//2, (y_min + y_max)//2)

        # Check for best mask
        best_mask = get_best_mask_for_box(box, all_masks)
        if show_segmentation and best_mask is not None:
            mask_uint8 = (best_mask['segmentation'] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea).squeeze()
                if largest_contour.ndim >= 2:
                    # Optionally fill
                    if fill_segmentation:
                        polygon = largest_contour.reshape(-1, 2)
                        ax.fill(polygon[:,0], polygon[:,1],
                                color=color, alpha=0.3, zorder=1)
                    # Outline
                    ax.plot(largest_contour[:,0], largest_contour[:,1],
                            color=color, linewidth=2, zorder=2)

                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center_pt = (cx, cy)
                    # Nudge label outside the mask
                    center_pt = move_point_outside_contour(center_pt, largest_contour)
                else:
                    # Fallback bounding box if mask is degenerate
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor=color, facecolor='none', zorder=2
                    )
                    ax.add_patch(rect)
            else:
                # Fallback bounding box if no contours
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor=color, facecolor='none', zorder=2
                )
                ax.add_patch(rect)
        else:
            # Show bounding box if no segmentation
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=color, facecolor='none', zorder=2
            )
            ax.add_patch(rect)

        center_pt = place_label_neatly(center_pt)
        placed_positions.append(center_pt)

        if display_labels:
            if show_confidence:
                text_label = f"{vis_labels[i]}: {scores[i]:.2f}"
            else:
                text_label = vis_labels[i]
            detection_labels_info.append((center_pt, text_label, color))

    # Optionally draw relationships
    if draw_relationships and len(relationships) > 0:
        centers = []
        for b in boxes:
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            centers.append((cx, cy))

        for rel in relationships:
            s = rel["src_idx"]
            t = rel["tgt_idx"]
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

            if view_relations_labels:
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
                candidate = adjust_position(candidate, placed_positions, overlap_threshold)
                placed_positions.append(candidate)

                ax.text(
                    candidate[0], candidate[1],
                    rel_label,
                    fontsize=8, color=color,
                    rotation=angle_deg,
                    rotation_mode='anchor',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),  # More transparent label background
                    zorder=5
                )

    # Finally, place the text labels for objects
    for (pt, text, color) in detection_labels_info:
        ax.text(
            pt[0], pt[1], text,
            fontsize=10, color=color,
            bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'),  # More transparent label background
            zorder=7
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved output to {save_path}")
    else:
        plt.show()


###############################################################################
# Detector Loading & Inference
###############################################################################
def load_owlvit_detector(device="cuda"):
    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
    model.to(device)
    model.eval()
    return processor, model

@torch.inference_mode()
def run_owlvit_detection(image_pil: Image.Image,
                         queries: list,
                         processor,
                         model,
                         threshold: float = 0.1,
                         device="cuda") -> list:
    w, h = image_pil.size
    inputs = processor(text=queries, images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.Tensor([[h, w]]).to(device)
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
    for box, score, lab_idx in zip(boxes, scores, labels):
        if score >= threshold:
            detections.append({
                "box": box.tolist(),
                "label": queries[lab_idx],
                "score": float(score)
            })
    return detections

def load_yolov8_detector(model_path="yolov8x.pt", device="cuda"):
    model = YOLO(model_path)
    model.to(device)
    return model

@torch.inference_mode()
def run_yolov8_detection(image_pil: Image.Image,
                         yolo_model,
                         threshold: float=0.1,
                         device="cuda") -> list:
    if yolo_model is None:
        return []
    image_np = np.array(image_pil)
    results = yolo_model.predict(image_np, device=device)[0]

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

def load_detectron2_detector(model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                             score_thresh=0.1,
                             device="cuda"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    return predictor, metadata

@torch.inference_mode()
def run_detectron2_detection(image_pil: Image.Image,
                             detectron2_predictor,
                             detectron2_metadata,
                             device="cuda"):
    if detectron2_predictor is None:
        return []
    image_np = np.array(image_pil)
    outputs = detectron2_predictor(image_np)
    instances = outputs["instances"].to("cpu")

    boxes   = instances.pred_boxes
    scores  = instances.scores
    classes = instances.pred_classes
    if boxes is None or scores is None or classes is None:
        return []

    detections = []
    for box_xyxy, sc, cls_idx in zip(boxes, scores, classes):
        x_min, y_min, x_max, y_max = box_xyxy.tolist()
        label_name = detectron2_metadata.thing_classes[cls_idx]
        detections.append({
            "box": [x_min, y_min, x_max, y_max],
            "label": label_name,
            "score": float(sc)
        })
    return detections


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    args = parse_args()

    # Retrieve arguments
    input_path = args.input_path
    output_folder = args.output_folder
    dataset_name = args.dataset
    split = args.split
    image_column = args.image_column

    detectors_str = args.detectors.strip('[]')
    detectors_to_use = [d.strip() for d in detectors_str.split(',')]
    print(f"[INFO] Detectors to use: {detectors_to_use}")
    relationship_type = args.relationship_type
    max_relations = args.max_relations

    start_index = args.start_index
    end_index = args.end_index
    num_instances = args.num_instances

    owl_threshold = args.owl_threshold
    yolo_threshold = args.yolo_threshold
    detectron_threshold = args.detectron_threshold

    label_nms_threshold = args.label_nms_threshold
    seg_iou_threshold = args.seg_iou_threshold

    overlap_thresh = args.overlap_thresh
    margin = args.margin
    min_distance = args.min_distance
    max_distance = args.max_distance

    points_per_side = args.points_per_side
    pred_iou_thresh = args.pred_iou_thresh
    stability_score_thresh = args.stability_score_thresh
    min_mask_region_area = args.min_mask_region_area

    # Ensure output folder
    os.makedirs(output_folder, exist_ok=True)

    # Decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on device: {device}")

    # ----------------- Load SAM model -----------------
    print("[INFO] Loading SAM model ...")
    sam_checkpoint = "src/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area
    )

    # ----------------- Load detectors -----------------
    owlvit_processor, owlvit_model = (None, None)
    if "owlvit" in detectors_to_use:
        print("[INFO] Loading OWL-ViT ...")
        owlvit_processor, owlvit_model = load_owlvit_detector(device=device)

    yolo_model = None
    if "yolov8" in detectors_to_use:
        print("[INFO] Loading YOLOv8 ...")
        yolo_model = load_yolov8_detector("yolov8x.pt", device=device)

    d2_predictor, d2_metadata = (None, None)
    if "detectron2" in detectors_to_use:
        print("[INFO] Loading Detectron2 ...")
        d2_predictor, d2_metadata = load_detectron2_detector(
            model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            score_thresh=detectron_threshold,
            device=device
        )

    def process_single_image(image_pil: Image.Image, image_name: str):
        print(f"\n[PROCESS] {image_name}")
        start_t = time.time()

        # 1) Gather detections from chosen detectors
        all_detections = []

        # OWL-ViT
        if "owlvit" in detectors_to_use and owlvit_processor and owlvit_model:
            # For OWL-ViT, passing a default set of queries
            queries_owlvit = [
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
                "printer","monitor"
            ]
            owl_dets = run_owlvit_detection(
                image_pil=image_pil,
                queries=queries_owlvit,
                processor=owlvit_processor,
                model=owlvit_model,
                threshold=owl_threshold,
                device=device
            )
            all_detections.extend(owl_dets)

        # YOLOv8
        if "yolov8" in detectors_to_use and yolo_model:
            yolo_dets = run_yolov8_detection(
                image_pil=image_pil,
                yolo_model=yolo_model,
                threshold=yolo_threshold,
                device=device
            )
            all_detections.extend(yolo_dets)

        # Detectron2
        if "detectron2" in detectors_to_use and d2_predictor and d2_metadata:
            d2_dets = run_detectron2_detection(
                image_pil=image_pil,
                detectron2_predictor=d2_predictor,
                detectron2_metadata=d2_metadata,
                device=device
            )
            all_detections.extend(d2_dets)

        if not all_detections:
            print("[INFO] No detections found. Skipping.")
            return

        # 2) Merge by label-based NMS
        by_label = defaultdict(list)
        for i, d in enumerate(all_detections):
            by_label[d["label"].lower()].append(i)

        final_indices = []
        for lbl, idx_list in by_label.items():
            label_boxes = [all_detections[i]["box"] for i in idx_list]
            label_scores= [all_detections[i]["score"] for i in idx_list]
            keep_idxs   = non_maximum_suppression(label_boxes, label_scores, label_nms_threshold)
            final_indices.extend([idx_list[k] for k in keep_idxs])
        final_indices.sort()

        # Build final detection arrays
        boxes  = [all_detections[i]["box"] for i in final_indices]
        labels = [all_detections[i]["label"] for i in final_indices]
        scores = [all_detections[i]["score"] for i in final_indices]

        # 3) Generate SAM masks, remove duplicates with segmentation
        image_np = np.array(image_pil)
        all_masks = mask_generator.generate(image_np)
        boxes, labels, scores = filter_segmentation_duplicates(
            boxes, labels, scores, all_masks, seg_iou_threshold
        )

        # 4) Relationship inference
        rels = infer_relationships_improved(
            boxes=boxes,
            labels=labels,
            overlap_thresh=overlap_thresh,
            margin=margin,
            min_distance=min_distance,
            max_distance=max_distance,
            top_k=max_relations
        )

        # Limit relationships per object (example usage)
        rels = limit_relationships_per_object(
            rels, boxes,
            max_per_object=4,
            min_per_object=1,
            min_relation_distance=60
        )

        # Ensure each pair of objects has only ONE relationship
        rels = unify_pair_relations(rels)

        # 5) Visualization
        out_name = f"{image_name}_output.jpg"
        out_path = os.path.join(output_folder, out_name)

        visualize_detections_and_relationships_with_auto_masks(
            image=image_pil,
            boxes=boxes,
            labels=labels,
            scores=scores,
            relationships=rels,
            all_masks=all_masks,
            view_relations_labels=False,  
            label_mode="original",
            show_confidence=False,
            draw_relationships=True,
            display_labels=True,
            show_segmentation=True,
            fill_segmentation=False,
            save_path=out_path
        )

        elapsed = time.time() - start_t
        print(f"[DONE] {image_name} processed in {elapsed:.2f}s")

    # =========================================================================
    # MAIN EXECUTION: dataset or local files
    # =========================================================================
    if dataset_name:
        # We do the dataset approach (Hugging Face)
        if not HAVE_DATASETS:
            print("[ERROR] 'datasets' library not installed, cannot load a dataset.")
            return

        print(f"[INFO] Loading dataset '{dataset_name}' with split='{split}'")
        dataset = load_dataset(dataset_name, split=split)
        print(f"[INFO] Dataset loaded with {len(dataset)} items")

        # Determine start/end indices
        if start_index < 0:
            start_index = 0
        if num_instances > 0:
            computed_end = start_index + num_instances - 1
            end_index = computed_end if end_index < 0 else min(end_index, computed_end)
        elif end_index < 0:
            end_index = len(dataset) - 1
        end_index = min(end_index, len(dataset) - 1)

        print(f"[INFO] Will process dataset items from index {start_index} to {end_index}")

        for i in range(start_index, end_index + 1):
            if i >= len(dataset):
                break
            example = dataset[i]
            # Attempt to retrieve the image
            if image_column not in example:
                print(f"[ERROR] image_column='{image_column}' not found in dataset columns.")
                print("[INFO] Available columns:", list(example.keys()))
                break

            img_data = example[image_column]
            if isinstance(img_data, Image.Image):
                img_pil = img_data
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                from io import BytesIO
                img_pil = Image.open(BytesIO(img_data['bytes'])).convert("RGB")
            elif isinstance(img_data, np.ndarray):
                img_pil = Image.fromarray(img_data).convert("RGB")
            else:
                print(f"[WARNING] Index {i}: image is not a recognized format. Skipping.")
                continue

            image_name = str(example.get('id', f"img_{i}"))
            process_single_image(img_pil, image_name)

    else:
        # Local file approach
        if not input_path:
            print("[ERROR] No input_path provided and no dataset given.")
            return

        if os.path.isdir(input_path):
            valid_exts = (".jpg",".jpeg",".png")
            all_files  = [f for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
            image_paths = [os.path.join(input_path, f) for f in all_files]
        else:
            if not os.path.exists(input_path):
                print(f"[ERROR] Input path '{input_path}' does not exist.")
                return
            image_paths = [input_path]

        if not image_paths:
            print("[ERROR] No images found in the specified path.")
            return

        for ipath in image_paths:
            try:
                img_pil = Image.open(ipath).convert("RGB")
                base_name = os.path.splitext(os.path.basename(ipath))[0]
                process_single_image(img_pil, base_name)
            except Exception as e:
                print(f"[ERROR] Could not process '{ipath}': {e}")



if __name__ == "__main__":
    main()
