#!/usr/bin/env python

import os
import cv2
import json
import torch
from torch.amp import autocast
import spacy
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import wordnet
from spacy.matcher import PhraseMatcher
from collections import defaultdict
from matplotlib.patches import FancyArrowPatch
import string
import math
from pathlib import Path
from torch.hub import download_url_to_file

# YOLOv8
from ultralytics import YOLO

# Detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

# OWL-ViT
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Segment Anything (SAM)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Optional HF datasets
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False

###############################################################################
# ARGUMENT PARSING
###############################################################################
def parse_preproc_args():
    parser = argparse.ArgumentParser(description="Image Graph Preprocessor")

    # Input/output
    parser.add_argument("--input_path",    type=str, help="Path to image or folder")
    parser.add_argument("--output_folder", type=str, default="output_images")
    parser.add_argument("--dataset",       type=str, default=None)
    parser.add_argument("--split",         type=str, default="train")
    parser.add_argument("--image_column",  type=str, default="image")

    # Batch limit
    parser.add_argument("--num_instances", type=int, default=-1,
                        help="Se >0, process only le prime N istanze")


    # Question filtering
    parser.add_argument("--question", type=str, default="",
                        help="Domanda per filtrare oggetti/relazioni")
    parser.add_argument("--disable_question_filter", action="store_true")

    # Detectors & relations
    parser.add_argument("--detectors",         type=str, default="owlvit,yolov8,detectron2")
    parser.add_argument("--relationship_type", type=str, default="all")
    parser.add_argument("--max_relations",     type=int, default=8)

    # Thresholds
    parser.add_argument("--owl_threshold",      type=float, default=0.15)
    parser.add_argument("--yolo_threshold",     type=float, default=0.3)
    parser.add_argument("--detectron_threshold",type=float, default=0.3)

    # NMS
    parser.add_argument("--label_nms_threshold", type=float, default=0.5)
    parser.add_argument("--seg_iou_threshold",   type=float, default=0.8)

    # Relation inference
    parser.add_argument("--overlap_thresh", type=float, default=0.3)
    parser.add_argument("--margin",         type=int,   default=20)
    parser.add_argument("--min_distance",   type=float, default=90)
    parser.add_argument("--max_distance",   type=float, default=20000)

    # SAM
    parser.add_argument("--points_per_side",        type=int,   default=32)
    parser.add_argument("--pred_iou_thresh",        type=float, default=0.9)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--min_mask_region_area",   type=int,   default=100)

    # JSON mode
    parser.add_argument("--json_file", type=str, default="")

    # Device override
    parser.add_argument("--preproc_device", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args

###############################################################################
# Setup
###############################################################################
nlp = spacy.load("en_core_web_md")

###############################################################################
# Helpers: WordNet, NMS, SAM masks, relations...
###############################################################################
def get_wordnet_synonyms(term: str) -> set:
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def extract_query_terms(question: str) -> list:
    doc = nlp(question)
    candidates = []
    for token in doc:
        if not token.is_stop and token.pos_ in ["NOUN","PROPN"]:
            candidates.append(token.text.lower())
            for s in get_wordnet_synonyms(token.text):
                candidates.append(s)
    return list(dict.fromkeys(candidates))

def build_relation_mapping() -> dict:
    base_map = {
        "above":   ["above"],
        "below":   ["below","under"],
        "left_of": ["left","to the left of"],
        "right_of":["right","to the right of"]
    }
    return base_map

def extract_relation_terms(question: str) -> list:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    rel_map = build_relation_mapping()
    for rel, phrases in rel_map.items():
        matcher.add(rel, [nlp.make_doc(ph) for ph in phrases])
    doc = nlp(question)
    found = set()
    for mid, start, end in matcher(doc):
        found.add(nlp.vocab.strings[mid])
    return list(found)

def compute_iou(b1, b2) -> float:
    x1,y1,x2,y2 = b1; xA,yA,xB,yB = b2
    ix1,iy1 = max(x1,xA), max(y1,yA)
    ix2,iy2 = min(x2,xB), min(y2,yB)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if not inter:
        return 0.0
    a1 = (x2-x1)*(y2-y1); a2 = (xB-xA)*(yB-yA)
    return inter / float(a1+a2-inter)

def non_maximum_suppression(boxes, scores, iou_thresh):
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if compute_iou(boxes[current], boxes[i])<iou_thresh]
    return keep

def get_best_mask_for_box(box, masks, iou_thresh=0.3):
    best, best_iou, best_score = None, 0.0, 0.0
    for m in masks:
        bx,by,w,h = m['bbox']
        mask_box = [bx,by,bx+w,by+h]
        iou = compute_iou(box, mask_box)
        score = m.get('predicted_iou',0)
        if iou>best_iou and score>best_score:
            best_iou, best_score, best = iou, score, m
    return best if best_iou>=iou_thresh else None

def bbox_from_contour(c):
    x,y,w,h = cv2.boundingRect(c)
    return [x,y,x+w,y+h]

def filter_segmentation_duplicates(boxes, labels, scores, masks, iou_thresh):
    refined = []
    for b,l,s in zip(boxes,labels,scores):
        m = get_best_mask_for_box(b, masks, 0.3)
        refined.append({"box":b,"label":l,"score":s,"mask":m})
    idxs = non_maximum_suppression([r["box"] for r in refined],
                                   [r["score"] for r in refined],
                                   iou_thresh)
    return ([refined[i]["box"] for i in idxs],
            [refined[i]["label"] for i in idxs],
            [refined[i]["score"] for i in idxs])

def infer_relationships_improved(boxes, overlap_thresh=0.3, margin=20,
                                 min_distance=90, max_distance=20000, top_k=None):
    centers = [((b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes]
    rels = []
    n = len(boxes)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            dx,dy = centers[j][0]-centers[i][0], centers[j][1]-centers[i][1]
            dist_sq = dx*dx+dy*dy
            if dist_sq<min_distance**2 or dist_sq>max_distance**2: continue
            # overlap check
            b1 = [boxes[i][0]-margin, boxes[i][1]-margin,
                  boxes[i][2]+margin, boxes[i][3]+margin]
            b2 = [boxes[j][0]-margin, boxes[j][1]-margin,
                  boxes[j][2]+margin, boxes[j][3]+margin]
            if compute_iou(b1,b2)>overlap_thresh: continue
            # direction
            if abs(dy)>abs(dx):
                rel = "above" if dy>0 else "below"
            else:
                rel = "left_of" if dx>0 else "right_of"
            rels.append({"subject":i,"object":j,
                         "relation":rel,"distance":math.sqrt(dist_sq)})
    if top_k and len(rels)>top_k:
        rels = sorted(rels, key=lambda x:x["distance"])[:top_k]
    return rels

def limit_relationships_per_object(rels, boxes, max_per_obj, min_per_obj=None, min_rel_dist=80):
    by_src = defaultdict(list)
    for r in rels: by_src[r["subject"]].append(r)
    out=[]
    for lst in by_src.values():
        ordered = sorted(lst, key=lambda r:r["distance"])
        kept=[]
        for r in ordered:
            if r["distance"]<min_rel_dist: continue
            if len(kept)<max_per_obj: kept.append(r)
        out+=kept
    return out

def unify_pair_relations(rels):
    best={}
    for r in rels:
        key = tuple(sorted((r["subject"],r["object"])))
        if key not in best or r["distance"]<best[key]["distance"]:
            best[key]=r
    return list(best.values())

def move_point_outside_contour(pt, contour, step=5, max_iter=20):
    new_pt = np.array(pt, dtype=float)
    M = cv2.moments(contour)
    if M["m00"]!=0:
        cx,cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    else:
        return pt
    dir_vec = new_pt-np.array([cx,cy])
    norm = np.linalg.norm(dir_vec) or 1.0
    dir_vec /= norm
    for _ in range(max_iter):
        if cv2.pointPolygonTest(contour, tuple(new_pt), False)<0:
            break
        new_pt += dir_vec*step
    return tuple(new_pt)

def adjust_position(pt, placed, overlap, max_iter=10):
    new_pt = np.array(pt, dtype=float)
    for _ in range(max_iter):
        moved=False
        for p in placed:
            diff=new_pt-np.array(p)
            d=np.linalg.norm(diff)
            if d<overlap:
                if d==0: diff=np.random.randn(2); d=np.linalg.norm(diff)
                new_pt += diff/d*(overlap-d)
                moved=True; break
        if not moved: break
    return tuple(new_pt)

def visualize_detections_and_relationships_with_auto_masks(
    image, boxes, labels, scores, relationships, all_masks,
    view_relations_labels=False, label_mode="original",
    show_confidence=True, draw_relationships=True,
    display_labels=True, show_segmentation=True,
    fill_segmentation=False, save_path=None
):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(image); ax.axis("off")

    placed=[]; overlap_thresh=30
    colors=['red','green','blue','magenta','cyan','orange','purple','brown']
    obj_colors=[colors[i%len(colors)] for i in range(len(boxes))]

    if label_mode=="numeric":
        vis_labels=[str(i+1) for i in range(len(boxes))]
    elif label_mode=="alphabetic":
        vis_labels=list(string.ascii_uppercase[:len(boxes)])
    else:
        vis_labels=labels

    def place_label(pt):
        adj=adjust_position(pt, placed, overlap_thresh)
        placed.append(adj)
        return adj

    detection_info=[]; arrow_counts=defaultdict(int)

    # draw masks/bboxes
    for i,box in enumerate(boxes):
        color=obj_colors[i]
        x1,y1,x2,y2=map(int,box)
        best_m = get_best_mask_for_box(box, all_masks)
        if show_segmentation and best_m is not None:
            m_uint=(best_m["segmentation"]*255).astype(np.uint8)
            ctrs,_=cv2.findContours(m_uint,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if ctrs:
                cnt=max(ctrs,key=cv2.contourArea).squeeze()
                if fill_segmentation:
                    ax.fill(cnt[:,0],cnt[:,1],color=color,alpha=0.2,zorder=1)
                ax.plot(cnt[:,0],cnt[:,1],color=color,linewidth=2,zorder=2)
                M=cv2.moments(cnt)
                if M["m00"]!=0:
                    cx,cy=M["m10"]/M["m00"],M["m01"]/M["m00"]
                    center=(cx,cy)
                    center=move_point_outside_contour(center,cnt)
                else:
                    center=((x1+x2)/2,(y1+y2)/2)
            else:
                ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=2,zorder=2))
                center=((x1+x2)/2,(y1+y2)/2)
        else:
            ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=2,zorder=2))
            center=((x1+x2)/2,(y1+y2)/2)

        if display_labels:
            txt = f"{vis_labels[i]}:{scores[i]:.2f}" if show_confidence else vis_labels[i]
            pt_label=place_label(center)
            detection_info.append((pt_label,txt,color))

    # draw relationships
    if draw_relationships and relationships:
        centers=[((b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes]
        for r in relationships:
            s,o=r["subject"],r["object"]
            c1,c2=centers[s],centers[o]; color=obj_colors[s]
            arrow_counts[(s,o)]+=1
            rad=0.2+0.1*(arrow_counts[(s,o)]-1)
            arr=FancyArrowPatch(c1,c2,arrowstyle='->',
                                color=color,linewidth=2,
                                connectionstyle=f'arc3,rad={rad}',
                                mutation_scale=12,zorder=4)
            ax.add_patch(arr)
            if view_relations_labels:
                mx,my=(c1[0]+c2[0])/2,(c1[1]+c2[1])/2
                dx,dy=c2[0]-c1[0],c2[1]-c1[1]
                ang=math.degrees(math.atan2(dy,dx))
                offx,offy = -dy/np.hypot(dx,dy)*10, dx/np.hypot(dx,dy)*10
                pt_txt=adjust_position((mx+offx,my+offy),placed,overlap_thresh)
                placed.append(pt_txt)
                ax.text(pt_txt[0],pt_txt[1],r["relation"],
                        fontsize=8,color=color,
                        rotation=ang,rotation_mode='anchor',
                        bbox=dict(facecolor='white',alpha=0.8,edgecolor=color),
                        zorder=5)

    # object labels
    for pt,txt,color in detection_info:
        ax.text(pt[0],pt[1],txt,
                fontsize=10,color=color,
                bbox=dict(facecolor='white',alpha=0.8,edgecolor=color),
                zorder=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

###############################################################################
# ImageGraphPreprocessor
###############################################################################
class ImageGraphPreprocessor:
    def __init__(self, args):
        self.args = vars(args) if hasattr(args,'__dict__') else args
        self.apply_q_filter = not self.args.get('disable_question_filter',False)
        os.makedirs(self.args['output_folder'],exist_ok=True)
        self.device = (self.args.get('preproc_device') or
                      ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Scarica e carica SAM
        SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        SAM_LOCAL = Path(self.args["output_folder"]) / "sam_vit_h_4b8939.pth"
        if not SAM_LOCAL.exists():
            print("[INFO] scarico checkpoint SAM …")
            download_url_to_file(SAM_CKPT_URL, str(SAM_LOCAL))
        self.sam = sam_model_registry["vit_h"](checkpoint=str(SAM_LOCAL)).to(self.device)
        self.mask_gen = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=self.args["points_per_side"],
            pred_iou_thresh=self.args["pred_iou_thresh"],
            stability_score_thresh=self.args["stability_score_thresh"],
            min_mask_region_area=self.args["min_mask_region_area"],
        )

        # Detectors
        detectors_str = self.args.get('detectors','owlvit,yolov8,detectron2')
        self.detector_list = [d.strip().lower() for d in detectors_str.split(',')]

        # Default OWL-ViT queries
        self.default_owlvit_queries = [
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

        # Lazy detectors
        self.owl_proc=None; self.owl_model=None
        self.yolo_model=None
        self.d2_pred=None; self.d2_meta=None

    def process_single_image(self, image_pil: Image.Image, name: str, question: str=""):
        # Estrazione termini
        obj_terms, rel_terms = set(), set()
        if self.apply_q_filter and question.strip():
            obj_terms = set(extract_query_terms(question))
            rel_terms = set(extract_relation_terms(question))

        # 1) Detection
        dets=[]
        if 'owlvit' in self.detector_list:
            if self.owl_proc is None:
                self.owl_proc = Owlv2Processor.from_pretrained('google/owlv2-large-patch14-ensemble')
                self.owl_model= Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14-ensemble')
                self.owl_model.to(self.device).eval()
            if (not self.apply_q_filter) or (not question.strip()):
                queries=self.default_owlvit_queries
            else:
                queries=list(obj_terms)
            dets += run_owlvit_detection(image_pil,queries,self.owl_proc,self.owl_model,
                                         threshold=self.args['owl_threshold'],device=self.device)

        if 'yolov8' in self.detector_list:
            if self.yolo_model is None:
                self.yolo_model= YOLO('yolov8x.pt').to(self.device)
            dets += run_yolov8_detection(image_pil,self.yolo_model,
                                         threshold=self.args['yolo_threshold'],device=self.device)

        if 'detectron2' in self.detector_list:
            if self.d2_pred is None:
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(
                    'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args['detectron_threshold']
                cfg.MODEL.DEVICE = self.device
                self.d2_pred=DefaultPredictor(cfg)
                self.d2_meta=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            dets += run_detectron2_detection(image_pil,self.d2_pred,self.d2_meta,device=self.device)

        if not dets: return

        # 2) Label-based NMS
        by_lbl=defaultdict(list)
        for i,d in enumerate(dets): by_lbl[d['label'].lower()].append(i)
        keep_idx=[]
        for idxs in by_lbl.values():
            boxes_=[dets[i]['box'] for i in idxs]
            scores_=[dets[i]['score'] for i in idxs]
            kept=non_maximum_suppression(boxes_,scores_,self.args['label_nms_threshold'])
            keep_idx+= [idxs[i] for i in kept]
        keep_idx.sort()
        boxes  = [dets[i]['box']   for i in keep_idx]
        labels = [dets[i]['label'] for i in keep_idx]
        scores = [dets[i]['score'] for i in keep_idx]

        # 3) SAM + dedupe
        img_np=np.array(image_pil)
        masks=self.mask_gen.generate(img_np)
        torch.cuda.empty_cache()
        boxes,labels,scores = filter_segmentation_duplicates(
            boxes,labels,scores,masks,self.args['seg_iou_threshold'])

        # 4) Relations
        rels = infer_relationships_improved(
            boxes,
            overlap_thresh=self.args['overlap_thresh'],
            margin=self.args['margin'],
            min_distance=self.args['min_distance'],
            max_distance=self.args['max_distance'],
            top_k=self.args['max_relations']
        )
        rels = limit_relationships_per_object(
            rels,boxes,
            max_per_obj=1,
            min_per_obj=1,
            min_rel_dist=self.args['min_distance']
        )
        rels = unify_pair_relations(rels)

        # 5) Filtro avanzato se ho obj_terms
        if self.apply_q_filter and obj_terms:
            # indici matched iniziali
            matched=[i for i,lbl in enumerate(labels) if lbl.lower() in obj_terms]
            if matched:
                related=[r for r in rels if r['subject'] in matched or r['object'] in matched]
                keep_ids=set()
                for r in related:
                    keep_ids.add(r['subject']); keep_ids.add(r['object'])
                selected=sorted(keep_ids)
                old2new={old:new for new,old in enumerate(selected)}
                boxes  =[boxes[i]   for i in selected]
                labels =[labels[i]  for i in selected]
                scores =[scores[i]  for i in selected]
                # rimappa rels
                new_rels=[]
                for r in related:
                    s,o=r['subject'],r['object']
                    if s in old2new and o in old2new:
                        new_rels.append({
                            'subject':old2new[s],
                            'object': old2new[o],
                            'relation':r['relation'],
                            'distance':r['distance']
                        })
                rels=new_rels

        # 6) Visualize
        out_path=os.path.join(self.args['output_folder'],f"{name}_output.jpg")
        visualize_detections_and_relationships_with_auto_masks(
            image=image_pil,
            boxes=boxes,
            labels=labels,
            scores=scores,
            relationships=rels,
            all_masks=masks,
            view_relations_labels=False,
            label_mode="original",
            show_confidence=False,
            draw_relationships=True,
            display_labels=True,
            show_segmentation=True,
            fill_segmentation=True,
            save_path=out_path
        )

    def run(self):
        args = self.args
        # JSON mode
        if args.get('json_file'):
            with open(args['json_file'],'r',encoding='utf-8') as f:
                rows=json.load(f)
            if args.get('num_instances', -1) > 0:
                rows = rows[: args['num_instances']]
            for row in rows:
                img_p=row['image_path']
                q=row.get('question','') if not args.get('disable_question_filter') else ''
                try:
                    if img_p.startswith('http'):
                        import requests
                        img=Image.open(requests.get(img_p,timeout=30).raw).convert('RGB')
                    else:
                        img=Image.open(img_p).convert('RGB')
                except Exception as e:
                    print(f"[ERROR] Loading {img_p}: {e}")
                    continue
                name=os.path.splitext(os.path.basename(img_p))[0]
                self.process_single_image(img,name,q)
            return

        # Dataset or folder
        if args.get('dataset'):
            if not HAVE_DATASETS:
                print("[ERROR] datasets library not installed."); return
            ds=load_dataset(args['dataset'],split=args['split'])
            start=0 if args.get('start_index',-1)<0 else args['start_index']
            end=len(ds)-1 if args.get('end_index',-1)<0 else args['end_index']
            if args.get('num_instances',-1)>0:
                end=min(end,start+args['num_instances']-1)
            for i in range(start,end+1):
                ex=ds[i]
                img_data=ex.get(args['image_column'])
                if isinstance(img_data,dict) and 'bytes' in img_data:
                    from io import BytesIO
                    img=Image.open(BytesIO(img_data['bytes'])).convert('RGB')
                elif isinstance(img_data,np.ndarray):
                    img=Image.fromarray(img_data).convert('RGB')
                else:
                    continue
                name=str(ex.get('id',f"img_{i}"))
                self.process_single_image(img,name,args['question'] if self.apply_q_filter else '')
        else:
            if os.path.isdir(args['input_path']):
                exts=('.jpg','.jpeg','.png')
                files=[f for f in os.listdir(args['input_path']) if f.lower().endswith(exts)]
                paths=[os.path.join(args['input_path'],f) for f in files]
            else:
                paths=[args['input_path']]
            for p in paths:
                try:
                    img=Image.open(p).convert('RGB')
                except:
                    continue
                name=os.path.splitext(os.path.basename(p))[0]
                self.process_single_image(img,name,args['question'] if self.apply_q_filter else '')

###############################################################################
# Detector wrappers
###############################################################################
@torch.inference_mode()
def run_owlvit_detection(
    image_pil: Image.Image,
    queries,
    processor: Owlv2Processor,
    model: Owlv2ForObjectDetection,
    threshold: float = 0.1,
    device="cuda"
) -> list:
    # Se queries è una lista, facciamo una detection separata per ciascuna voce
    if isinstance(queries, list):
        all_dets = []
        for q in queries:
            all_dets.extend(
                run_owlvit_detection(
                    image_pil,
                    q,
                    processor,
                    model,
                    threshold=threshold,
                    device=device
                )
            )
        return all_dets

    # Altrimenti queries è una singola stringa
    with autocast(device_type="cuda"):
        inputs = processor(
            text=queries,
            images=[image_pil],
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [image_pil.size[::-1]],
        device=device
    )
    results_list = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=target_sizes
    )
    if not results_list or results_list[0] is None:
        return []

    res = results_list[0]
    dets = []
    for box, score, lab_idx in zip(
        res["boxes"].cpu(),
        res["scores"].cpu(),
        res["labels"].cpu()
    ):
        if float(score) >= threshold:
            b = box.tolist()
            l = processor.tokenizer.decode([lab_idx]).strip()
            dets.append({"box": b, "label": l, "score": float(score)})
    return dets


@torch.inference_mode()
def run_yolov8_detection(image_pil: Image.Image,
                         yolo_model,
                         threshold: float=0.1,
                         device="cuda") -> list:
    if yolo_model is None: return []
    max_side=800; w,h=image_pil.size
    if max(w,h)>max_side:
        scale=max_side/max(w,h)
        image_pil=image_pil.resize((int(w*scale),int(h*scale)))
    image_np=np.array(image_pil)
    with autocast(device_type="cuda"):
        results=yolo_model.predict(image_np,device=device)[0]
    dets=[]
    for box_xyxy,conf,cls_idx in zip(results.boxes.xyxy,results.boxes.conf,results.boxes.cls):
        score=float(conf)
        if score<threshold: continue
        b=box_xyxy.tolist()
        lbl=results.names.get(int(cls_idx),f"class_{int(cls_idx)}")
        dets.append({"box":b,"label":lbl,"score":score})
    return dets

@torch.inference_mode()
def run_detectron2_detection(image_pil: Image.Image,
                             predictor,
                             metadata,
                             device="cuda") -> list:
    if predictor is None: return []
    max_side=800; w,h=image_pil.size
    if max(w,h)>max_side:
        scale=max_side/max(w,h)
        image_pil=image_pil.resize((int(w*scale),int(h*scale)))
    image_np=np.array(image_pil)
    with autocast(device_type="cuda"):
        outputs=predictor(image_np)
    inst=outputs.get("instances")
    if inst is None: return []
    inst=inst.to('cpu')
    dets=[]
    boxes=inst.pred_boxes; scores=inst.scores; classes=inst.pred_classes
    for box,sc,ci in zip(boxes,scores,classes):
        s=float(sc)
        if s<predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST: continue
        b=box.tolist()
        lbl=metadata.thing_classes[int(ci)]
        dets.append({"box":b,"label":lbl,"score":s})
    return dets

###############################################################################
# MAIN
###############################################################################
def main():
    args=parse_preproc_args()
    preproc=ImageGraphPreprocessor(args)
    preproc.run()

if __name__=="__main__":
    main()
