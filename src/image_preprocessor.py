
import os
import gc
import psutil
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
import matplotlib.patheffects as pe
import string
from collections import defaultdict
import argparse
from pathlib import Path
from torch.hub import download_url_to_file
from adjustText import adjust_text
import networkx as nx
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision.models.optical_flow import raft_large
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import requests, functools, urllib.parse
from nltk.stem import WordNetLemmatizer
import difflib
from ensemble_boxes import weighted_boxes_fusion
import re
from matplotlib.transforms import Bbox


# SAM-HQ (nuovo import)
try:
    from segment_anything_hq import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    HAVE_SAM_HQ = True
except ImportError:
    sam_model_registry = SamPredictor = SamAutomaticMaskGenerator = None
    HAVE_SAM_HQ = False


try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    # Il modulo verrà importato solo se sam_version == 2
    build_sam2 = SAM2ImagePredictor = SAM2AutomaticMaskGenerator = None


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


# For optional Hugging Face dataset usage
try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False

import colorsys, random

_CONCEPTNET_CACHE: dict[tuple[str,str], list[dict]] = {}

BASIC_COLORS = [
     "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
     "#ffff33", "#a65628", "#f781bf", "#999999", "#1f78b4",
     "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#cab2d6",
     "#6a3d9a", "#b2df8a", "#ffed6f", "#a6cee3", "#b15928"
]

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
        # 0)  Ensure external resources are present
        # ---------------------------------------------------------
        self._ensure_nltk_corpora(["wordnet", "omw-1.4"])
        self._ensure_spacy_model("en_core_web_md")

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
        self._on_top_gap_px = 8          # tolleranza verticale in pixel
        self._on_top_horiz_overlap = 0.35  # overlap orizzontale minimo (ratio)

        # Label & visualization settings
        self.question = config.get("question", "")
        self.apply_question_filter = config.get("apply_question_filter", True)
        self.aggressive_pruning = config.get("aggressive_pruning", False)
        self.label_mode = config.get("label_mode", "original")  # "original","numeric","alphabetic"
        self.display_labels = config.get("display_labels", True)
        self.display_relationships = config.get("display_relationships", True)
        self.show_segmentation = config.get("show_segmentation", True)
        self.fill_segmentation = config.get("fill_segmentation", True)
        self.display_legend = config.get("display_legend", True)
        self.text_outline_lw = config.get("text_outline_lw", 1.5)
        self.rel_text_outline_lw = config.get("rel_text_outline_lw", 1.2)
        self.seg_fill_alpha    = config.get("seg_fill_alpha", 0.55)    # opacità riempimento
        self.bbox_linewidth    = config.get("bbox_linewidth", 3.0)     # spessore bordi
        self.resolve_overlaps = config.get("resolve_overlaps", True)
        self.show_bboxes       = config.get("show_bboxes", True)
        self.rel_arrow_lw = config.get("rel_arrow_linewidth", 3.5)
        self.rel_arrow_ms = config.get("rel_arrow_mutation_scale", 22)


        self.obj_fs_in  = config.get("obj_fontsize_inside", 12)   # dentro al box
        self.obj_fs_out = config.get("obj_fontsize_outside", 12)  # etichette esterne
        self.rel_fs     = config.get("rel_fontsize", 10)          # relazioni
        self.legend_fs  = config.get("legend_fontsize", 8)


        # Additional toggles
        self.show_confidence = config.get("show_confidence", False)
        self.display_relation_labels = config.get("display_relation_labels", True)

        # Area filtering
        self.enable_area_filter = config.get("enable_area_filter", False)
        self.min_area = config.get("min_area", 500)
        self.max_area = config.get("max_area", None)

        # Overlap & NMS thresholds
        self.label_nms_threshold = config.get("label_nms_threshold", 0.5)
        self.seg_iou_threshold = config.get("seg_iou_threshold", 0.8)

        # Relationship inference geometry
        self.margin = config.get("margin", 20)
        self.min_distance = config.get("min_distance", 90)
        self.max_distance = config.get("max_distance", 20000)

        # SAM parameters
        self.points_per_side = config.get("points_per_side", 32)
        self.pred_iou_thresh = config.get("pred_iou_thresh", 0.9)
        self.stability_score_thresh = config.get("stability_score_thresh", 0.92)
        self.min_mask_region_area = config.get("min_mask_region_area", 100)

        # Dataset indexing
        self.start_index = config.get("start_index", -1)
        self.end_index = config.get("end_index", -1)
        self.num_instances = config.get("num_instances", -1)

        self.cnet_timeout   = config.get("conceptnet_timeout", 4)   # secondi
        self.cnet_max_retry = config.get("conceptnet_max_retry", 3)
        self.cnet_skip_on_fail = config.get("conceptnet_skip_on_fail", True)


        self.sam_version = config.get("sam_version", "1")
        self.sam_hq_model_type = config.get("sam_hq_model_type", "vit_h")

        # Cached detenction
        self._det_cache = {}

        self._global_model_cache = {}
        self._detection_cache = {}

        # Configurazione cache
        self.enable_detection_cache = config.get("enable_detection_cache", True)
        self.max_cache_size = config.get("max_cache_size", 100)  # Limite memoria

        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            if total_mem < 20 * 1024**3:  # Meno di 20GB
                print(f"[WARNING] GPU memory: {total_mem/1024**3:.1f}GB might be insufficient")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # Forza l'uso di un singolo thread per evitare race conditions
        torch.set_num_threads(1)
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
        if self.sam_version == "hq":
            self._load_sam_hq_model()
        elif self.sam_version == "2":
            self._load_sam2_model()
        else:
            self._load_sam_model()

        # --------------------------------------------------------------------------
        # 3bis)  CLIP  ‒  usato per costruire embedding dei nodi
        # --------------------------------------------------------------------------
        print("[INFO] Loading CLIP ViT-L/14 …")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device).eval()


        print("[INFO] Loading MiDaS …")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
        self.midas_trans = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device).eval()

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

        self.alias2label = self._auto_build_alias2label(
            extra_manual={
                "tv": "tv",
                "fridge": "refrigerator",
                "bike": "bicycle"
            }
        )

        self.label_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp(lbl.lower()) for lbl in self.detectors_label_list]
        self.label_matcher.add("LABEL", None, *patterns)


        self.base_cmap   = None          # 20 colori distinti
        self._label2col  = {}

        self.curr_img_size = (0, 0)

        # —––––––––––––––––––––––––––––––––––––––––––––––—
        # 4) ORA posso ricostruire i termini della domanda
        # —––––––––––––––––––––––––––––––––––––––––––––––—

        # Reset dei set (utile se __init__ viene richiamato due volte)
        self._parsed_question_object_terms   = set()
        self._parsed_question_relation_terms = set()

        # Ed ecco la chiamata che ora funziona correttamente:
        self._build_question_semantics()

        print(f"[DEBUG] display_legend: {self.display_legend}")
        print(f"[DEBUG] aggressive_pruning: {self.aggressive_pruning}")
        print(f"[DEBUG] apply_question_filter: {self.apply_question_filter}")


    # ------------------------------------------------------------------
    # helper statici
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_nltk_corpora(corpus_list, dl_dir="/root/nltk_data"):
        """
        Verifica che i corpora NLTK siano installati; se mancano li scarica
        nella cartella `dl_dir` (uno dei path che NLTK controlla di default).
        """
        import nltk
        from pathlib import Path

        for corp in corpus_list:
            try:
                nltk.data.find(f"corpora/{corp}")
            except LookupError:
                print(f"[INFO] Downloading NLTK corpus '{corp}' …")
                Path(dl_dir).mkdir(parents=True, exist_ok=True)
                nltk.download(corp, download_dir=dl_dir, quiet=True)

    @staticmethod
    def _ensure_spacy_model(model_name):
        """
        Carica `model_name`; se non esiste tenta lo `spacy download`.
        """
        import spacy
        try:
            spacy.load(model_name)
        except OSError:
            print(f"[INFO] Downloading spaCy model '{model_name}' …")
            from spacy.cli import download
            download(model_name, quiet=True)


    def _wbf_fusion(self, detections, iou_thr: float = 0.55):
        """
        Fonde i bounding-box provenienti da più detector.
        Ogni elemento di `detections` deve contenere:
            { "box":[x1,y1,x2,y2], "score":float, "label":str, "from":str }
        Ritorna tre liste allineate: boxes, labels, scores
        """
        if not detections:
            return [], [], []

        w, h = self.curr_img_size
        # costruiamo la mappa label → id una sola volta
        if not hasattr(self, "_label2id"):
            self._label2id = {lbl: idx for idx, lbl in enumerate(self.detectors_label_list)}

        list_boxes, list_scores, list_labels, list_weights = [], [], [], []
        for det in detections:
            lab = det["label"]
            if lab not in self._label2id:
                continue                       # salta subito l’intera detection

            x1, y1, x2, y2 = det["box"]
            list_boxes.append([[x1 / w, y1 / h, x2 / w, y2 / h]])
            list_scores.append([det["score"]])
            list_labels.append([self._label2id[lab]])
            list_weights.append(
                2.0 if det["from"] == "owlvit" else
                1.5 if det["from"] == "yolo"   else 1.0
            )
        fused_b, fused_s, fused_l = weighted_boxes_fusion(
            list_boxes, list_scores, list_labels,
            weights=list_weights, iou_thr=iou_thr, skip_box_thr=0.0
        )
        # back-scale a pixel
        fused_b = [ [b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in fused_b ]
        fused_l = [ self.detectors_label_list[int(i)] for i in fused_l ]
        fused_s = [ float(s) for s in fused_s ]
        return fused_b, fused_l, fused_s

    # ------------------------------------------------------------
    #  utility per flip orizzontale (TTA)
    # ------------------------------------------------------------
    def _hflip_pil(self, pil_img: Image.Image) -> Image.Image:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    ###########################################################################
    # QUESTION SEMANTICS
    ###########################################################################
    def _build_relation_matcher(self):
        base_map = {
            "above":      ["above"],
            "below":      ["below", "under"],
            "left_of":    ["left of", "to the left of"],
            "right_of":   ["right of", "to the right of"],
            "on_top_of":  ["on top of", "on", "onto", "resting on", "sitting on"]
        }
        for label, terms in base_map.items():
            patterns = [self.nlp(term) for term in terms]
            self.relation_matcher.add(label, None, *patterns)


    # ------------------------------------------------------------------
    #  ALIAS ↔ LABEL  (costruzione automatica)
    # ------------------------------------------------------------------
    def _auto_build_alias2label(
            self,
            extra_manual: dict[str, str] | None = None,
            min_wordnet_freq: int = 2,
            fuzzy_match: bool = True
    ) -> dict[str, str]:
        """
        Genera un mapping {alias → label_canonico} partendo da
        self.detectors_label_list e da WordNet.

        - `extra_manual` per forzare alias particolari.
        - `min_wordnet_freq` filtra sinonimi rari.
        - `fuzzy_match` aggiunge abbreviazioni trovate con difflib.
        """
        lemm = WordNetLemmatizer()
        alias2label: dict[str, str] = {}

        # helper singolare/plurale minimale
        def _sing_plur_forms(w: str) -> set[str]:
            s = set()
            if w.endswith("ies") and len(w) > 3:
                s.add(w[:-3] + "y")
            elif w.endswith("es"):
                s.add(w[:-2])
            elif w.endswith("s") and len(w) > 2:
                s.add(w[:-1])
            else:
                s.add(w + "s")
            return s

        for lab in self.detectors_label_list:
            canon = lab.lower()
            variants = {
                canon,
                canon.replace(" ", ""),
                canon.replace("-", ""),
                *(_sing_plur_forms(canon)),
                lemm.lemmatize(canon, pos="n"),
            }

            # sinonimi WordNet
            for syn in wordnet.synsets(canon, pos=wordnet.NOUN):
                for lem in syn.lemmas():
                    if lem.count() >= min_wordnet_freq:
                        name = lem.name().replace("_", " ").lower()
                        variants.add(name)
                        variants.update(_sing_plur_forms(name))

            for v in variants:
                alias2label.setdefault(v, canon)

        # fuzzy-match per abbreviazioni corte
        if fuzzy_match:
            vocab = list(alias2label.keys())
            for word in vocab:
                if len(word) <= 4:
                    best = difflib.get_close_matches(word, vocab, n=1, cutoff=0.83)
                    if best:
                        alias2label[word] = alias2label[best[0]]

        # override manuali
        if extra_manual:
            for k, v in extra_manual.items():
                alias2label[k.lower()] = v.lower()

        return alias2label


    def _build_question_semantics(self):
        """
        Popola:
            • self._parsed_question_object_terms   ← set di label detector rilevanti
            • self._parsed_question_relation_terms ← set di relazioni geometriche
        Pipeline:
            1. match esatto con PhraseMatcher
            2. alias/sinonimi (self.alias2label)
            3. fallback WordNet sui nomi comuni
            4. fallback CLIP (similarità testo-immagine)
        """
        self._parsed_question_object_terms = set()
        self._parsed_question_relation_terms = set()

        if (not self.apply_question_filter) or (not self.question.strip()):
            return

        doc = self.nlp(self.question)

        # 1) match esatto su label dei detector
        for _, s, e in self.label_matcher(doc):
            matched_text = doc[s:e].text.lower()
            self._parsed_question_object_terms.add(matched_text)

        # 2) alias / sinonimi
        for tok in doc:
            if tok.text.lower() in self.alias2label:
                canonical = self.alias2label[tok.text.lower()]
                self._parsed_question_object_terms.add(canonical)

        # 3) WordNet se ancora vuoto
        if not self._parsed_question_object_terms:
            query_terms = self._extract_query_terms(self.question)
            for qt in query_terms:
                syns = self._get_wordnet_synonyms(qt)
                for syn in syns:
                    if syn in self.detectors_label_list:
                        self._parsed_question_object_terms.add(syn)

        # 4) fallback CLIP con gestione corretta dei tensori
        if (not self._parsed_question_object_terms) and hasattr(self, "clip_model"):
            with torch.no_grad():  # 👈 Aggiungi questo context manager
                # Embedding del testo della domanda
                txt_inputs = self.clip_processor(
                    text=[self.question],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                txt_emb = self.clip_model.get_text_features(**txt_inputs)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

                # Embedding delle label
                label_embs = []
                for lbl in self.detectors_label_list:
                    lbl_inputs = self.clip_processor(
                        text=[lbl],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    lbl_emb = self.clip_model.get_text_features(**lbl_inputs)
                    lbl_emb = lbl_emb / lbl_emb.norm(dim=-1, keepdim=True)
                    label_embs.append(lbl_emb)

                # Clona i tensori prima del calcolo di similarità
                txt_emb_clone = txt_emb.clone().detach()
                label_embs_stack = torch.stack(label_embs).squeeze(-2)
                label_embs_clone = label_embs_stack.clone().detach()

                # Calcolo similarità con tensori clonati
                sims = torch.mm(txt_emb_clone, label_embs_clone.T).squeeze(0)

                # Soglia e selezione
                threshold = 0.25
                top_indices = (sims > threshold).nonzero(as_tuple=True)[0]

                for idx in top_indices:
                    self._parsed_question_object_terms.add(self.detectors_label_list[idx])

        # Estrazione termini relazione
        self._parsed_question_relation_terms = set(self._extract_relation_terms(self.question))

        print(f"[DEBUG] Question: '{self.question}'")
        print(f"[DEBUG] Parsed object terms: {self._parsed_question_object_terms}")
        print(f"[DEBUG] Parsed relation terms: {self._parsed_question_relation_terms}")





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
            "above":      ["above"],
            "below":      ["below", "under"],
            "left_of":    ["left", "to the left of"],
            "right_of":   ["right", "to the right of"],
            "on_top_of":  ["on top of", "on", "onto", "resting on", "sitting on"]
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
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

        if self.sam_version == "2":
            return self._load_sam2_model()

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

        self.sam_model_gpu = sam_model_registry[sam_model_type](checkpoint=str(checkpoint_path))
        # self.sam_model_gpu.to(self.device).half().eval()
        self.sam_model_cpu = sam_model_registry[sam_model_type](checkpoint=str(checkpoint_path)).eval()


        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam_model_gpu,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
            crop_n_layers=0
        )
        self.sam_predictor = SamPredictor(self.sam_model_gpu)

    def _load_sam2_model(self):
        """
        • Verifica presenza checkpoint e YAML di SAM-2.1.
        • Se assenti li scarica da dl.fbaipublicfiles.com con progress bar.
        • Poi inizializza sam2_model, sam2_predictor, mask_generator.
        """
        import requests, sys, math
        from pathlib import Path
        from tqdm.auto import tqdm   # progress bar elegante (tqdm è pre-installato su Colab)

        # ---- percorsi locali ----------------------------------------------------
        ckpt_path = Path("./checkpoints/sam2.1_hiera_large.pt")
        cfg_path  = Path("./configs/sam2.1/sam2.1_hiera_l.yaml")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- url sorgenti -------------------------------------------------------
        BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
        CKPT_URL = f"{BASE}/sam2.1_hiera_large.pt"
        # sam2/configs/sam2.1/sam2.1_hiera_l.yaml
        YAML_URL = (
            "https://raw.githubusercontent.com/facebookresearch/"
            "configs/sam2.1/sam2.1_hiera_l.yaml"
        )

        # ------------------------------------------------------------------------
        def _download(url: str, dest: Path):
            """Scarica con stream + progress bar, salta se già completo."""
            headers = {"User-Agent": "Mozilla/5.0"}         # evita 403
            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"↓ {dest.name}",
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                tmp.rename(dest)

        # ---- checkpoint ---------------------------------------------------------
        if not ckpt_path.exists():
            print("[INFO] Checkpoint SAM-2.1 assente: avvio download (~2 GB)…")
            _download(CKPT_URL, ckpt_path)
        else:
            print("[INFO] Checkpoint SAM-2.1 già presente.")

        # ---- yaml ---------------------------------------------------------------
        #if not cfg_path.exists():
        #    print("[INFO] YAML di configurazione assente: download…")
        #    _download(YAML_URL, cfg_path)
        #else:
        #    print("[INFO] YAML già presente.")

        # ---- inizializzazione modello ------------------------------------------
        print("[INFO] Loading SAM-2.1 …")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        torch.cuda.empty_cache()
        self.sam2_model = build_sam2(
            model_cfg,
            str(ckpt_path),
            device=self.device,
            precision='fp16'
        ).eval()

        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam2_model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def _load_sam_hq_model(self):
        """
        Carica SAM-HQ con checkpoint e inizializza predictor e mask generator.
        """

        try:
            from segment_anything_hq import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            HAVE_SAM_HQ = True
        except ImportError:
            sam_model_registry = SamPredictor = SamAutomaticMaskGenerator = None
            HAVE_SAM_HQ = False

        if not HAVE_SAM_HQ:
            raise ImportError("SAM-HQ not available. Install with: pip install git+https://github.com/SysCV/sam-hq.git")

        print("[INFO] Loading SAM-HQ model ...")

        # URL per i checkpoint SAM-HQ
        SAM_HQ_URLS = {
            "vit_b": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
            "vit_l": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
            "vit_h": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
        }

        # Percorso checkpoint
        checkpoint_name = f"sam_hq_{self.sam_hq_model_type}.pth"
        checkpoint_path = Path(self.output_folder) / checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Scarica se non esiste
        if not checkpoint_path.exists():
            print(f"[INFO] Downloading SAM-HQ checkpoint: {checkpoint_name}")
            url = SAM_HQ_URLS.get(self.sam_hq_model_type, SAM_HQ_URLS["vit_h"])
            download_url_to_file(url, str(checkpoint_path), progress=True)

        # Carica modello SAM-HQ
        self.sam_model_gpu = sam_model_registry[self.sam_hq_model_type](checkpoint=str(checkpoint_path))
        self.sam_model_gpu.to(self.device).eval()
        self.sam_model_cpu = sam_model_registry[self.sam_hq_model_type](checkpoint=str(checkpoint_path)).eval()

        # Inizializza con SAM-HQ
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam_model_gpu,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
            crop_n_layers=0
        )
        self.sam_predictor = SamPredictor(self.sam_model_gpu)

        print(f"[INFO] SAM-HQ {self.sam_hq_model_type} loaded successfully")



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
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).half()
        model.to(self.device)
        model.eval()
        return processor, model

    def _load_yolov8_detector(self, model_path="yolov8x.pt"):
        model = YOLO(model_path)
        model.to(self.device).eval()
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
    # CLIP EMBEDDING  ---------------------------------------------------------
    ###########################################################################
    @torch.inference_mode()
    def _clip_embed_crop(self, image_pil: Image.Image, box: list[float]) -> torch.Tensor:
        """
        Ritaglia l’oggetto, calcola l’embedding CLIP (512-d normalizzato) e
        lo restituisce come tensore CPU.
        """
        x1, y1, x2, y2 = map(int, box)
        crop = image_pil.crop((x1, y1, x2, y2))

        # ─── Assicurati che abbia 3 canali ───────────────────────────────
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        inputs = self.clip_processor(
            images=crop,
            return_tensors="pt"
        ).to(self.device)

        emb = self.clip_model.get_image_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu()          # (1, 512)



    def _dominant_color_lab(self, image_pil, box):
        x1,y1,x2,y2 = map(int, box)
        crop = np.array(image_pil.crop((x1,y1,x2,y2)))
        crop = crop.reshape(-1,3).astype(np.float32)
        if len(crop) < 50:                       # troppi pochi pixel
            return "unknown"
        kmeans = KMeans(n_clusters=3, n_init='auto').fit(crop)
        centers = kmeans.cluster_centers_        # BGR
        labs = cv2.cvtColor(centers[np.newaxis,:,:].astype(np.uint8), cv2.COLOR_BGR2LAB)[0]
        # scorciatoia ruvida: L alto → “white”, a+ → “red”, b+ → “yellow”, ecc.
        lab = labs[np.argmax(kmeans.labels_==kmeans.labels_[0])]
        L,a,b = lab
        if L>200:  return "white"
        if L<50:   return "black"
        if a>140 and b<120: return "red"
        if a<120 and b>140: return "yellow"
        if a<120 and b<120: return "green"
        return "unknown"

    @torch.inference_mode()
    def _relative_depth(self, image_pil, centres):
        """
        Ritorna una lista di profondità normalizzate [0,1] (valore alto = punto vicino).
        Centroidi in coordinate originali → mappati sul depth-map MiDaS ridimensionato.
        """
        # 1) PIL → numpy BGR
        img_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # 2) trasformazione ufficiale MiDaS + forward
        im_t   = self.midas_trans(img_np).to(self.device)
        depth  = self.midas(im_t).squeeze().cpu().numpy()     # shape (H_d, W_d)

        H_d, W_d    = depth.shape            # CORRETTO: prima altezza (rows), poi larghezza
        W_orig, H_orig = image_pil.size      # PIL: (W, H)

        vals = []
        for cx, cy in centres:
            # 3) scala le coordinate dal sistema originale a quello MiDaS
            x_d = int(np.clip(cx / W_orig * W_d, 0, W_d - 1))
            y_d = int(np.clip(cy / H_orig * H_d, 0, H_d - 1))
            vals.append(depth[y_d, x_d])

        vals = np.array(vals, dtype=float)
        if vals.size == 0:
            return [0.5] * len(centres)
        rng = np.ptp(vals)
        if rng < 1e-6:
            return [0.5] * len(vals)

        # 4) normalizza in [0,1]
        vals = (vals - vals.min()) / rng
        return vals.tolist()




    @torch.inference_mode()
    def _blip_caption(self, image_pil):
        inputs = self.blip_proc(images=image_pil, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs, max_new_tokens=20)
        return self.blip_proc.decode(out[0], skip_special_tokens=True)



    ###########################################################################
    # GRAPH CONSTRUCTION
    ###########################################################################

    def _build_scene_graph(self, image_pil, boxes, labels, scores):
        G = nx.DiGraph()                          # nodi indicizzati da 0..N-1
        h, w = image_pil.size[1], image_pil.size[0]

        # ---------- NODI ----------
        for idx, (box, lab, score) in enumerate(zip(boxes, labels, scores)):
            emb_clip = self._clip_embed_crop(image_pil, box)
            x1,y1,x2,y2 = box
            G.add_node(
                idx,
                label       = lab,
                score       = float(score),
                clip_emb    = emb_clip.squeeze().tolist(),      # serializzabile
                bbox_norm   = [x1/w, y1/h, x2/w, y2/h],
                area_norm   = ((x2-x1)*(y2-y1)) / (w*h)
            )
            color = self._dominant_color_lab(image_pil, box)
            G.nodes[idx]["color"] = color                # attributo discorsivo


        centres = [( (b[0]+b[2])/2, (b[1]+b[3])/2 ) for b in boxes]
        depth_vals = self._relative_depth(image_pil, centres)
        for idx, d in enumerate(depth_vals):
            G.nodes[idx]["depth_norm"] = float(d)

        caption = self._blip_caption(image_pil)
        scene_id = len(G)
        G.add_node(scene_id, label="scene", caption=caption)
        for i in range(len(boxes)):
            G.add_edge(scene_id, i)      # arco unidirezionale “descrive”


        # ---------- ARCHI ----------
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j: continue
                # geom features
                dx = centres[j][0] - centres[i][0]
                dy = centres[j][1] - centres[i][1]
                dist = math.hypot(dx,dy) / max(w,h)
                angle = (math.degrees(math.atan2(dy,dx))+360)%360
                iou   = self._compute_iou(boxes[i], boxes[j])

                # CLIP similarity
                sim  = float(torch.matmul(
                              torch.tensor(G.nodes[i]["clip_emb"]),
                              torch.tensor(G.nodes[j]["clip_emb"])
                            ))

                # soglia di pruning (evita grafo completo)
                if dist > 0.4 or iou < 0.01 and sim < 0.20:
                    continue

                depth_delta = depth_vals[j] - depth_vals[i]
                G.add_edge(i, j,
                          dx_norm   = dx / w,
                          dy_norm   = dy / h,
                          dist_norm = dist,
                          angle_deg = angle,
                          iou       = iou,
                          clip_sim  = sim,
                          depth_delta=depth_delta)
        return G

    def _graph_to_pyg(self, G):
        from torch_geometric.data import Data
        node_emb  = torch.tensor([n[1]["clip_emb"] for n in G.nodes(data=True)])
        edge_idx  = torch.tensor(list(G.edges)).t().contiguous()
        edge_attr = torch.tensor([[e[2]["dx_norm"], e[2]["dy_norm"],
                                  e[2]["dist_norm"], e[2]["iou"],
                                  e[2]["clip_sim"]] for e in G.edges(data=True)])
        return Data(x=node_emb, edge_index=edge_idx, edge_attr=edge_attr)

    def _to_prompt(self, G):
        # trova il nodo “scene”
        scene_id = next((n for n,d in G.nodes(data=True) if d.get("label")=="scene"), None)

        nodes_txt = []
        for idx,data in G.nodes(data=True):
            if data["label"] == "scene":
                nodes_txt.append(f'scene:"{data["caption"]}"')
                continue
            desc = f'{data.get("color","")} {data["label"]} (area={data["area_norm"]:.2f})'
            nodes_txt.append(f'{idx}:{desc}')

        edges_txt = []
        for u,v,e in G.edges(data=True):
            if u == scene_id:
                continue
            rel = []
            if e["iou"] > 0.25:
                rel.append("overlaps")
            if abs(e["depth_delta"]) > 0.1:
                rel.append("front_of" if e["depth_delta"] < 0 else "behind")
            if not rel:
                rel.append("near")
            edges_txt.append(f'({u})-{"/".join(rel)}->({v})')

        return "; ".join(nodes_txt + edges_txt)

    def _save_gpickle(self, G, path):
        """
        Salva il grafo in formato pickle (.gpickle) senza fare affidamento
        sulle utility di NetworkX (che cambiano tra le versioni).
        """
        import pickle, gzip, os

        # Assicurati che la cartella esista
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Puoi usare gzip per ridurre lo spazio (facoltativo)
        with gzip.open(path, "wb") if path.endswith(".gz") else open(path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)




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
        boxes  = res["boxes"].cpu()
        scores = res["scores"].cpu()
        labels = res["labels"].cpu()

        # 8) Filter by your threshold and package
        detections = []
        for box, score, lab_idx in zip(boxes, scores, labels):
            if score >= self.threshold_owl:
                detections.append({
                    "box":   box.tolist(),
                    "label": queries[lab_idx],  # map back to your query
                    "score": float(score),
                    "from":  "owlvit"
                })
        # ---------- Test-Time Augmentation FLIP ----------
        flip_pil = self._hflip_pil(image_pil)
        w_flip   = image_pil.size[0]
        flip_det = self._run_owlvit_detection(flip_pil, queries) if False else []
        # (richiamare ricorsivamente è costoso; puoi disattivare o gestire diversamente)
        for d in flip_det:
            x1,y1,x2,y2 = d["box"]
            d["box"] = [ w_flip - x2, y1, w_flip - x1, y2 ]
            detections.append(d)
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
                "score": score_val,
                "from":  "yolo"
            })
        # ---------- TTA flip ----------
        flip_pil = self._hflip_pil(image_pil)
        flip_np  = np.array(flip_pil)
        res_flip = self.yolo_model.predict(flip_np, device=self.device)[0]
        W = image_pil.size[0]
        for b, conf_f, cls_f in zip(res_flip.boxes.xyxy, res_flip.boxes.conf, res_flip.boxes.cls):
            sv = float(conf_f.item())
            if sv < threshold:
                continue
            x1f, y1f, x2f, y2f = b.tolist()
            detections.append({
                "box":   [ W - x2f, y1f, W - x1f, y2f ],
                "label": results.names[int(cls_f.item())],
                "score": sv,
                "from":  "yolo"
            })
        return detections

    @torch.inference_mode()
    def _run_detectron2_detection(self, image_pil: Image.Image):
        if not (self.d2_predictor and self.d2_metadata):
            return []

        # 1) run the model
        image_np = np.array(image_pil)
        outputs  = self.d2_predictor(image_np)
        instances = outputs["instances"].to("cpu")

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
                "det2_mask": mask.astype(bool),
                "from":      "detectron"
            })

        return detections


    @torch.inference_mode()
    def _run_all_detectors(self, image_pil: Image.Image) -> dict:
        """
        Esegue tutti i detector configurati e restituisce le detection unificate.
        """
        all_detections = []
        det_counts = {}

        # OWL-ViT
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
                "printer","monitor","picture","pillow","stone","glasses","wheel","building","bridge","tomato"
            ]
            dets_owl = self._run_owlvit_detection(image_pil, owl_queries)
            det_counts["OWL-ViT"] = len(dets_owl)
            all_detections.extend(dets_owl)

        # YOLOv8
        if "yolov8" in self.detectors_to_use:
            dets_yolo = self._run_yolov8_detection(image_pil)
            det_counts["YOLOv8"] = len(dets_yolo)
            all_detections.extend(dets_yolo)

        # Detectron2
        if "detectron2" in self.detectors_to_use:
            dets_d2 = self._run_detectron2_detection(image_pil)
            det_counts["Detectron2"] = len(dets_d2)
            all_detections.extend(dets_d2)

        return {
            "detections": all_detections,
            "counts": det_counts,
            "boxes": [d["box"] for d in all_detections],
            "labels": [d["label"] for d in all_detections],
            "scores": [d["score"] for d in all_detections]
        }

    @torch.inference_mode()
    def _cached_detection(self, image_pil: Image.Image, cache_key: str = None) -> dict:
        """
        Cache delle predizioni per evitare ricomputo.

        Args:
            image_pil: Immagine PIL da processare
            cache_key: Chiave univoca per la cache (es. hash dell'immagine + parametri)

        Returns:
            Dict con detections, boxes, labels, scores
        """
        if not self.enable_detection_cache or not cache_key:
            return self._run_all_detectors(image_pil)

        # Cache hit
        if cache_key in self._detection_cache:
            print(f"[CACHE] Detection cache hit for key: {cache_key[:16]}...")
            return self._detection_cache[cache_key]

        # Cache miss - esegui detection
        print(f"[CACHE] Detection cache miss for key: {cache_key[:16]}...")
        result = self._run_all_detectors(image_pil)

        # Gestione dimensione cache
        if len(self._detection_cache) >= self.max_cache_size:
            # Rimuovi la voce più vecchia (FIFO)
            oldest_key = next(iter(self._detection_cache))
            del self._detection_cache[oldest_key]
            print(f"[CACHE] Removed oldest cache entry: {oldest_key[:16]}...")

        # Salva in cache
        self._detection_cache[cache_key] = result
        print(f"[CACHE] Cached detection result for key: {cache_key[:16]}...")

        return result


    def _generate_cache_key(self, image_pil: Image.Image, question: str = "") -> str:
        """
        Genera una chiave cache basata su:
        - Hash dell'immagine
        - Parametri dei detector
        - Domanda (se presente)
        """
        import hashlib

        # Hash dell'immagine
        img_bytes = image_pil.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()[:16]

        # Parametri detector
        detector_params = {
            "detectors": sorted(self.detectors_to_use),
            "owl_thresh": self.threshold_owl,
            "yolo_thresh": self.threshold_yolo,
            "det2_thresh": self.threshold_detectron,
            "question": question.strip().lower()
        }

        param_str = str(sorted(detector_params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{img_hash}_{param_hash}"

    ###########################################################################
    # BOX / SEGMENTATION / RELATIONSHIP UTILS
    ###########################################################################
    def _mask_iou(self, m1: np.ndarray, m2: np.ndarray) -> float:
      inter = np.logical_and(m1, m2).sum()
      union = np.logical_or(m1, m2).sum()
      return inter / union if union > 0 else 0.0

    def _close_mask_holes(self, mask: np.ndarray,
                          ksize: int = 7,
                          min_hole_area: int = 100) -> np.ndarray:
        """
        Chiude buchi interni alla maschera:
          1) morph. closing per fessure sottili
          2) riempie i componenti connessi dello sfondo più piccoli di `min_hole_area`
        mask: bool o uint8 (0/1 o 0/255)
        """
        m = mask.astype(np.uint8)
        if m.max() == 255:
            m //= 255

        # 1) closing leggero per chiudere gap sottili
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        # 2) riempi i buchi veri (componenti di background racchiuse)
        inv = 1 - m
        num, lab = cv2.connectedComponents(inv, connectivity=4)
        for lab_id in range(1, num):
            area = np.sum(lab == lab_id)
            if area < min_hole_area:          # è un buco → riempi
                m[lab == lab_id] = 1

        return m.astype(bool)


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

    def _center_distance(self, b1, b2):
        cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
        cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
        return math.hypot(cx2 - cx1, cy2 - cy1)

    def _edge_gap(self, b1, b2):
        # distanza minima tra i bordi dei due box (0 se si sovrappongono)
        gap_x = max(0, max(b1[0] - b2[2], b2[0] - b1[2]))
        gap_y = max(0, max(b1[1] - b2[3], b2[1] - b1[3]))
        return math.hypot(gap_x, gap_y)

    def _is_on_top_of(self,
                      box_a, box_b,
                      mask_a: np.ndarray | None = None,
                      mask_b: np.ndarray | None = None,
                      depth_a: float | None = None,
                      depth_b: float | None = None) -> bool:
        x1a,y1a,x2a,y2a = box_a
        x1b,y1b,x2b,y2b = box_b

        # 1) A deve stare sopra B (centro Y)
        if (y1a + y2a) / 2 >= (y1b + y2b) / 2:
            return False

        # 2) gap verticale piccolo (normalizzato)
        bottom_a, top_b = y2a, y1b
        h_ref = min(y2a - y1a, y2b - y1b)
        tol_px = max(self._on_top_gap_px, int(0.06 * h_ref))  # dinamico
        # Sostituisci questo blocco in _is_on_top_of
        gap = top_b - bottom_a
        if gap > tol_px:
            return False
        # se c'è overlap verticale (gap < 0), accettalo se non è troppo grande
        if gap < 0:
            vert_overlap = min(y2a, y2b) - max(y1a, y1b)  # overlap verticale effettivo
            if vert_overlap / max(1, (y2a - y1a)) > 0.35:  # >35% dell'altezza della bowl → probabilmente non è "sopra"
                return False


        # 3) overlap orizzontale (normalizzato sulla larghezza di A)
        overlap_x = max(0, min(x2a, x2b) - max(x1a, x1b))
        ratio_x = overlap_x / max(1e-6, min((x2a - x1a), (x2b - x1b)))
        if ratio_x < self._on_top_horiz_overlap:
            return False

        # 4) contatto tra maschere (se disponibili)
        if mask_a is not None and mask_b is not None:
            band = max(2, int(0.02 * h_ref))
            H = mask_a.shape[0]
            ya0 = np.clip(int(bottom_a - band), 0, H-1)
            ya1 = np.clip(int(bottom_a + band), 0, H-1)
            yb0 = np.clip(int(top_b - band),    0, H-1)
            yb1 = np.clip(int(top_b + band),    0, H-1)

            band_a = mask_a[ya0:ya1, :]
            band_b = mask_b[yb0:yb1, :]

            contact = np.logical_and(band_a, band_b).any()
            if not contact:
                # prova con una dilatazione leggera
                k = np.ones((3,3), np.uint8)
                if not cv2.bitwise_and(
                    cv2.dilate(band_a.astype(np.uint8), k),
                    cv2.dilate(band_b.astype(np.uint8), k)
                ).any():
                    return False

        # 5) profondità opzionale: A non deve essere molto più lontano della base
        if depth_a is not None and depth_b is not None:
            if depth_a > depth_b + 0.05:  # 5% della scala normalizzata
                return False

        return True

    def _is_below_of(self, box_a, box_b,
                    mask_a: np.ndarray | None = None,
                    mask_b: np.ndarray | None = None,
                    depth_a: float | None = None,
                    depth_b: float | None = None) -> bool:
        # "a is below b" ⇔ "b is on top of a"
        return self._is_on_top_of(
            box_b, box_a,
            mask_b, mask_a,
            depth_b, depth_a
        )


    def _orientation_label(self, dx, dy, margin=5):
        # margin in pixel per evitare rumore
        if abs(dx) >= abs(dy) and abs(dx) > margin:
            return "right_of" if dx > 0 else "left_of"
        if abs(dy) > margin:
            return "below" if dy > 0 else "above"
        return "near"  # fallback

    def _build_precise_nearest_relation(self, i, j, boxes):
        b1, b2 = boxes[i], boxes[j]
        w, h = self.curr_img_size  # impostato in _wbf_fusion
        dist_px = self._center_distance(b1, b2)
        dist_norm = dist_px / max(w, h)

        # orientamento
        cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
        cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
        orient = self._orientation_label(cx2 - cx1, cy2 - cy1, margin=self.margin)

        iou = self._compute_iou(b1, b2)
        gap = self._edge_gap(b1, b2)

        # classi di vicinanza (puoi tarare le soglie)
        if iou > 0.1 or gap <= 3:
            prox = "touching"
        elif dist_norm < 0.05:
            prox = "very_close"
        elif dist_norm < 0.12:
            prox = "close"
        else:
            prox = "near"

        # se prox è 'near' tieni solo orient oppure solo 'near'
        if prox == "near":
            # Se riesci a stimare l'orientamento, usa quello. Altrimenti 'near'.
            relation = orient if orient != "near" else "near"
        else:
            # per le altre prossimità puoi mantenere la combinazione (touching_left_of, etc.)
            relation = f"{prox}_{orient}" if orient != "near" else prox
        return {
            "src_idx": i,
            "tgt_idx": j,
            "relation": relation,
            "distance": dist_px
        }


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

    def _deduplicate_detections(
            self,
            boxes:  list[list[float]],
            labels: list[str],
            scores: list[float],
            iou_thr: float = 0.7
    ):
        """
        NMS per classe *subito dopo* aver fuso le detection dei diversi
        detector, così evitiamo duplicati nelle fasi successive.
        """
        by_lab = defaultdict(list)
        for i, lab in enumerate(labels):
            by_lab[lab.lower()].append(i)

        keep = []
        for idxs in by_lab.values():
            kept = self._non_maximum_suppression(
                [boxes[i]  for i in idxs],
                [scores[i] for i in idxs],
                iou_thr
            )
            keep.extend(idxs[k] for k in kept)

        keep.sort()
        return ([boxes[i]  for i in keep],
                [labels[i] for i in keep],
                [scores[i] for i in keep],
                keep)                           # ultimo serve per filtrare all_detections


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

    def _union_crop(
            self,
            image_pil: Image.Image,
            box_a: list[float],
            box_b: list[float],
    ) -> Image.Image:
        """
        Ritorna un PIL.Image contenente l’area di bounding-box che abbraccia
        *entrambi* i box passati.  Le coordinate sono clampate ai bordi immagine.
        """
        w, h = image_pil.size
        x1 = int(max(0,   min(box_a[0], box_b[0])))
        y1 = int(max(0,   min(box_a[1], box_b[1])))
        x2 = int(min(w-1, max(box_a[2], box_b[2])))
        y2 = int(min(h-1, max(box_a[3], box_b[3])))

        # garantisco rettangolo valido
        if x2 <= x1 or y2 <= y1:
            # fallback: 1 × 1 pixel
            return image_pil.crop((0, 0, 1, 1)).copy()

        return image_pil.crop((x1, y1, x2, y2)).convert("RGB")



    ###########################################################################
    # RELATIONSHIP INFERENCE
    ###########################################################################
    def _infer_relationships_improved(self, image_pil, boxes, labels, masks=None, depths=None):
        if labels is None:
            labels = [f"obj{i}" for i in range(len(boxes))]

        n = len(boxes)
        centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]

        # Testi per CLIP (subj/obj verranno inseriti da _clip_relation)
        REL_TEMPLATES = [
            "on top of",
            "under",
            "inside",
            "holding",
            "riding",
            "touching",
            "next to",
            "in front of",
            "behind",
        ]

        CANON = {
            "on top of": "on_top_of",
            "under": "below",
            "inside": "inside",
            "holding": "holding",
            "riding": "riding",
            "touching": "touching",
            "next to": "next_to",
            "in front of": "in_front_of",
            "behind": "behind",
        }

        def canon_rel(txt: str) -> str:
            t = re.sub(r"\s+", " ", txt.lower().strip())
            return CANON.get(t, t.replace(" ", "_"))

        rels = []

        # ---------- 1) geometrico: on_top_of preciso ----------
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._is_on_top_of(
                    boxes[i], boxes[j],
                    mask_a=masks[i]["segmentation"] if masks else None,
                    mask_b=masks[j]["segmentation"] if masks else None,
                    depth_a=depths[i] if depths else None,
                    depth_b=depths[j] if depths else None
                ):
                    dist_ij = self._center_distance(boxes[i], boxes[j])
                    # on_top_of
                    rels.append({
                        "src_idx": i, "tgt_idx": j,
                        "relation": "on_top_of",
                        "distance": dist_ij
                    })
                    # below (speculare)
                    rels.append({
                        "src_idx": j, "tgt_idx": i,
                        "relation": "below",
                        "distance": dist_ij
                    })


        # ---------- 2) geometrico: above/below/left/right ----------
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                dx, dy = cx2 - cx1, cy2 - cy1
                dist = math.hypot(dx, dy)
                if dist < self.min_distance or dist > self.max_distance:
                    continue
                if abs(dy) >= abs(dx) and abs(dy) > self.margin:
                    relation = "above" if dy < 0 else "below"
                elif abs(dx) > self.margin:
                    relation = "right_of" if dx > 0 else "left_of"
                else:
                    continue
                rels.append({
                    "src_idx": i,
                    "tgt_idx": j,
                    "relation": relation,
                    "distance": dist
                })

        # ---------- 3) CLIP fallback ----------
        for i, box_i in enumerate(boxes):
            for j, box_j in enumerate(boxes):
                if i == j:
                    continue
                union = self._union_crop(image_pil, box_i, box_j)
                lab_i = labels[i].rsplit("_", 1)[0]
                lab_j = labels[j].rsplit("_", 1)[0]

                rel_txt, score = self._clip_relation(union, lab_i, lab_j, REL_TEMPLATES)
                if score > 0.23:
                    dist = math.hypot(centers[j][0] - centers[i][0],
                                      centers[j][1] - centers[i][1])
                    rels.append({
                        "src_idx": i,
                        "tgt_idx": j,
                        "relation": canon_rel(rel_txt),
                        "relation_raw": rel_txt,
                        "clip_sim": score,
                        "distance": dist
                    })

        # ---------- 4) ConceptNet ----------
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                extra = self._conceptnet_edges(labels[i], labels[j])
                for e in extra:
                    e.update({"src_idx": i, "tgt_idx": j})
                rels.extend(extra)

        return rels



    # ------------------------------------------------------------------
    #  RELAZIONI DA CONCEPTNET
    # ------------------------------------------------------------------


    def _camel_to_snake(self, s: str) -> str:
        """'PartOf' -> 'part_of'  |  'UsedFor' -> 'used_for' """
        return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

    def _conceptnet_edges(
            self,
            label_a: str,
            label_b: str,
            max_edges: int = 2,
    ) -> list[dict]:
        """
        Ritorna al massimo `max_edges` relazioni ConceptNet fra `label_a` → `label_b`.
        Se l’API non risponde, cache FAIL e ritorna [].
        """
        key = tuple(sorted((label_a, label_b)))

        # skip se abbiamo già fallito
        if self.cnet_skip_on_fail and _CONCEPTNET_CACHE.get(key) == "FAIL":
            print("[WARN] ConceptNet failed to load.")
            return []

        # ---------- cache hit ----------------------------------------
        if key in _CONCEPTNET_CACHE and _CONCEPTNET_CACHE[key] != "FAIL":
            edges = _CONCEPTNET_CACHE[key]
        else:
            # ---------- chiamata HTTP ---------------------------------
            def _uri(term: str) -> str:
                return f"/c/en/{urllib.parse.quote(term.lower().replace(' ', '_'))}"

            url = (
                "https://api.conceptnet.io/query"
                f"?node={_uri(label_a)}&other={_uri(label_b)}&limit=50"
            )

            edges = None
            for attempt in range(1, self.cnet_max_retry + 1):
                try:
                    r = requests.get(url, timeout=self.cnet_timeout)
                    r.raise_for_status()
                    edges = r.json().get("edges", [])
                    break
                except Exception as e:
                    if attempt == self.cnet_max_retry:
                        print(f"[WARN] ConceptNet ({label_a},{label_b}) failed: {e}")
                    time.sleep(0.3 * (2 ** (attempt - 1)))

            if edges is None:
                _CONCEPTNET_CACHE[key] = "FAIL"
                return []

            _CONCEPTNET_CACHE[key] = edges

        # ---------- estrai tutte le relazioni ------------------------
        chosen = []
        for e in edges:
            # manteniamo l’orientamento A → B
            if e["start"]["label"].lower() != label_a.lower() or \
               e["end"]["label"].lower()   != label_b.lower():
                continue

            rel_type  = e["rel"]["label"]          # es. "PartOf", "UsedFor"…
            rel_snake = self._camel_to_snake(rel_type)  # "part_of", "used_for"
            weight    = e.get("weight", 1.0)

            chosen.append((rel_snake, rel_type, weight))

        # ordina per peso (più forte → prima) e taglia
        chosen.sort(key=lambda t: -t[2])
        return [
            {
                "relation":      f"cnet_{snk}",   # usato internamente per i filtri
                "relation_raw":  raw,             # testo originale ConceptNet
                "distance":      9_999
            }
            for snk, raw, _ in chosen[:max_edges]
        ]




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
                    rel_nearest = self._build_precise_nearest_relation(i, best_j, boxes)

                    if all(rr["tgt_idx"] != best_j for rr in rels_by_src[i]):
                        rels_by_src[i].append(rel_nearest)

        final_list = []
        for src_idx, rlist in rels_by_src.items():
            # Priorità: prima quelle citate nella domanda, poi per distanza
            sorted_rlist = sorted(
                rlist,
                key=lambda r: (
                    0 if self._is_question_relation(r["relation"]) else 1,
                    r.get("distance", 1e9)
                )
            )
            final_list.extend(sorted_rlist[: self.max_relations_per_object])
        return final_list

    def _unify_pair_relations(self, relationships: list) -> list:
        best_for_pair = {}
        for r in relationships:
            i = r["src_idx"]
            j = r["tgt_idx"]
            pair = (i, j)
            if pair not in best_for_pair:
                best_for_pair[pair] = r
            else:
                if r.get("distance", 1e9) < best_for_pair[pair].get("distance", 1e9):
                    best_for_pair[pair] = r
        return list(best_for_pair.values())

    def _drop_inverse_duplicates(self, relationships: list[dict]) -> list[dict]:
        """
        Elimina relazioni ridondanti/inverse: se esiste A->B = 'in_front_of',
        scarta B->A = 'behind' (e viceversa), idem per left/right, above/below, ecc.
        Tiene la prima che trova (puoi cambiare la logica di priorità se vuoi).
        """
        inverse = {
            "left_of":      "right_of",
            "right_of":     "left_of",
            "above":        "below",
            "below":        "above",
            "in_front_of":  "behind",
            "behind":       "in_front_of",
            "on_top_of":    "below",   # opzionale, togli se non vuoi legarla a 'below'
            "under":        "on_top_of"  # se usi 'under' da qualche parte
        }

        kept = []
        seen = set()  # memorizza triple (src, tgt, rel)
        for r in relationships:
            i, j, rel = r["src_idx"], r["tgt_idx"], r["relation"]
            inv_rel = inverse.get(rel)
            # se esiste già l'inverso, salta
            if inv_rel and (j, i, inv_rel) in seen:
                continue
            seen.add((i, j, rel))
            kept.append(r)
        return kept


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


    @torch.inference_mode()
    def _clip_relation(self, crop_img, lab_i, lab_j, rel_templates):
        # 1) encode image crop
        inputs = self.clip_processor(images=crop_img, return_tensors="pt").to(self.device)
        im_feat = self.clip_model.get_image_features(**inputs)
        im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True)

        # 2) encode candidate texts in batch
        texts = [tmpl.format(subj=lab_i, obj=lab_j) for tmpl in rel_templates]
        txt_inputs = self.clip_processor(text=texts, return_tensors="pt",
                                        padding=True).to(self.device)
        txt_feat = self.clip_model.get_text_features(**txt_inputs)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # 3) similarity & arg-max
        sims = torch.matmul(im_feat, txt_feat.T).squeeze(0)   # shape [len(R)]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        return rel_templates[best_idx], best_sim

    def _filter_relationships_by_question(self, relationships):
        """
        Filters a list of relationship dicts based on question-specified relation terms
        and a spaCy similarity threshold.
        """
        # Se il filtro è disattivato o la domanda non contiene relazioni → non filtrare
        if (not self.filter_relations_by_question) or (not self._parsed_question_relation_terms):
            return relationships

        filtered = []
        question_terms = self._parsed_question_relation_terms

        for r in relationships:
            rel_label = r["relation"]

            # 1) Exact match against parsed question terms
            if rel_label in question_terms:
                filtered.append(r)
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

    def _is_question_relation(self, rel_label: str) -> bool:
        if not self._parsed_question_relation_terms:
            return False
        if rel_label in self._parsed_question_relation_terms:
            return True
        rel_doc = self.nlp(rel_label.replace("_", " "))
        return any(
            rel_doc.similarity(self.nlp(term)) >= self.threshold_relation_similarity
            for term in self._parsed_question_relation_terms
        )



    ###########################################################################
    # VISUALIZATION
    ###########################################################################

    def _shrink_segment_px(self, p0, p1, shrink_px, ax):
        """
        Accorcia il segmento p0->p1 di 'shrink_px' pixel ad entrambe le estremità.
        Ritorna due punti in coordinate dati (data coords).
        """
        to_px   = ax.transData.transform
        to_data = ax.transData.inverted().transform

        P0 = np.array(to_px(p0))
        P1 = np.array(to_px(p1))
        v  = P1 - P0
        L  = np.linalg.norm(v)
        if L < 1:
            return p0, p1
        v_norm = v / L
        P0n = P0 + v_norm * shrink_px
        P1n = P1 - v_norm * shrink_px
        return tuple(to_data(P0n)), tuple(to_data(P1n))


    # --- utilities ---------------------------------
    # -------------------------------------------------------------
    def _pad_img_one_pixel(self, img_np: np.ndarray) -> np.ndarray:
        """Edge-replicate padding so shape becomes (H+2, W+2, C)."""
        return np.pad(img_np, ((1, 1), (1, 1), (0, 0)), mode="edge")
    # -------------------------------------------------------------


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

    # ------------------------------------------------------------------
    #  Colori: un hue per ogni classe, distribuiti col golden-ratio
    # ------------------------------------------------------------------

    def _boost_color(self, hex_col, sat_factor=1.3, val_factor=1.15):
        import matplotlib.colors as mcolors, colorsys
        r, g, b = mcolors.to_rgb(hex_col)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = min(1.0, s * sat_factor)
        v = min(1.0, v * val_factor)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return mcolors.to_hex((r, g, b))

    def _color_for_label(self, lab: str, idx: int = 0) -> str:
        """
        Restituisce un colore fisso per ogni classe usando BASIC_COLORS in ciclo.
        Niente contrasti, cmap, neon, ecc.
        """
        base_lab = lab.rsplit("_", 1)[0].lower()  # togli suffissi tipo _1, _2
        if not hasattr(self, "_label2col"):
            self._label2col = {}
        if base_lab not in self._label2col:
            raw = BASIC_COLORS[len(self._label2col) % len(BASIC_COLORS)]
            sat = self.config.get("color_sat_boost", 1.3)
            val = self.config.get("color_val_boost", 1.15)
            self._label2col[base_lab] = self._boost_color(raw, sat, val)
        return self._label2col[base_lab]

    def _text_color_for_bg(self, hex_col: str) -> str:
        """
        Scelta minimale: testo bianco se lo sfondo è scuro, nero se è chiaro.
        (Niente WCAG avanzato)
        """
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(hex_col)
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        return '#000000' if luminance > 0.6 else '#ffffff'

    def _pick_obj_color(self, image_pil, box, mask_dict, label, idx):
        """Versione minimale: usa direttamente _color_for_label."""
        return self._color_for_label(label, idx)


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

    def _pixels_to_data(self, ax, dx_px, dy_px):
        inv = ax.transData.inverted()
        x0, y0 = inv.transform((0, 0))
        x1, y1 = inv.transform((dx_px, dy_px))
        return x1 - x0, y1 - y0

    def _check_text_overlaps(self, ax, texts, pad_px=2):
        """Ritorna True se ci sono overlap tra text artist."""
        if len(texts) < 2:
            return False
        fig = ax.figure
        fig.canvas.draw_idle()
        renderer = fig.canvas.get_renderer()
        bbs = [t.get_window_extent(renderer=renderer).expanded(1.0, 1.0) for t in texts]
        for i in range(len(bbs)):
            for j in range(i+1, len(bbs)):
                if bbs[i].expanded(1.0,1.0).overlaps(bbs[j].expanded(1.0,1.0)):
                    return True
        return False

    def _arrow_bboxes_px(self, arrows, renderer):
        bbs = []
        for a in arrows:
            try:
                path = a.get_path().transformed(a.get_transform())
                bb = path.get_extents()
                # bb è in data coords → converti in pixel
                bb_px = bb.transformed(a.axes.transData + a.figure.dpi_scale_trans)
                bbs.append(bb_px)
            except Exception:
                pass
        return bbs


    def _final_overlap_fix(self, ax, texts, anchors, rel_mask, avoid_artists=None):
        """
        Sistema le sovrapposizioni tra testi e (opzionalmente) altri artist (frecce).
        avoid_artists: lista di patch/artist da evitare (es. frecce).
        """
        from adjustText import adjust_text
        if avoid_artists is None:
            avoid_artists = []

        # 1) primo passaggio con adjust_text includendo le frecce
        adjust_text(
            texts,
            x=[p[0] for p in anchors],
            y=[p[1] for p in anchors],
            ax=ax,
            only_move={"points": "y", "text": "xy"},
            force_text=0.8,
            expand_text=(1.35, 1.35),
            expand_points=(1.25, 1.25),
            expand_objects=(1.25, 1.25),
            add_objects=avoid_artists,
            arrowprops=None
        )

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # 2) loop di micro-push: testi↔testi e testi↔frecce
        for _ in range(60):  # max iterazioni
            moved = False

            # bbox dei testi in pixel
            text_bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for t in texts]
            # bbox delle frecce
            arrow_bbs = self._arrow_bboxes_px(avoid_artists, renderer)

            # --- collisioni testo ↔ testo
            for i in range(len(text_bbs)):
                for j in range(i + 1, len(text_bbs)):
                    if text_bbs[i].overlaps(text_bbs[j]):
                        # sposta preferibilmente il testo della relazione (rel_mask=True)
                        target = j if rel_mask[j] else i
                        src_bb = text_bbs[i] if target == j else text_bbs[j]
                        tgt_bb = text_bbs[target]

                        dx_px = (tgt_bb.x1 - src_bb.x0) * 0.15
                        dy_px = (tgt_bb.y1 - src_bb.y0) * 0.15
                        dx, dy = self._pixels_to_data(ax, dx_px, dy_px)
                        x, y = texts[target].get_position()
                        texts[target].set_position((x + dx, y + dy))
                        moved = True

            # --- collisioni testo ↔ freccia
            for k, bb in enumerate(text_bbs):
                for abb in arrow_bbs:
                    if bb.overlaps(abb):
                        dx_px = (bb.x1 - abb.x0) * 0.12
                        dy_px = (bb.y1 - abb.y0) * 0.12
                        dx, dy = self._pixels_to_data(ax, dx_px, dy_px)
                        x, y = texts[k].get_position()
                        texts[k].set_position((x + dx, y + dy))
                        moved = True

            if not moved:
                break
            fig.canvas.draw_idle()
            # ricalcola bbs aggiornati
            text_bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for t in texts]



    def _visualize_detections_and_relationships_with_auto_masks(
        self,
        image: Image.Image,
        boxes: list,
        labels: list,
        scores: list,
        relationships: list,
        all_masks: list,
        save_path: str = None,
        draw_background: bool = True,
        bg_color=(1, 1, 1, 0)   # RGBA: 0 alpha = trasparente
    ):
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(10, 8))
        if draw_background:
            ax.imshow(image)
            ax.axis("off")
        else:
            W, H = image.size
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)   # mantiene l’origine in alto a sinistra, come imshow
            ax.axis("off")
            ax.set_facecolor(bg_color)
            # se alpha=0 rendo anche il fig trasparente
            if len(bg_color) == 4 and bg_color[3] == 0:
                fig.patch.set_alpha(0)

        placed_positions = []
        overlap_threshold = 30
        obj_colors = [
            self._pick_obj_color(image, boxes[i],
                                 all_masks[i] if i < len(all_masks) else None,
                                 labels[i], i)
            for i in range(len(boxes))
        ]

        # ---------- label style ----------
        if self.label_mode == "numeric":
            vis_labels = [str(i + 1) for i in range(len(boxes))]
        elif self.label_mode == "alphabetic":
            vis_labels = list(string.ascii_uppercase[: len(boxes)])
        else:
            vis_labels = labels

        detection_labels_info = []
        arrow_counts = defaultdict(int)

        rel_texts = []
        rel_anchor_pts = []

        # ---------- draw detections ----------
        for i, box in enumerate(boxes):
            color = obj_colors[i]
            x_min, y_min, x_max, y_max = map(int, box)
            center_pt = ((x_min + x_max) // 2, (y_min + y_max) // 2)

            best_mask = self._get_best_mask_for_box(box, all_masks)

            if self.show_segmentation and best_mask is not None:
                mask_bool  = best_mask["segmentation"]
                mask_uint8 = (mask_bool.astype(np.uint8)) * 255

                # --- chiudi i buchi della maschera -----------------------------------
                if self.config.get("close_holes", True):
                    k = int(self.config.get("hole_kernel", 5))
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    # small gaps
                    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

                    # fill real holes with flood-fill
                    inv = 255 - mask_uint8
                    h, w = mask_uint8.shape
                    flood = inv.copy()
                    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(flood, ff_mask, (0, 0), 0)
                    holes = cv2.bitwise_and(inv, flood)
                    mask_uint8 = cv2.bitwise_or(mask_uint8, holes)

                # --- trova contorni (interni ed esterni) ------------------------------
                contours, hierarchy = cv2.findContours(
                    mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours and len(contours) > 0:
                    # contorno principale per calcolare il centro e spostare il testo
                    areas = [cv2.contourArea(c) for c in contours]
                    idx_main = int(np.argmax(areas))
                    main_cnt = contours[idx_main].squeeze()

                    # disegna tutti i contorni
                    for cnt in contours:
                        cnt = cnt.squeeze()
                        if cnt.ndim != 2 or len(cnt) < 3:
                            continue

                        if self.fill_segmentation:
                            ax.fill(
                                cnt[:, 0], cnt[:, 1],
                                color=color,
                                alpha=self.seg_fill_alpha,
                                zorder=1
                            )

                        ax.plot(
                            cnt[:, 0], cnt[:, 1],
                            color=color,
                            linewidth=self.bbox_linewidth,
                            alpha=0.95,
                            zorder=2
                        )

                    # centroide della maschera piena
                    M = cv2.moments(mask_uint8, binaryImage=True)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center_pt = (cx, cy)

                    # sposta il punto fuori dal contorno principale
                    if main_cnt.ndim == 2 and len(main_cnt) >= 3:
                        center_pt = self._move_point_outside_contour(center_pt, main_cnt)

                else:
                    # fallback: disegna solo il rettangolo
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=self.bbox_linewidth,
                        edgecolor=color,
                        facecolor="none",
                        zorder=2,
                    )
                    ax.add_patch(rect)
            elif self.show_bboxes:
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=self.bbox_linewidth,
                    edgecolor=color,
                    facecolor="none",
                    zorder=2,
                )
                ax.add_patch(rect)

            # ---------- etichetta compatta ----------
            area_px = (x_max - x_min) * (y_max - y_min)
            lbl_txt = (
                f"{vis_labels[i]} ({scores[i]*100:.0f}%)"
                if self.show_confidence else vis_labels[i]
            )

            if self.display_labels and area_px > 9000:
                font_col = self._text_color_for_bg(color)
                ax.text(
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    lbl_txt,
                    ha="center", va="center",
                    fontsize=self.obj_fs_in,
                    color=font_col,
                    bbox=dict(facecolor=color, alpha=0.6, lw=0),
                    zorder=7,
                )
            elif self.display_labels:
                detection_labels_info.append((center_pt, lbl_txt, color))

            center_pt = self._adjust_position(center_pt, placed_positions, overlap_threshold)
            placed_positions.append(center_pt)


        # ---------- draw relationships ----------
        # ---------- PREPARA le relazioni (NO frecce ora) ----------
        arrow_patches = []
        rel_draw_data = []      # info per disegnare le frecce dopo
        rel_texts = []
        rel_anchor_pts = []

        if self.display_relationships and len(relationships) > 0:
            centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            arrow_counts = defaultdict(int)

            SPATIAL_KEYS = (
                "left_of", "right_of", "above", "below",
                "on_top_of", "under", "in_front_of", "behind"
            )

            for rel in relationships:
                # indici originali
                s0, t0 = rel["src_idx"], rel["tgt_idx"]

                # nome relazione in minuscolo
                rel_name = rel.get("relation", "").lower()

                # se è una relazione spaziale (anche se prefissata da 'touching_', 'close_', ecc.)
                # inverti la direzione in modo che la freccia punti verso l'oggetto di riferimento
                if any(k in rel_name for k in SPATIAL_KEYS):
                    s, t = t0, s0
                else:
                    s, t = s0, t0

                if s >= len(centers) or t >= len(centers):
                    continue

                color = obj_colors[s]
                arrow_counts[(s, t)] += 1
                rad_offset = 0.2 + 0.1 * (arrow_counts[(s, t)] - 1)

                # punto medio usato come anchor iniziale del testo
                start_x, start_y = centers[s]
                end_x, end_y     = centers[t]
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                # offset per le curve (come avevi già)
                if rad_offset != 0:
                    dx = end_x - start_x
                    dy = end_y - start_y
                    length = math.hypot(dx, dy)
                    if length > 0:
                        perp_x = -dy / length
                        perp_y =  dx / length
                        offset_distance = 15 * (1 if rad_offset > 0 else -1)
                        mid_x += perp_x * offset_distance
                        mid_y += perp_y * offset_distance

                # testo relazione "pulito"
                raw = rel.get('relation_raw', rel['relation'])
                if raw.startswith('cnet_'): raw = raw[5:]
                if re.search(r'[A-Z]', raw):
                    raw = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw)
                else:
                    raw = raw.replace('_', ' ')
                rel_text_str = raw.strip().title()

                # Creo l'oggetto testo ORA (così può essere spostato da adjust_text)
                t_rel = ax.text(
                    mid_x, mid_y, rel_text_str,
                    fontsize=self.rel_fs,
                    ha='center', va='center',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85,
                              edgecolor=color, linewidth=0.6),
                    zorder=5
                )

                rel_texts.append(t_rel)
                rel_anchor_pts.append(((start_x + end_x) / 2, (start_y + end_y) / 2))

                # Salvo tutto quello che mi serve per le frecce
                rel_draw_data.append({
                    "src_pt": centers[s],
                    "tgt_pt": centers[t],
                    "color": color,
                    "rad": rad_offset
                })

        # ---------- testo esterno (solo quelli che non stavano nel box) ----------
        texts = []
        anchor_pts = []
        for (pt, text, color) in detection_labels_info:
            font_col = self._text_color_for_bg(color)  # testo leggibile sul colore di sfondo
            t = ax.text(
                pt[0], pt[1], text,
                fontsize=self.obj_fs_out,
                color=font_col,
                bbox=dict(facecolor=color, alpha=0.6, lw=0),  # identico agli interni
                zorder=7,
            )

            texts.append(t)
            anchor_pts.append(pt)

        # dopo aver creato texts (etichette oggetti) e rel_texts (etichette relazioni):
        all_texts   = texts + rel_texts
        all_anchors = anchor_pts + rel_anchor_pts
        rel_mask    = [False]*len(texts) + [True]*len(rel_texts)

        # Ordina: prima le etichette delle relazioni (True), poi quelle degli oggetti (False)
        order     = np.argsort(rel_mask)[::-1]
        t_fix     = [all_texts[i]   for i in order]
        a_fix     = [all_anchors[i] for i in order]
        rm_fix    = [rel_mask[i]    for i in order]

        # Serve per avere i bbox aggiornati
        fig.canvas.draw()   # non draw_idle

        if self.resolve_overlaps:
            self._final_overlap_fix(ax, t_fix, a_fix, rm_fix, avoid_artists=None)

        # ---------- ORA disegna le frecce ----------
        if self.display_relationships and rel_draw_data:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # opzionale: accorcia di qualche pixel per non incollarsi ai testi
            SHRINK_PX = 6

            for d in rel_draw_data:
                p0, p1 = d["src_pt"], d["tgt_pt"]
                # accorcia il segmento alle estremità (evita di entrare nei box di testo)
                p0, p1 = self._shrink_segment_px(p0, p1, SHRINK_PX, ax)

                arrow = patches.FancyArrowPatch(
                    p0, p1,
                    arrowstyle="->",
                    color=d["color"],
                    linewidth=self.rel_arrow_lw,
                    connectionstyle=f"arc3,rad={d['rad']}",
                    mutation_scale=self.rel_arrow_ms,
                    zorder=4
                )
                ax.add_patch(arrow)
                arrow_patches.append(arrow)

        # 3) ridisegna le linee che collegano posizione finale del testo al suo anchor
        for t, pt, is_rel in zip(all_texts, all_anchors, rel_mask):
            ax.annotate(
                "", xy=pt, xytext=t.get_position(),
                arrowprops=dict(arrowstyle="-", color="gray",
                                alpha=0.45, shrinkA=4, shrinkB=4,
                                linewidth=1, linestyle="dotted" if is_rel else "-"),
                zorder=6,
            )


        # ---------- legenda rapida --------------------------------------
        if self.display_legend:
            uniq_base = sorted({lab.rsplit('_', 1)[0] for lab in labels})
            handles = [
                patches.Patch(color=self._color_for_label(lb, 0), label=lb)
                for lb in uniq_base[:10]          # max 10 voci
            ]
            if handles:
                ax.legend(handles=handles, fontsize=self.legend_fs, loc="upper right")


        plt.tight_layout()
        if save_path:
            plt.savefig(
                save_path,
                bbox_inches="tight",
                transparent=(not draw_background and (len(bg_color)==4 and bg_color[3]==0))
            )
            plt.close(fig)
            print(f"[INFO] Saved output to {save_path}")
        else:
            plt.show()

        uniq = len(boxes)
        rel  = len(relationships)
        print(f"[STATS] oggetti: {uniq}, relazioni: {rel}")



    def _refine_mask_with_point(self, image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refina `mask` con un singolo punto centrale (foreground).
        Funziona con SAM-1, SAM-2 e SAM-HQ.
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return mask

        H, W = image_np.shape[:2]
        if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
            return mask

        # centroide reale della mask
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(xs.mean()), int(ys.mean())
        point_coords = np.array([[cx, cy]])
        point_labels = np.array([1], dtype=int)

        # Scegli il predictor corretto
        if self.sam_version == "hq":
            predictor = self.sam_predictor  # SAM-HQ usa lo stesso interface
        elif self.sam_version == "2":
            predictor = self.sam2_predictor
        else:
            predictor = self.sam_predictor  # SAM-1

        try:
            predictor.set_image(image_np)
            masks_pts, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            return masks_pts[0]
        except Exception as e:
            print("[WARN] point-prompt skipped:", e)
            return mask



    @torch.inference_mode()
    def _run_sam_combined_segmentation(
        self,
        image_pil: Image.Image,
        boxes: list[list[float]],
    ):
        """
        Restituisce una lista di dict:
            {"segmentation": np.ndarray(bool,H,W),
            "bbox":         box,
            "predicted_iou": float}
        Compatibile sia con SAM1 che con SAM2.
        """

        image_np = np.array(image_pil)

        # ══════════════════════════════════════════════════════════════
        # ─── BRANCH SAM-HQ ────────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════
        if self.sam_version == "hq":
            self.sam_predictor.set_image(image_np)

            try:
                refined_masks = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Prova prima con bounding box
                    try:
                        box_arr = np.array([[x1, y1, x2, y2]], dtype=float)
                        masks_box, scores_box, _ = self.sam_predictor.predict(
                            box=box_arr,
                            multimask_output=False
                        )
                        mask = masks_box[0]
                        score = float(scores_box[0])

                        # Se la maschera è troppo piccola, prova con punto centrale
                        if mask.sum() < 50:
                            point_coords = np.array([[cx, cy]])
                            point_labels = np.array([1])
                            masks_pt, scores_pt, _ = self.sam_predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=False
                            )
                            mask = masks_pt[0]
                            score = float(scores_pt[0])

                        if self.config.get("close_holes", False):
                            mask = self._close_mask_holes(
                                mask,
                                ksize=self.config.get("hole_kernel", 7),
                                min_hole_area=self.config.get("min_hole_area", 100)
                            )
                        refined_masks.append({
                            "segmentation": mask.astype(bool),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "predicted_iou": score,
                        })

                    except Exception as e:
                        print(f"[WARN] SAM-HQ failed on box {box}: {e}")
                        refined_masks.append({
                            "segmentation": np.zeros(image_np.shape[:2], dtype=bool),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "predicted_iou": 0.0,
                        })

                return refined_masks

            finally:
                self.sam_predictor.reset_image()
                if hasattr(self.sam_predictor, "features"):
                    del self.sam_predictor.features
                torch.cuda.empty_cache()

        # ══════════════════════════════════════════════════════════════
        # ─── BRANCH SAM 2 ─────────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════

        elif self.sam_version == "2":
            image_np = np.array(image_pil)

            # 1) carico l'immagine nel predictor una volta sola
            self.sam2_predictor.set_image(image_np)

            refined_masks = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # centro del bbox = punto foreground
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                point_coords = None            # ← disabilitiamo il punto
                point_labels = None
                input_box    = np.array([[x1, y1, x2, y2]])

                try:
                    # prima tentiamo SOLO col bounding-box
                    masks, scores, _ = self.sam2_predictor.predict(
                        box=input_box,
                        multimask_output=False,
                    )

                    # se la maschera è troppo piccola/strana ritentiamo col punto
                    if masks[0].sum() < 30:     # area in pixel – soglia da tarare
                        point_coords = np.array([[cx, cy]])
                        point_labels = np.array([1], dtype=int)
                        masks, scores, _ = self.sam2_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=False,
                        )
                    m = masks[0]
                    bbox_xywh = [x1, y1, (x2 - x1), (y2 - y1)]

                    if self.config.get("close_holes", False):
                        m = self._close_mask_holes(
                            m,
                            ksize=self.config.get("hole_kernel", 7),
                            min_hole_area=self.config.get("min_hole_area", 100)
                        )

                    refined_masks.append({
                        "segmentation": m.astype(bool),
                        "bbox":         bbox_xywh,
                        "predicted_iou": float(scores[0]) if scores is not None else 1.0,
                    })
                except Exception as e:
                    # fallback: bbox vuoto → maschera piena = 0
                    print(f"[WARN] SAM-2 point-prompt failed on box {box}: {e}")
                    refined_masks.append({
                        "segmentation": np.zeros(image_np.shape[:2], dtype=bool),
                        "bbox": [x1, y1, x2, y2],
                        "predicted_iou": 0.0,
                    })

            # pulizia RAM GPU
            if hasattr(self.sam2_predictor, "features"):
                del self.sam2_predictor.features
            torch.cuda.empty_cache()
            return refined_masks


        # ══════════════════════════════════════════════════════════════
        # ─── BRANCH SAM 1  (codice originale quasi invariato) ─────────
        # ══════════════════════════════════════════════════════════════

        else:
            image_np = np.array(image_pil)
            self.sam_predictor.set_image(image_np)

            H, W = image_np.shape[:2]

            try:
                refined_masks = []
                # ---------- 1) “clamp” iniziale dei box ----------
                tmp_boxes = []
                for bx in boxes:
                    x1, y1, x2, y2 = bx
                    x1 = max(0,   min(int(round(x1)), W - 2))
                    y1 = max(0,   min(int(round(y1)), H - 2))
                    x2 = max(x1 + 1, min(int(round(x2)), W - 1))
                    y2 = max(y1 + 1, min(int(round(y2)), H - 1))
                    tmp_boxes.append([x1, y1, x2, y2])

                # ---------- 2) per ogni box, tentativi shrink 0-2-4-8-… -----
                for box0 in tmp_boxes:
                    for shrink in (0, 2, 4, 8, 12, 16):
                        x1, y1, x2, y2 = box0
                        xs = max(0,   x1 + shrink)
                        ys = max(0,   y1 + shrink)
                        xe = max(xs + 1, x2 - shrink)
                        ye = max(ys + 1, y2 - shrink)
                        if xe <= xs or ye <= ys:
                            continue
                        try:
                            box_arr = np.asarray([xs, ys, xe, ye], dtype=float)[None, :]
                            masks_box, scores_box, _ = self.sam_predictor.predict(
                                box=box_arr, multimask_output=True
                            )
                            best = int(np.argmax(scores_box))
                            final_mask = self._refine_mask_with_point(image_np, masks_box[best])
                            if self.config.get("close_holes", False):
                                final_mask = self._close_mask_holes(
                                    final_mask,
                                    ksize=self.config.get("hole_kernel", 7),
                                    min_hole_area=self.config.get("min_hole_area", 100)
                                )
                            refined_masks.append({
                                "segmentation": final_mask.astype(bool),
                                "bbox":         [xs, ys, xe, ye],
                                "predicted_iou": float(scores_box[best]),
                            })
                            break          # → box ok, passa al prossimo
                        except cv2.error:
                            continue       # riprova con shrink maggiore
                    else:
                        print(f"[WARN] SAM-1 skipped problematic box: {box0!r}")

                return refined_masks

            finally:
                self.sam_predictor.reset_image()
                if hasattr(self.sam_predictor, "features"):
                    del self.sam_predictor.features
                torch.cuda.empty_cache()

    ###########################################################################
    # CACHE
    ###########################################################################

    def get_cache_stats(self) -> dict:
        """Restituisce statistiche sulla cache"""
        return {
            "detection_cache_size": len(self._detection_cache),
            "det_cache_size": len(self._det_cache),
            "max_cache_size": self.max_cache_size,
            "cache_enabled": self.enable_detection_cache
        }

    def clear_caches(self):
        """Pulisce tutte le cache"""
        self._detection_cache.clear()
        self._det_cache.clear()
        self._global_model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] All caches cleared")

    ###########################################################################
    # PROCESS A SINGLE IMAGE
    ###########################################################################
    def _canon_label(self, lab: str) -> str:
        base = lab.rsplit("_", 1)[0].lower()            # togli eventuale _1, _2…
        canon = self.alias2label.get(base, base)        # alias2label già esiste
        return canon


    def process_single_image(
            self,
            image_pil: Image.Image,
            image_name: str,
            det_cache_key: str | None = None,
    ):

        if det_cache_key is None:
            det_cache_key = self._generate_cache_key(image_pil, self.question)

        cache_key = det_cache_key or image_name
        print(f"\n[PROCESS] {image_name}")
        t0 = time.time()

        # ------------------------------------------------------------------ init
        fallback_seed_idx: int | None = None
        skip_label_nms = False                     # <- inizializziamo subito

        # ===================================================== 1) DETECTION + SAM
        if cache_key in self._det_cache and not self.config.get("aggressive_pruning"):
            print("[DBG]  using cached detections")
            cached       = self._det_cache[cache_key]
            boxes_full   = cached["boxes"]
            labels_full  = cached["labels"]
            scores_full  = cached["scores"]
            masks_full   = cached["all_masks_full"]

            orig_boxes, orig_labels, orig_scores = (
                boxes_full.copy(), labels_full.copy(), scores_full.copy()
            )
        else:
            # --------------------------- detector inference --------------------
            detection_result = self._cached_detection(image_pil, cache_key)
            all_detections = detection_result["detections"]
            det_counts = detection_result["counts"]

            print("[DBG]  count per detector →", det_counts)

            if not all_detections:
                out_path = os.path.join(self.output_folder, f"{image_name}_output.jpg")
                image_pil.save(out_path)
                print("[INFO] Nessuna detection. Immagine salvata intera.")
                return

            # unzip detections
            boxes_full  = [d["box"]   for d in all_detections]
            labels_full = [d["label"] for d in all_detections]
            scores_full = [d["score"] for d in all_detections]

            # ---------- Fusion via WBF ---------------------------------------
            self.curr_img_size = image_pil.size  # (w,h) necessario in _wbf_fusion
            boxes_full, labels_full, scores_full = self._wbf_fusion(all_detections)

            labels_full = [self._canon_label(l) for l in labels_full]

            # ---------- riallinea la lista all_detections alla fusione ----------
            all_detections = [
                {"box": b, "label": l, "score": s}
                for b, l, s in zip(boxes_full, labels_full, scores_full)
            ]
            print(f"[DBG]  WBF → {len(boxes_full)} box dopo fusione")


            # salviamo gli originali
            orig_boxes, orig_labels, orig_scores = (
                boxes_full.copy(), labels_full.copy(), scores_full.copy()
            )

            # ---------------------- HARD-PRUNING (aggressive) ------------------
            if self.config.get("aggressive_pruning"):
                pr_boxes, pr_labels, pr_scores = self._filter_by_question(
                    boxes_full, labels_full, scores_full
                )
                keep_idx      = {i for i, b in enumerate(boxes_full) if b in pr_boxes}
                pr_detections = [all_detections[i] for i in keep_idx]

                print(f"[DBG]  after question-filter → {len(pr_boxes)} / {len(boxes_full)} boxes")

                if len(pr_boxes) <= 1:                       # fallback singleton
                    if pr_boxes:
                        seed_label, seed_score = pr_labels[0], pr_scores[0]
                    else:
                        seed_label, seed_score = orig_labels[0], orig_scores[0]

                    fallback_seed_idx = next(
                        i for i, (l, s) in enumerate(zip(orig_labels, orig_scores))
                        if l == seed_label and abs(s - seed_score) < 1e-6
                    )
                    print(f"[DBG]  singleton fallback → idx={fallback_seed_idx} "
                          f"({seed_label}, {seed_score:.3f})")

                    do_hard_relation_pruning = True
                else:                                        # pruning “normale”
                    boxes_full, labels_full, scores_full = pr_boxes, pr_labels, pr_scores
                    all_detections = pr_detections
                    do_hard_relation_pruning = True
                    print(f"[DBG]  kept {len(boxes_full)} boxes after hard pruning")
            else:
                do_hard_relation_pruning = False
                print("[DBG]  aggressive_pruning OFF")

            # --------------------------- SAM + fusion --------------------------
            masks_full = self._run_sam_combined_segmentation(image_pil, boxes_full)
            print(f"[DBG]  SAM masks generated: {len(masks_full)}")

            for det, sam_dict in zip(all_detections, masks_full):
                dmask = det.get("det2_mask")
                if dmask is not None:
                    sam_dict["segmentation"] = self._fuse_masks(
                        sam_dict["segmentation"], dmask,
                        method="iou_union", iou_thresh=0.5
                    )

                if self.config.get("close_holes", False):
                    sam_dict["segmentation"] = self._close_mask_holes(
                        sam_dict["segmentation"],
                        ksize=self.config.get("hole_kernel", 7),
                        min_hole_area=self.config.get("min_hole_area", 100)
                    )

            print("[DBG]  masks fused (if det2_mask present)")


            # cache
            self._det_cache[cache_key] = {
                "boxes": boxes_full, "labels": labels_full,
                "scores": scores_full, "all_masks_full": masks_full,
            }

        # ===================================================== 2) RELAZIONI GLOBALI
        boxes, labels, scores = boxes_full.copy(), labels_full.copy(), scores_full.copy()
        masks = masks_full

        # compute relative depth once for the current boxes
        centres    = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
        depth_vals = self._relative_depth(image_pil, centres)

        rels_all = self._unify_pair_relations(
            self._infer_relationships_improved(image_pil, boxes, labels, masks, depth_vals)
        )
        rels_all = self._drop_inverse_duplicates(rels_all)

        # --- NEW: filtra in base alla domanda PRIMA del taglio per-oggetto ---
        if self.filter_relations_by_question:
            rels_q = self._filter_relationships_by_question(rels_all)
            # se nulla passa il filtro, tieni comunque tutte le relazioni
            if rels_q:
                rels_all = rels_q
        # Ordina globalmente con la stessa priorità
        if self._parsed_question_relation_terms:
            rels_all = sorted(
                rels_all,
                key=lambda r: (
                    0 if self._is_question_relation(r["relation"]) else 1,
                    r.get("distance", 1e9)
                )
            )
        # ===================================================== 3) FALLBACK singleton

        if fallback_seed_idx is not None:
            if self.filter_relations_by_question:
                tmp = self._filter_relationships_by_question(rels_all)
                if tmp:
                    rels_all = tmp
            rels_all = self._limit_relationships_per_object(rels_all, boxes)
            rels_all = self._drop_inverse_duplicates(rels_all)
            rels = [r for r in rels_all
                    if r["src_idx"] == fallback_seed_idx or r["tgt_idx"] == fallback_seed_idx]

            scene_graph = self._build_scene_graph(image_pil, boxes, labels, scores)
            scene_graph.remove_edges_from([
                (u, v) for u, v in scene_graph.edges()
                if u != fallback_seed_idx and v != fallback_seed_idx
            ])

            # Controlla le flag anche per il fallback
            save_image_only = self.config.get("save_image_only", False)
            skip_graph = self.config.get("skip_graph", False) or save_image_only
            skip_prompt = self.config.get("skip_prompt", False) or save_image_only
            skip_visualization = self.config.get("skip_visualization", False)

            if not skip_graph:
                self._save_gpickle(scene_graph,
                    os.path.join(self.output_folder, f"{image_name}_graph.gpickle"))
                with open(os.path.join(self.output_folder,
                                    f"{image_name}_graph.json"), "w") as jf:
                    json.dump(nx.node_link_data(scene_graph), jf)

            if not skip_prompt:
                with open(os.path.join(self.output_folder,
                                    f"{image_name}_scene_prompt.txt"), "w") as fp:
                    fp.write(self._to_prompt(scene_graph))

            if not skip_visualization or save_image_only:
                self._visualize_detections_and_relationships_with_auto_masks(
                    image=image_pil,
                    boxes=boxes, labels=labels, scores=scores,
                    relationships=rels, all_masks=masks,
                    save_path=os.path.join(self.output_folder, f"{image_name}_output.jpg"),
                )

            print(f"[DBG]  fallback → boxes:{len(boxes)}  rels:{len(rels)}")
            skip_label_nms = True
            if self.config.get("export_preproc_only", False):
                overlay_path = os.path.join(self.output_folder, f"{image_name}_preproc.png")
                self._visualize_detections_and_relationships_with_auto_masks(
                    image=image_pil,
                    boxes=boxes, labels=labels, scores=scores,
                    relationships=rels, all_masks=masks,
                    save_path=overlay_path,
                    draw_background=False, bg_color=(1,1,1,0)
                )
            print(f"[DONE] {image_name} (singleton fallback) in {time.time()-t0:.2f}s")
            return

        else:

            if self.apply_question_filter and self._parsed_question_object_terms:
                question_idxs = {
                    i for i, lab in enumerate(labels)
                    if lab.lower() in self._parsed_question_object_terms
                }

                if question_idxs:
                    # ──────────────────────────────────────────────────────────────
                    #  CASO 1 ‒ pruning "duro" (solo i seed citati nella domanda)
                    # ──────────────────────────────────────────────────────────────
                    if do_hard_relation_pruning:
                        filtered_idxs = sorted(question_idxs)
                        filtered_rels = []            # nessuna relazione, per ora

                        # ─── fallback: 1 solo oggetto → tieni tutta la scena ───
                        if len(filtered_idxs) == 1:
                            seed = filtered_idxs[0]

                            # 1) mantieni **tutti** gli oggetti rimasti dopo i detector
                            filtered_idxs = list(range(len(boxes)))

                            # 2) relazioni: solo quelle che coinvolgono il seed
                            filtered_rels = [
                                {
                                    "src_idx": (seed if r["src_idx"] == seed else filtered_idxs.index(r["src_idx"])),
                                    "tgt_idx": (seed if r["tgt_idx"] == seed else filtered_idxs.index(r["tgt_idx"])),
                                    "relation": r["relation"],
                                    "distance": r["distance"],
                                }
                                for r in rels_all
                                if r["src_idx"] == seed or r["tgt_idx"] == seed
                            ]


                    # ──────────────────────────────────────────────────────────────
                    #  CASO 2 ‒ pruning "morbido": seed + vicini
                    # ──────────────────────────────────────────────────────────────
                    else:
                        neighbor_idxs = (
                            {r["tgt_idx"] for r in rels_all if r["src_idx"] in question_idxs}
                            |
                            {r["src_idx"] for r in rels_all if r["tgt_idx"] in question_idxs}
                        )
                        filtered_idxs = sorted(question_idxs | neighbor_idxs)

                        filtered_rels = [
                            {
                                "src_idx": filtered_idxs.index(r["src_idx"]),
                                "tgt_idx": filtered_idxs.index(r["tgt_idx"]),
                                "relation": r["relation"],
                                "distance": r["distance"],
                            }
                            for r in rels_all
                            if (
                                r["src_idx"] in question_idxs and r["tgt_idx"] in neighbor_idxs
                            ) or (
                                r["tgt_idx"] in question_idxs and r["src_idx"] in neighbor_idxs
                            )
                        ]
                        filtered_rels = self._limit_relationships_per_object(
                            filtered_rels, boxes
                        )
                        filtered_rels = self._drop_inverse_duplicates(filtered_rels)
                else:
                    print("[INFO] Matched terms but no seeds—falling back to full set.")
                    filtered_idxs = list(range(len(boxes)))
                    filtered_rels = []
            else:
                filtered_idxs = list(range(len(boxes)))
                filtered_rels = []


        # ricostruisci liste pruned
        boxes  = [boxes[i]   for i in filtered_idxs]
        labels = [labels[i]  for i in filtered_idxs]
        scores = [scores[i]  for i in filtered_idxs]

        # ── NMS per etichetta ─────────────────────────────────────────────
        if skip_label_nms:
            keep = list(range(len(boxes)))      # conserva tutti i box
        else:
            by_label = defaultdict(list)
            for i, lab in enumerate(labels):
                by_label[lab.lower()].append(i)

            keep = []
            for idxs in by_label.values():
                kept = self._non_maximum_suppression(
                    [boxes[i]   for i in idxs],
                    [scores[i]  for i in idxs],
                    self.label_nms_threshold,
                )
                keep.extend(idxs[k] for k in kept)
            keep.sort()



        # ---------------------------------------------------------------
        #  Ricostruisci liste finali con il nuovo “keep”
        # ---------------------------------------------------------------
        boxes  = [boxes[i]   for i in keep]
        labels = [labels[i]  for i in keep]
        scores = [scores[i]  for i in keep]
        labels = [f"{self._canon_label(lab)}_{idx+1}" for idx, lab in enumerate(labels)]


        masks   = [masks[i]   for i in keep]

        keep_set   = set(keep)
        old2new_id = {old: new for new, old in enumerate(keep)}

        rels = [
            {
                "src_idx": old2new_id[r["src_idx"]],
                "tgt_idx": old2new_id[r["tgt_idx"]],
                "relation": r["relation"],
                "distance": r["distance"],
            }
            for r in rels_all
            if r["src_idx"] in keep_set and r["tgt_idx"] in keep_set
        ]
        # riesegui il filtro dopo il remap degli indici
        if self.filter_relations_by_question:
            tmp = self._filter_relationships_by_question(rels)
            if tmp:
                rels = tmp
        rels = self._limit_relationships_per_object(rels, boxes)
        rels = self._drop_inverse_duplicates(rels)


        # ------------------------------------------------------------------
        # 3) SALVATAGGI / VISUALIZZAZIONE
        # ------------------------------------------------------------------
        # Controlla le flag per determinare cosa salvare
        save_image_only = self.config.get("save_image_only", False)
        skip_graph = self.config.get("skip_graph", False) or save_image_only
        skip_prompt = self.config.get("skip_prompt", False) or save_image_only
        skip_visualization = self.config.get("skip_visualization", False)

        # Salva sempre il scene graph se necessario per altri output
        scene_graph = None
        if not skip_graph or not skip_prompt or not skip_visualization:
            scene_graph = self._build_scene_graph(image_pil, boxes, labels, scores)

        # Salva graph files solo se non skippato
        if not skip_graph:
            self._save_gpickle(
                scene_graph,
                os.path.join(self.output_folder, f"{image_name}_graph.gpickle")
            )
            with open(os.path.join(self.output_folder, f"{image_name}_graph.json"), "w") as jf:
                json.dump(nx.node_link_data(scene_graph), jf)

        # Salva prompt solo se non skippato
        if not skip_prompt:
            with open(os.path.join(self.output_folder, f"{image_name}_scene_prompt.txt"), "w") as fp:
                fp.write(self._to_prompt(scene_graph))

        # Salva visualizzazione solo se non skippata
        if not skip_visualization:
            out_path = os.path.join(self.output_folder, f"{image_name}_output.jpg")
            self._visualize_detections_and_relationships_with_auto_masks(
                image=image_pil,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=rels,
                all_masks=masks,
                save_path=out_path,
            )
        elif save_image_only:
            # Se save_image_only è True, salva solo l'immagine processata
            out_path = os.path.join(self.output_folder, f"{image_name}_output.jpg")
            self._visualize_detections_and_relationships_with_auto_masks(
                image=image_pil,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=rels,
                all_masks=masks,
                save_path=out_path,
            )
        if self.config.get("export_preproc_only", False):
            overlay_path = os.path.join(self.output_folder, f"{image_name}_preproc.png")
            self._visualize_detections_and_relationships_with_auto_masks(
                image=image_pil,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=rels,
                all_masks=masks,
                save_path=overlay_path,
                draw_background=False,
                bg_color=(1, 1, 1, 0)          # trasparente; usa (1,1,1,1) per bianco
            )

        print(f"[DONE] {image_name} processed in {time.time() - t0:.2f}s")

        # --- pulizia memoria ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()




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
        parser.add_argument("--seg_fill_alpha", type=float, default=0.55,
                            help="Opacità del riempimento delle maschere")
        parser.add_argument("--bbox_linewidth", type=float, default=3.0,
                            help="Spessore del bordo dei bounding box")

        parser.add_argument("--obj_fontsize_inside",  type=int, default=8)
        parser.add_argument("--obj_fontsize_outside", type=int, default=8)
        parser.add_argument("--rel_fontsize",         type=int, default=8)
        parser.add_argument("--legend_fontsize",      type=int, default=8)

        parser.add_argument("--rel_arrow_linewidth",      type=float, default=2)
        parser.add_argument("--rel_arrow_mutation_scale", type=float, default=22)


        parser.add_argument("--no_bboxes", action="store_true", help="Non disegnare i bounding box")
        parser.add_argument("--no_masks",  action="store_true", help="Non disegnare le maschere SAM")
        parser.add_argument("--no_instances", action="store_true",
                            help="Nascondi sia maschere che bounding box (override)")


        # NMS e segmentazione
        parser.add_argument("--label_nms_threshold", type=float, default=0.5)
        parser.add_argument("--seg_iou_threshold",   type=float, default=0.5)
        parser.add_argument("--close_holes", action="store_true",
                            help="Chiudi eventuali buchi interni nelle maschere SAM")
        parser.add_argument("--hole_kernel", type=int, default=5,
                            help="Dimensione del kernel (pixel) per il morph. closing")
        parser.add_argument("--min_hole_area", type=int, default=500,
                            help="Area minima del buco da riempire (pixel)")

        # Relazioni geometriche
        parser.add_argument("--overlap_thresh", type=float, default=0.3)
        parser.add_argument("--margin",         type=int,   default=20)
        parser.add_argument("--min_distance",   type=float, default=1)
        parser.add_argument("--max_distance",   type=float, default=20000)

        # SAM
        parser.add_argument("--points_per_side",        type=int,   default=32)
        parser.add_argument("--pred_iou_thresh",        type=float, default=0.8)
        parser.add_argument("--stability_score_thresh", type=float, default=0.85)
        parser.add_argument("--min_mask_region_area",   type=int,   default=300)

        # Dispositivo
        parser.add_argument("--preproc_device", type=str, default=None)

        parser.add_argument(
            "--sam_version",
            type=str,
            choices=["1", "2", "hq"],
            default="1",
            help="1 = Segment-Anything v1, 2 = Segment-Anything 2, hq = SAM-HQ (default: 1)",
        )

        parser.add_argument("--sam_hq_model_type", type=str,
                        choices=["vit_b", "vit_l", "vit_h"],
                        default="vit_h",
                        help="SAM-HQ model size (only used if sam_version=hq)")


        parser.add_argument("--conceptnet_timeout", type=float, default=4)
        parser.add_argument("--conceptnet_max_retry", type=int, default=3)
        parser.add_argument("--conceptnet_skip_on_fail", action="store_true")

        parser.add_argument("--no_legend",
                    action="store_true",
                    help="Disabilita la legenda dei colori delle classi")

        parser.add_argument(
            "--aggressive_pruning",
            action="store_true",
            help="Tieni SOLO gli oggetti esplicitamente citati nella domanda \
                  e filtra le relazioni sul tipo richiesto"
        )

        parser.add_argument("--display_relation_labels", action="store_true",
                    help="Show relation labels on arrows/lines")
        parser.add_argument("--resolve_overlaps", action="store_true",
                    help="Evita sovrapposizioni tra label di oggetti e relazioni")



        parser.add_argument("--save_image_only", action="store_true",
                    help="Save only the processed image, skip graph files")
        parser.add_argument("--skip_graph", action="store_true",
                    help="Skip saving graph files (.gpickle, .json)")
        parser.add_argument("--skip_prompt", action="store_true",
                    help="Skip saving scene prompt file")
        parser.add_argument("--skip_visualization", action="store_true",
                    help="Skip saving the visualization image")
        parser.add_argument("--export_preproc_only", action="store_true",
            help="Salva anche un'immagine con SOLO segmentazioni/etichette/relazioni (senza sfondo)")

        parser.add_argument("--enable_detection_cache", action="store_true",
                       help="Enable detection caching for faster reprocessing")
        parser.add_argument("--max_cache_size", type=int, default=100,
                          help="Maximum number of cached detection results")
        parser.add_argument("--clear_cache", action="store_true",
                          help="Clear detection cache before starting")




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
        "display_relation_labels": args.display_relation_labels,
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
        "sam_version": args.sam_version,
        "sam_hq_model_type": getattr(args, 'sam_hq_model_type', 'vit_h'),
        "display_legend": not args.no_legend,
        "aggressive_pruning": args.aggressive_pruning,
        "save_image_only": args.save_image_only,
        "skip_graph": args.skip_graph,
        "skip_prompt": args.skip_prompt,
        "skip_visualization": args.skip_visualization,
        "enable_detection_cache": args.enable_detection_cache,
        "max_cache_size": args.max_cache_size,
        "seg_fill_alpha":      getattr(args, "seg_fill_alpha", 0.6),
        "bbox_linewidth":      getattr(args, "bbox_linewidth", 2.0),
        "resolve_overlaps": args.resolve_overlaps,
        "close_holes": args.close_holes,
        "hole_kernel": args.hole_kernel,
        "min_hole_area": args.min_hole_area,
        "export_preproc_only": args.export_preproc_only,
        "show_segmentation": False if args.no_masks or args.no_instances else args.show_segmentation,
        "show_bboxes":       False if args.no_bboxes or args.no_instances else True,
        "obj_fontsize_inside":  args.obj_fontsize_inside,
        "obj_fontsize_outside": args.obj_fontsize_outside,
        "rel_fontsize":         args.rel_fontsize,
        "legend_fontsize":      args.legend_fontsize,
        "rel_arrow_linewidth":      args.rel_arrow_linewidth,
        "rel_arrow_mutation_scale": args.rel_arrow_mutation_scale,
    }

    if args.clear_cache:
        # Crea un'istanza temporanea per pulire la cache
        temp_preproc = ImageGraphPreprocessor(config)
        if hasattr(temp_preproc, '_detection_cache'):
            temp_preproc._detection_cache.clear()
        if hasattr(temp_preproc, '_det_cache'):
            temp_preproc._det_cache.clear()
        print("[INFO] Detection cache cleared")
        del temp_preproc  # Libera la memoria

        # Se clear_cache è l'unico scopo, esci
        if not (args.input_path or args.json_file or args.dataset):
            print("[INFO] Cache cleared. No processing requested.")
            exit(0)

    preproc = ImageGraphPreprocessor(config)
    preproc.run()

