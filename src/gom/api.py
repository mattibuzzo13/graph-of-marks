"""
High-level API for Graph of Marks

Provides a simplified interface with support for custom functions
for detection, segmentation, depth estimation, and relationship extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .config import PreprocessorConfig
from .pipeline.preprocessor import ImageGraphPreprocessor
from .types import Detection, Relationship

logger = logging.getLogger(__name__)


class GraphOfMarks:
    """
    High-level interface for the Graph of Marks visual scene understanding pipeline.

    This class provides a simplified API with support for custom functions for key
    components (detection, segmentation, depth estimation, relationship extraction).

    Args:
        detectors: List of detector names to use (e.g., ["yolov8", "owlvit"])
        sam_version: SAM version to use ("sam1", "sam2", "hq", "fast")
        use_depth: Whether to use depth estimation
        use_clip_relations: Whether to use CLIP for semantic relationships
        output_folder: Output directory for results
        custom_detector: Custom detection function (optional)
        custom_segmenter: Custom segmentation function (optional)
        custom_depth_estimator: Custom depth estimation function (optional)
        custom_relation_extractor: Custom relationship extraction function (optional)
        **config_kwargs: Additional configuration options passed to PreprocessorConfig

    Custom Function Signatures:
        custom_detector(image: np.ndarray, **kwargs) -> List[Detection]
        custom_segmenter(image: np.ndarray, boxes: List[Box], **kwargs) -> Dict[str, Any]
        custom_depth_estimator(image: np.ndarray, **kwargs) -> np.ndarray
        custom_relation_extractor(detections: List[Detection], image: np.ndarray, **kwargs) -> List[Relationship]

    Examples:
        Basic usage:
            >>> gom = GraphOfMarks()
            >>> result = gom.process_image("scene.jpg")

        With configuration:
            >>> gom = GraphOfMarks(
            ...     detectors=["yolov8", "owlvit"],
            ...     sam_version="sam2",
            ...     use_depth=True
            ... )
            >>> result = gom.process_image("scene.jpg", question="What is in the room?")

        With custom segmentation:
            >>> def my_segmenter(image, boxes, **kwargs):
            ...     # Your segmentation logic
            ...     masks = []
            ...     for box in boxes:
            ...         mask = segment_box(image, box)
            ...         masks.append(mask)
            ...     return {'masks': masks}
            >>>
            >>> gom = GraphOfMarks(custom_segmenter=my_segmenter)
            >>> result = gom.process_image("scene.jpg")

        With custom depth estimation:
            >>> def my_depth_estimator(image, **kwargs):
            ...     # Your depth estimation logic
            ...     depth_map = estimate_depth(image)
            ...     # Normalize to [0, 1]
            ...     depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            ...     return depth_map
            >>>
            >>> gom = GraphOfMarks(
            ...     custom_depth_estimator=my_depth_estimator,
            ...     use_depth=True
            ... )
            >>> result = gom.process_image("scene.jpg")
    """

    def __init__(
        self,
        detectors: Optional[List[str]] = None,
        sam_version: str = "sam1",
        use_depth: bool = False,
        use_clip_relations: bool = True,
        output_folder: str = "output_images",
        # GoM-specific visualization settings
        label_mode: str = "original",  # "original", "numeric" (numbers), or "alphabetic" (letters)
        show_masks: bool = True,
        show_relationships: bool = True,
        # Custom functions
        custom_detector: Optional[Callable] = None,
        custom_segmenter: Optional[Callable] = None,
        custom_depth_estimator: Optional[Callable] = None,
        custom_relation_extractor: Optional[Callable] = None,
        **config_kwargs: Any,
    ):
        """
        Initialize the Graph of Marks pipeline with optional custom functions.

        Args:
            detectors: List of detector names (["yolov8", "owlvit", "detectron2", "grounding_dino"])
            sam_version: Segmentation model ("sam1", "sam2", "hq", "fast")
            use_depth: Enable depth estimation for 3D-aware relationships
            use_clip_relations: Enable CLIP-based semantic relationships
            output_folder: Output directory for results
            label_mode: Label display mode:
                - "original": Show class names (e.g., "person", "car")
                - "numeric": Show numbers only (e.g., "1", "2", "3") - Set-of-Mark style
                - "alphabetic": Show letters (e.g., "A", "B", "C")
            show_masks: Display segmentation masks
            show_relationships: Display relationship arrows between objects
            custom_detector: Custom detection function (optional)
            custom_segmenter: Custom segmentation function (optional)
            custom_depth_estimator: Custom depth estimation function (optional)
            custom_relation_extractor: Custom relationship extraction function (optional)
            **config_kwargs: Additional configuration options
        """
        # Store custom functions
        self.custom_detector = custom_detector
        self.custom_segmenter = custom_segmenter
        self.custom_depth_estimator = custom_depth_estimator
        self.custom_relation_extractor = custom_relation_extractor

        # Build configuration
        config_dict = {
            "output_folder": output_folder,
            "sam_version": sam_version,
            "enable_spatial_3d": use_depth,  # Map use_depth to enable_spatial_3d
            # Visualization settings
            "show_segmentation": show_masks,
            "display_relationships": show_relationships,
        }

        if detectors is not None:
            config_dict["detectors_to_use"] = tuple(detectors)

        # Merge with additional config kwargs
        config_dict.update(config_kwargs)

        # Create config object
        self.config = PreprocessorConfig(**config_dict)

        # Set label mode in visualizer config
        if hasattr(self.config, 'visualizer_config') and self.config.visualizer_config:
            self.config.visualizer_config.label_mode = label_mode

        # Store label mode for later use
        self._label_mode = label_mode

        # Initialize preprocessor
        self.preprocessor = ImageGraphPreprocessor(self.config)

        # Apply label mode to visualizer if it exists
        if hasattr(self.preprocessor, 'visualizer') and self.preprocessor.visualizer:
            self.preprocessor.visualizer.cfg.label_mode = label_mode

        # Monkey-patch custom functions if provided
        self._patch_custom_functions()

        logger.info(f"Initialized Graph of Marks with config: {self.config}")

    def _patch_custom_functions(self):
        """Patch the preprocessor with custom functions if provided."""
        if self.custom_segmenter is not None:
            logger.info("Using custom segmentation function")
            self.preprocessor._custom_segmenter = self.custom_segmenter

        if self.custom_detector is not None:
            logger.info("Using custom detection function")
            self.preprocessor._custom_detector = self.custom_detector

        if self.custom_depth_estimator is not None:
            logger.info("Using custom depth estimation function")
            self.preprocessor._custom_depth_estimator = self.custom_depth_estimator

        if self.custom_relation_extractor is not None:
            logger.info("Using custom relationship extraction function")
            self.preprocessor._custom_relation_extractor = self.custom_relation_extractor

    def process_image(
        self,
        image_path: Union[str, Path],
        question: Optional[str] = None,
        save_visualization: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process a single image through the full pipeline.

        Args:
            image_path: Path to the image file
            question: Optional question for VQA-aware filtering
            save_visualization: Whether to save annotated visualization
            **kwargs: Additional options passed to the pipeline

        Returns:
            Dictionary containing:
                - detections: List of Detection objects
                - relations: List of Relationship objects
                - scene_graph: NetworkX graph object
                - scene_graph_json: JSON serialization of scene graph
                - output_path: Path to annotated visualization (if saved)
                - depth_map: Depth map (if use_depth=True)
                - processing_time: Total processing time in seconds

        Example:
            >>> gom = GraphOfMarks()
            >>> result = gom.process_image("scene.jpg", question="What is on the table?")
            >>> print(f"Found {len(result['detections'])} objects")
            >>> print(f"Found {len(result['relations'])} relationships")
        """
        import time
        import json
        from PIL import Image
        import networkx as nx

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Update config with question if provided
        if question is not None:
            self.config.question = question

        # Update config for visualization if needed
        if save_visualization is not None:
            original_skip_viz = self.config.skip_visualization
            self.config.skip_visualization = not save_visualization

        # Load image and process
        t0 = time.time()
        image_pil = Image.open(image_path)
        image_name = image_path.stem

        # Process through pipeline (returns None, saves to disk)
        self.preprocessor.process_single_image(
            image_pil,
            image_name,
            custom_question=question
        )

        processing_time = time.time() - t0

        # Restore original visualization setting
        if save_visualization is not None:
            self.config.skip_visualization = original_skip_viz

        # Build result dictionary from saved files
        result = {
            'detections': [],
            'relations': [],
            'scene_graph': None,
            'scene_graph_json': None,
            'output_path': None,
            'depth_map': None,
            'processing_time': processing_time,
        }

        # Load scene graph if it exists
        graph_json_path = self.config.output_folder / f"{image_name}_graph.json"
        if graph_json_path.exists():
            with open(graph_json_path, 'r') as f:
                result['scene_graph_json'] = json.load(f)

            # Reconstruct scene graph from JSON
            try:
                from gom.types import Detection, Relation

                # Extract detections
                if 'nodes' in result['scene_graph_json']:
                    for node_id, node_data in result['scene_graph_json']['nodes'].items():
                        if node_data.get('label') != 'scene':  # Skip scene node
                            det = Detection(
                                box=node_data.get('bbox', [0, 0, 0, 0]),
                                label=node_data.get('label', ''),
                                score=node_data.get('score', 0.0),
                                source=node_data.get('source', 'unknown')
                            )
                            result['detections'].append(det)

                # Extract relations
                if 'edges' in result['scene_graph_json']:
                    for edge in result['scene_graph_json']['edges']:
                        src_idx = edge.get('source', 0)
                        tgt_idx = edge.get('target', 0)
                        relation = edge.get('relation', 'unknown')

                        rel = Relation(
                            src_idx=src_idx,
                            tgt_idx=tgt_idx,
                            relation=relation
                        )
                        result['relations'].append(rel)
            except Exception as e:
                # If reconstruction fails, just pass the JSON
                pass

        # Add path to visualization if it was created
        output_format = getattr(self.config, 'output_format', 'jpg')
        if output_format not in ['jpg', 'png', 'svg']:
            output_format = 'jpg'
        viz_path = self.config.output_folder / f"{image_name}_output.{output_format}"
        if viz_path.exists():
            result['output_path'] = str(viz_path)

        return result

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        questions: Optional[List[str]] = None,
        save_visualizations: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images through the pipeline.

        Args:
            image_paths: List of paths to image files
            questions: Optional list of questions (one per image)
            save_visualizations: Whether to save annotated visualizations
            **kwargs: Additional options passed to the pipeline

        Returns:
            List of result dictionaries (one per image)

        Example:
            >>> gom = GraphOfMarks()
            >>> images = ["scene1.jpg", "scene2.jpg", "scene3.jpg"]
            >>> results = gom.process_batch(images)
            >>> for i, result in enumerate(results):
            ...     print(f"Image {i}: {len(result['detections'])} objects")
        """
        results = []

        if questions is None:
            questions = [None] * len(image_paths)

        if len(questions) != len(image_paths):
            raise ValueError(
                f"Number of questions ({len(questions)}) must match "
                f"number of images ({len(image_paths)})"
            )

        for image_path, question in zip(image_paths, questions):
            try:
                result = self.process_image(
                    image_path,
                    question=question,
                    save_visualization=save_visualizations,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({"error": str(e), "image_path": str(image_path)})

        return results

    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jpg",
        recursive: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.

        Args:
            directory: Directory containing images
            pattern: Glob pattern for image files (default: "*.jpg")
            recursive: Whether to search recursively
            **kwargs: Additional options passed to process_batch

        Returns:
            List of result dictionaries (one per image)

        Example:
            >>> gom = GraphOfMarks()
            >>> results = gom.process_directory("images/", pattern="*.png")
            >>> print(f"Processed {len(results)} images")
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all matching images
        if recursive:
            image_paths = list(directory.rglob(pattern))
        else:
            image_paths = list(directory.glob(pattern))

        logger.info(f"Found {len(image_paths)} images in {directory}")

        return self.process_batch(image_paths, **kwargs)

    def get_config(self) -> PreprocessorConfig:
        """Get the current configuration object."""
        return self.config

    def update_config(self, **kwargs: Any) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        Example:
            >>> gom = GraphOfMarks()
            >>> gom.update_config(use_depth=True, sam_version="sam2")
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")

        # Re-patch custom functions after config update
        self._patch_custom_functions()


def create_pipeline(
    detectors: Optional[List[str]] = None,
    sam_version: str = "sam1",
    **kwargs: Any,
) -> GraphOfMarks:
    """
    Factory function to create a Graph of Marks pipeline.

    Args:
        detectors: List of detector names
        sam_version: SAM version to use
        **kwargs: Additional configuration options

    Returns:
        GraphOfMarks instance

    Example:
        >>> gom = create_pipeline(detectors=["yolov8"], sam_version="sam2")
        >>> result = gom.process_image("scene.jpg")
    """
    return GraphOfMarks(detectors=detectors, sam_version=sam_version, **kwargs)
