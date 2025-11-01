import numpy as np
from igp.utils.detector_utils import make_detection
from igp.fusion.wbf import fuse_detections_wbf
from igp.types import Detection


def test_make_detection_mask_and_extra():
    box = (0, 0, 10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True
    d = make_detection(box, 'Cat', 0.9, source='test', mask=mask, extra={'foo': 'bar'})
    assert isinstance(d, Detection)
    assert d.box == (0.0, 0.0, 10.0, 10.0)
    assert d.label.lower() == 'cat' or d.label == 'Cat'
    assert 0.89 < d.score < 0.91
    assert d.extra is not None
    assert 'segmentation' in d.extra
    assert d.extra['segmentation'].shape == (10,10)
    assert d.extra.get('foo') == 'bar'


def test_wbf_empty_behavior():
    # No detections
    dets = []
    out = fuse_detections_wbf(dets, (100,100), fallback_to_original=False)
    assert out == []
    out2 = fuse_detections_wbf(dets, (100,100), fallback_to_original=True)
    assert out2 == []  # original is empty as well
