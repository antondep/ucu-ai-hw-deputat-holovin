"""vdet: shared utilities for the VisDrone-VID detection homework.

Modules
-------
paths       - locate the dataset on Colab / Kaggle / local machines
data        - load annotations into pandas, sequence-level splits, frame subsampling
evaluate    - one COCO evaluator used for every model (ignore regions handled)
submission  - convert predictions to the Kaggle submission format and validate it
viz         - draw ground truth / predictions on frames
anchors     - k-means anchor analysis for anchor-based detectors
yolo_export - build an Ultralytics-ready dataset (subsampled, ignore regions masked)
"""

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASSES)
