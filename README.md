# GD_SAM_PD_YOLO
A Zero-Shot Lightweight Method for Camellia oleifera Fruit Segmentation

get_box.py and labels_revise.py are used to obtain prediction labels from Grounding DINO and to perform bounding-box filtering, respectively. These scripts should be executed within the Grounding DINO 1.5 framework, whose official codebase is available at: https://github.com/IDEA-Research/Grounding-DINO-1.5-API.

from_box_get_mask.py performs instance segmentation using bounding-box prompts with the SAM 2.1 model and should be used within the SAM 2.1 framework, available at: https://github.com/facebookresearch/sam2.

torch_prune.py and torch_distillation.py implement model pruning and knowledge distillation for YOLO models, respectively. These scripts are designed to be used within the YOLO11 framework. The pruning and distillation implementations are adapted from the repository: https://github.com/zhahoi/yolov11_prune_distillation.

This code corresponds to the version used in the submitted manuscript.
