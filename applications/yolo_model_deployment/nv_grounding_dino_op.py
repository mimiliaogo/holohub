# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import numpy as np
from holoscan.core import Operator, OperatorSpec
from transformers import AutoTokenizer
from utils import tokenize_captions, post_process

# COCO label map dictionary
coco_label_map = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

class CaptionPreprocessorOp(Operator):
    def __init__(self, *args, tokenizer_model='bert-base-uncased', max_text_len=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_text_len = max_text_len
        self.cat_list = ['person', 'dog', 'cup']
        self.caption = [" . ".join(self.cat_list) + ' .'] 

    def setup(self, spec: OperatorSpec):
        # spec.input("captions")
        spec.output("out")
        spec.output("pos_map")
        
    def compute(self, op_input, op_output, context):
        # captions = op_input.receive("captions")
        input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map = tokenize_captions(self.tokenizer, self.cat_list, self.caption, self.max_text_len)
        
        # Convert boolean masks to uint8 (0 and 1) while preserving the semantic meaning
        attention_mask = cp.asarray(attention_mask).astype(cp.uint8)
        text_self_attention_masks = cp.asarray(text_self_attention_masks).astype(cp.uint8)
        
        # Ensure other tensors have appropriate types
        input_ids = cp.asarray(input_ids).astype(cp.int64)
        position_ids = cp.asarray(position_ids).astype(cp.int64)
        token_type_ids = cp.asarray(token_type_ids).astype(cp.int64)
        pos_map = cp.asarray(pos_map)
        
        out_message = { 
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }
        # Print the type of each item in out_message, not just ndarray
        for key, value in out_message.items():
            print(f"{key}: {value.dtype}")
        print("out_message of caption_preprocessor", out_message)
        op_output.emit(pos_map, "pos_map")
        op_output.emit(out_message, "out")

class DetectionPostprocessorOp(Operator):
    def __init__(self, *args, width=640, label_name_map=coco_label_map, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.label_name_map = label_name_map

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.input("pos_map")
        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        pos_map = cp.asarray(op_input.receive("pos_map")).get()

        pred_boxes = cp.asarray(
            in_message.get("inference_output_pred_boxes")
        ).get()  # (nbatch, nqueries, 4) --> (1, 900, 4)
        
        pred_logits = cp.asarray(
            in_message.get("inference_output_pred_logits")
        ).get()  # (nbatch, nqueries, 256) --> (1, 900, 256)

        print("pred_logits detailed stats:", {
            "shape": pred_logits.shape,
            "min": np.nanmin(pred_logits),  # Use nanmin to handle NaNs
            "max": np.nanmax(pred_logits),
            "mean": np.nanmean(pred_logits),
            "num_nan": np.isnan(pred_logits).sum(),
            "num_inf": np.isinf(pred_logits).sum(),
            "num_large": (np.abs(pred_logits) > 1e6).sum()  # Check for very large values
        })

        # stats for pred_boxes
        print("pred_boxes detailed stats:", {
            "shape": pred_boxes.shape,
            "value": pred_boxes,
            "min": np.nanmin(pred_boxes),  # Use nanmin to handle NaNs
            "max": np.nanmax(pred_boxes),
            "mean": np.nanmean(pred_boxes),
            "num_nan": np.isnan(pred_boxes).sum(),
            "num_inf": np.isinf(pred_boxes).sum(),
            "num_large": (np.abs(pred_boxes) > 1e6).sum()  # Check for very large values
        })
        
        class_labels, scores, boxes = post_process(pred_logits, pred_boxes, pos_map)
        
        # emit dummy value
        out_message = {"bbox": []}
        op_output.emit(out_message, "outputs") 