# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import logging

from holoscan.core import Operator, OperatorSpec, IOSpec
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import cupy as cp

class GroundingDINOOp(Operator):
    @property
    def formatted_labels(self):
        return ". ".join(self.labels) + "."

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger('GroundingDINOOp')
        self.processor = None
        self.model = None
        # Get configuration from kwargs
        self.cache_dir = kwargs.get('cache_dir', '/workspace/holohub/data/synchron/grounding-dino-tiny')
        self.local_only = kwargs.get('local_only', False)
        self.labels = kwargs.get('labels', [])
        self.box_threshold = kwargs.get('box_threshold', 0.4)
        self.text_threshold = kwargs.get('text_threshold', 0.3)
        self.height = kwargs.get('height', 480)
        self.width = kwargs.get('width', 640)

    def start(self):
        # Initialize Grounding DINO model and processor
        self.processor = AutoProcessor.from_pretrained(
            # self.cache_dir,
            "IDEA-Research/grounding-dino-tiny",
            cache_dir=self.cache_dir,
            # local_files_only=self.local_only,
            )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            # self.cache_dir,
            "IDEA-Research/grounding-dino-tiny",
            cache_dir=self.cache_dir,
            # local_files_only=self.local_only,
            ).to("cuda")
        self._logger.info('Finished HuggingFace initialization')

    def setup(self, spec: OperatorSpec):
        spec.input("video_stream")
        spec.output("results")

    def set_labels(self, labels):
        self.labels = labels

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("video_stream")
        logging.debug(f'Received messsage: {in_message} and labels: {self.labels}')
        in_message = in_message.get("")
        if in_message:
            # Create a b64 Image from the Holoscan Tensor
            cp_image = cp.from_dlpack(in_message)
            np_image = cp.asnumpy(cp_image)
            image = Image.fromarray(np_image)
            # TODO undo this resizing and leave the native size
            image = image.resize((self.width, self.height))

            inputs = self.processor(images=image, text=self.formatted_labels, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]
            )[0]
            
            # Normalize bounding box coordinates to (0,1) range
            if len(results["boxes"]) > 0:
                results["boxes"][:, [0, 2]] /= self.width  # normalize x coordinates
                results["boxes"][:, [1, 3]] /= self.height  # normalize y coordinates
            
            print("Grounding DINO results: ", results)
            # op_output.emit({"": results}, "results")


# Write a PostProcessorOp that process the results for holoviz
class GroundingDINOPostProcessorOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("results")
        # spec.output("output_specs")
        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("results")
        # results: {'scores': tensor([0.5357, 0.4618, 0.4078, 0.4619], device='cuda:0'), 'boxes': tensor([[0.1324, 0.7994, 0.2432, 0.9714],
        # [0.9681, 0.0958, 0.9998, 0.3059],
        # [0.1989, 0.6252, 0.2767, 0.7520],
        # [0.3636, 0.6057, 0.5447, 0.9985]], device='cuda:0'), 'text_labels': ['coffee mug', 'water bottle', 'coffee mug', 'monitor'], 'labels': ['coffee mug', 'water bottle', 'coffee mug', 'monitor']}

        # Process the results for holoviz
        # boxes: (nboxes, 4)
        # scores: (nboxes,)
        # text_labels: (nboxes,)
        # labels: (nboxes,)
        
        # 1. convert boxes to (cx, cy, w, h)
        # 2. rescale to (0, 1)
        # 3. reshape to (1, nboxes*2, ncoord/2)

        # 1. convert boxes to (cx, cy, w, h)
        boxes = in_message['boxes']
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        boxes = torch.stack([cx, cy, w, h], dim=1)

        # 2. rescale to (0, 1)
        boxes = boxes / torch.tensor([self.width, self.height, self.width, self.height])

        # 3. reshape to (1, nboxes*2, ncoord/2)
        boxes = boxes.reshape(1, -1, 2)

        out_message = {"bbox": boxes}
        op_output.emit(out_message, "outputs")



