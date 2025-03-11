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

import os
from argparse import ArgumentParser
from holoscan.core import Operator, OperatorSpec, ConditionType
from applications.yolo_model_deployment.grounding_dino_op import GroundingDINOOp, GroundingDINOPostProcessorOp
from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator
from nv_grounding_dino_op import CaptionPreprocessorOp, DetectionPostprocessorOp
class TensorPassthroughOp(Operator):
    """Simple operator that passes through a received tensor without backpressure"""
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_tensor")
        spec.output("out_tensor").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        tensor = op_input.receive("in_tensor")
        if tensor:
            op_output.emit(tensor, "out_tensor")
class YoloDetApp(Application):
    """
    YOLO Detection Application.

    This application performs object detection using a YOLO model. It supports
    video input from a replayer or a V4L2 device and visualizes the detection results.

    Parameters:
        video_dir (str): Path to the video directory.
        data (str): Path to the model data directory.
        source (str): Input source, either "replayer" or "v4l2".
    """

    def __init__(self, video_dir, data, source="replayer"):
        super().__init__()
        self.name = "YOLO Detection App"
        self.source = source

        # Set default paths if not provided
        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOHUB_DATA_PATH", "../data"), "yolo_model_deployment"
            )
        self.data = data

        if video_dir == "none":
            video_dir = data
        self.video_dir = video_dir

    def compose(self):
        # Resource allocator
        pool = UnboundedAllocator(self, name="pool")

        # Input source
        if self.source == "v4l2":
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
            source_output = "signal"
            in_dtype = "rgba8888"  # V4L2 outputs RGBA8888
        else:
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=self.video_dir,
                **self.kwargs("replayer"),
            )
            source_output = "output"
            in_dtype = "rgb888"

        # Operators
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("detection_preprocessor"),
        )

        caption_preprocessor = CaptionPreprocessorOp(
            self,
            name="caption_preprocessor",
            tokenizer_model='bert-base-uncased',
            max_text_len=256, # model 256
            **self.kwargs("caption_preprocessor"),
        )

        inference_kwargs = self.kwargs("detection_inference")
        for k, v in inference_kwargs["model_path_map"].items():
            inference_kwargs["model_path_map"][k] = os.path.join(self.data, v)

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="allocator"),
            **inference_kwargs,
        )

        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )
        # Hugging Face Grounding DINO
        grounding_dino = GroundingDINOOp(self, name="GroundingDINOOp", **self.kwargs("grounding_dino"))
        grounding_dino_post_processor = GroundingDINOPostProcessorOp(self, name="GroundingDINOPostProcessorOp", **self.kwargs("grounding_dino_post_processor"))

        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=[
                dict(name="", type="color"),
            ],
            **self.kwargs("detection_visualizer"),
        )

        if self.kwargs("app").get("model_type") == "nv_grounding_dino":
            self.add_flow(source, detection_visualizer, {(source_output, "receivers")})
            self.add_flow(source, detection_preprocessor)
            
            # image preprocessor
            self.add_flow(detection_preprocessor, detection_inference, {("", "receivers")})
            
            # caption preprocessor
            self.add_flow(caption_preprocessor, detection_inference, {("out", "receivers")})
            self.add_flow(caption_preprocessor, detection_postprocessor, {("pos_map", "pos_map")})
          
            self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "in")})
            self.add_flow(detection_postprocessor, detection_visualizer, {("outputs", "receivers")})
        else: # Hugging Face Grounding DINO
            self.add_flow(source, grounding_dino, {("", "video_stream")})
            self.add_flow(grounding_dino, grounding_dino_post_processor, {("results", "results")})
            self.add_flow(grounding_dino_post_processor, detection_visualizer, {("outputs", "receivers")})


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="YOLO Detection Demo Application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=("Input source: 'v4l2' for V4L2 device or 'replayer' for video stream replayer."),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help="Path to the model data directory.",
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        default="none",
        help="Path to the video directory.",
    )
    
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "yolo_detection.yaml")
    app = YoloDetApp(video_dir=args.video_dir, data=args.data, source=args.source)
    app.config(config_file)
    app.run()
