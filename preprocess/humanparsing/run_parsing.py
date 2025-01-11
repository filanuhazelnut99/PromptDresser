import os
import pdb
import sys
from pathlib import Path

import onnxruntime as ort

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from parsing_api import onnx_inference


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        # torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        #### jho modified >>>>
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': gpu_id,
            }),
            'CPUExecutionProvider',
        ]
        self.session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options, providers=providers)
        self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=providers)
        #### jho modified <<<<
        # session_options.add_session_config_entry('gpu_id', str(gpu_id))
        # self.session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_atr.onnx'),
        #                                     sess_options=session_options, providers=['CUDAExecutionProvider'])
        # self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_lip.onnx'),
        #                                         sess_options=session_options, providers=['CUDAExecutionProvider'])
        print(f"parsing init done (gpu: {gpu_id})")

    def __call__(self, input_image):
        torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
 