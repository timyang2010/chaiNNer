from __future__ import annotations

import json
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from ...impl.external_stable_diffusion import (
    RESIZE_MODE_LABELS,
    SAMPLER_NAME_LABELS,
    STABLE_DIFFUSION_IMG2IMG_PATH,
    STABLE_DIFFUSION_TEXT2IMG_PATH,
    ResizeMode,
    SamplerName,
    decode_base64_image,
    encode_base64_image,
    get,
    nearest_valid_size,
    post,
    verify_api_connection,
)
from ...node_base import NodeBase, group
from ...node_cache import cached
from ...node_factory import NodeFactory
from ...properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
    SeedInput,
    SliderInput,
    TextAreaInput,
    TextInput,
)
from ...properties.outputs import ImageOutput, TextOutput
from ...utils.seed import Seed
from ...utils.utils import get_h_w_c
from . import category as ExternalStableDiffusionCategory

verify_api_connection()



class Preprocessor(Enum):
    NONE = "none"
    CANNY = "canny"
    DEPTH = "depth"
    DEPTH_LERES = "depth_leres"
    OPENPOSE = "openpose"
    SCRIBBLE = "scribble"



@NodeFactory.register("chainner:stable_diffusion_extension:controlnet")
class ControlNetNode(NodeBase):
    def __init__(self):

        super().__init__()
        response = get("/controlnet/model_list")
        models = response["model_list"]
        model_enum_type = Enum("ControlNetModels", {m.split(" ")[0]:m for m in models})
        self.description = "ControlNet Configuration"
        self.inputs = [
            ImageInput("Image"),
            ImageInput("Mask", image_type="Input0").make_optional(),
            EnumInput(model_enum_type),
            SliderInput("guidance", minimum=0, default=1, maximum=1, slider_step=0.01),
            SliderInput("guidance_start", minimum=0, default=0, maximum=1, slider_step=0.01),
            SliderInput("guidance_end", minimum=0, default=1, maximum=1, slider_step=0.01),
            BoolInput("Guess Mode", default=False),
        ]
        self.outputs = [
            TextOutput("Config", output_type="string")
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "ControlNet"
        self.icon = "MdTextFields"
        self.sub = "ControlNet"

    def run(self, image, mask, model, guidance, guidance_start, guidance_end, guess: bool) -> str:
        if(mask is None): mask = np.zeros_like(image)
        return json.dumps(
             {
                "input_image": encode_base64_image(image),
                "mask": encode_base64_image(mask),
                "module": 'none',
                "model": model.value,
                "weight": 1,
                "resize_mode": "Scale to Fit (Inner Fit)",
                "lowvram": False,
                "guidance": guidance,
                "guidance_start": guidance_start,
                "guidance_end": guidance_end,
                "guessmode": guess
            }
        )


@NodeFactory.register("chainner:stable_diffusion_extension:controlnet_preprocess_depth")
class ControlNetDepthPreprocessNode(NodeBase):
    def __init__(self):

        super().__init__()
        self.description = "ControlNet Depth detector"
        self.inputs = [
            ImageInput(),
            SliderInput("Midas Resolution", minimum=64, default=384, maximum=2048),
        ]
        self.outputs = [
            ImageOutput(image_type="Input0", channels=None)
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Depth"
        self.icon = "BsFillImageFill"
        self.sub = "Preprocess"

    def run(self, image, arg1) -> np.ndarray:


        response = post("/controlnet/detect",
            {
                "controlnet_module": "depth",
                "controlnet_input_images": [encode_base64_image(image)],
                "controlnet_processor_res": arg1
            }
        )
        return decode_base64_image(response["images"][0])

@NodeFactory.register("chainner:stable_diffusion_extension:controlnet_preprocess_depth_leres")
class ControlNetDepthLeresPreprocessNode(NodeBase):
    def __init__(self):

        super().__init__()
        self.description = "ControlNet Depth detector"
        self.inputs = [
            ImageInput(),
            SliderInput("Midas Resolution", minimum=64, default=384, maximum=2048),
            SliderInput("Remove Near %", minimum=0, default=0, maximum=100, slider_step=0.1),
            SliderInput("Remove Background %", minimum=0, default=0, maximum=100, slider_step=0.1),
        ]
        self.outputs = [
            ImageOutput(image_type="Input0", channels=None)
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Depth_leres"
        self.icon = "BsFillImageFill"
        self.sub = "Preprocess"

    def run(self, image,  arg1, arg2, arg3) -> np.ndarray:


        response = post("/controlnet/detect",
            {
                "controlnet_module": "depth_leres",
                "controlnet_input_images": [encode_base64_image(image)],
                "controlnet_processor_res": arg1,
                 "controlnet_threshold_a": arg2,
                "controlnet_threshold_b": arg3
            }
        )
        return decode_base64_image(response["images"][0])

@NodeFactory.register("chainner:stable_diffusion_extension:controlnet_preprocess_canny")
class ControlNetCannyPreprocessNode(NodeBase):
    def __init__(self):

        super().__init__()
        self.description = "ControlNet Canny detector"
        self.inputs = [
            ImageInput(),
            SliderInput("Annotator resolution", minimum=64, default=512, maximum=2048),
            SliderInput("Canny low threshold", minimum=0, default=100, maximum=255),
            SliderInput("Canny high threshold", minimum=0, default=200, maximum=255),
        ]
        self.outputs = [
            ImageOutput(image_type="Input0", channels=None)
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Canny"
        self.icon = "BsFillImageFill"
        self.sub = "Preprocess"

    def run(self, image, arg1, arg2, arg3) -> np.ndarray:


        response = post("/controlnet/detect",
            {
                "controlnet_module": "canny",
                "controlnet_input_images": [encode_base64_image(image)],
                "controlnet_processor_res": arg1,
                 "controlnet_threshold_a": arg2,
                "controlnet_threshold_b": arg3
            }
        )
        return decode_base64_image(response["images"][0])

@NodeFactory.register("chainner:stable_diffusion_extension:controlnet_preprocess_openpose")
class ControlNetOpenposePreprocessNode(NodeBase):
    def __init__(self):

        super().__init__()
        self.description = "ControlNet OpenPose detector"
        self.inputs = [
            ImageInput(),
            SliderInput("Annotator resolution", minimum=64, default=512, maximum=2048),
        ]
        self.outputs = [
            ImageOutput(image_type="Input0", channels=None)
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "OpenPose"
        self.icon = "BsFillImageFill"
        self.sub = "Preprocess"

    def run(self, image, arg1) -> np.ndarray:

        response = post("/controlnet/detect",
            {
                "controlnet_module": "openpose",
                "controlnet_input_images": [encode_base64_image(image)],
                "controlnet_processor_res": arg1,
            }
        )
        return decode_base64_image(response["images"][0])

@NodeFactory.register("chainner:stable_diffusion_extension:controlnet_preprocess_segmantation")
class ControlNetSegmantationPreprocessNode(NodeBase):
    def __init__(self):

        super().__init__()
        self.description = "ControlNet Openpose detector"
        self.inputs = [
            ImageInput(),
            SliderInput("Annotator resolution", minimum=64, default=512, maximum=2048),
        ]
        self.outputs = [
            ImageOutput(image_type="Input0", channels=None)
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Segmantation"
        self.icon = "BsFillImageFill"
        self.sub = "Preprocess"

    def run(self, image, arg1) -> np.ndarray:


        response = post("/controlnet/detect",
            {
                "controlnet_module": "openpose",
                "controlnet_input_images": [encode_base64_image(image)],
                "controlnet_processor_res": arg1,
            }
        )
        return decode_base64_image(response["images"][0])
