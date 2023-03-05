from __future__ import annotations

from enum import Enum

import numpy as np

from ...impl.external_stable_diffusion import post, verify_api_connection
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import ImageInput, TextAreaInput
from ...properties.outputs import LargeImageOutput, TextOutput
from . import category as ExternalStableDiffusionCategory

verify_api_connection()



def load_sd_model(model: Enum):
    post("/sdapi/v1/options", {
                    "sd_model_checkpoint":model.value
                })

@NodeFactory.register("chainner:stable_diffusion_extension:view")
class ImViewNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "See an inline preview of the image in the editor."
        self.inputs = [ImageInput()]
        self.outputs = [
            LargeImageOutput("Preview", image_type="Input0", has_handle=True)
        ]
        self.category = ExternalStableDiffusionCategory
        self.name = "View Image"
        self.icon = "BsEyeFill"
        self.sub = "Input & Output"

        self.side_effects = True

    def run(self, img: np.ndarray):
        return img


@NodeFactory.register("chainner:stable_diffusion_extension:textarea")
class TextAreaValueNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Outputs the given text."
        self.inputs = [
            TextAreaInput("Text"),
        ]
        self.outputs = [
            TextOutput("Text", output_type="toString(Input0)"),
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "TextArea"
        self.icon = "MdTextFields"
        self.sub = "Value"

    def run(self, text: str) -> str:
        return text
