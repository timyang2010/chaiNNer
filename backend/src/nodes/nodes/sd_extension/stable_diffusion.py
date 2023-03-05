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
from ...properties.outputs import ImageOutput
from ...utils.seed import Seed
from ...utils.utils import get_h_w_c
from . import category as ExternalStableDiffusionCategory

verify_api_connection()



def load_sd_model(model: Enum):
    post("/sdapi/v1/options", {
                    "sd_model_checkpoint":model.value
                })


@NodeFactory.register("chainner:stable_diffusion_extension:txt2img")
class Txt2ImgAdvanced(NodeBase):
    def __init__(self):
        super().__init__()
        SD_MODELS =  Enum("StableDiffusionModels", {m["model_name"].replace(".", "_").replace("-", "_"):m["title"] for m in get("/sdapi/v1/sd-models")})

        self.description = "Generate an image using Automatic1111"
        self.inputs = [

            TextInput("Prompt").make_optional(),
            TextInput("Negative Prompt").make_optional(),
            group("seed")(SeedInput()),
            SliderInput("Steps", minimum=1, default=20, maximum=150),
            EnumInput(
                SamplerName,
                default_value=SamplerName.EULER,
                option_labels=SAMPLER_NAME_LABELS,
            ),
            SliderInput(
                "CFG Scale",
                minimum=1,
                default=7,
                maximum=20,
                controls_step=0.1,
                precision=1,
            ),
            SliderInput(
                "Width",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ).with_id(6),
            SliderInput(
                "Height",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ).with_id(7),
            BoolInput("Seamless Edges", default=False),
            TextInput("ControlNet0", placeholder="", default="").make_optional(),
            TextInput("ControlNet1", placeholder="", default="").make_optional(),
            TextInput("ControlNet2", placeholder="", default="").make_optional(),
            EnumInput(SD_MODELS),
        ]
        self.outputs = [
            ImageOutput(
                image_type="""def nearest_valid(n: number) = int & floor(n / 8) * 8;
                Image {
                    width: nearest_valid(Input6),
                    height: nearest_valid(Input7)
                }""",
                channels=3,
            ),
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Text to Image"
        self.icon = "MdChangeCircle"
        self.sub = "Automatic1111"

    @cached
    def run(
        self,

        prompt: Optional[str],
        negative_prompt: Optional[str],
        seed: Seed,
        steps: int,
        sampler_name: SamplerName,
        cfg_scale: float,
        width: int,
        height: int,
        tiling: bool,
        controlnet_0: str,
        controlnet_1: str,
        controlnet_2: str,
        model: Enum,
    ) -> np.ndarray:
        width, height = nearest_valid_size(
            width, height
        )  # This cooperates with the "image_type" of the ImageOutput
        load_sd_model(model)
        request_data = {
            "prompt": prompt or "",
            "negative_prompt": negative_prompt or "",
            "seed": seed.to_u32(),
            "steps": steps,
            "sampler_name": sampler_name.value,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "tiling": tiling,
            "controlnet_units":([
                json.loads(cu) for cu in [controlnet_0, controlnet_1, controlnet_2] if (cu!="")
            ])
        }
        response = post(path="/controlnet/txt2img", json_data=request_data)
        result = decode_base64_image(response["images"][0])
        h, w, _ = get_h_w_c(result)
        assert (w, h) == (
            width,
            height,
        ), f"Expected the returned image to be {width}x{height}px but found {w}x{h}px instead "
        return result


@NodeFactory.register("chainner:stable_diffusion_extension:img2img")
class Img2ImgAdvanced(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Modify an image using Automatic1111"
        SD_MODELS =  Enum("StableDiffusionModels", {m["model_name"].replace(".", "_").replace("-", "_"):m["title"] for m in get("/sdapi/v1/sd-models")})

        self.inputs = [

            ImageInput(),
            TextInput("Prompt").make_optional(),
            TextInput("Negative Prompt").make_optional(),
            SliderInput(
                "Denoising Strength",
                minimum=0,
                default=0.75,
                maximum=1,
                slider_step=0.01,
                controls_step=0.1,
                precision=2,
            ),
            group("seed")(SeedInput()),
            SliderInput("Steps", minimum=1, default=20, maximum=150),
            EnumInput(
                SamplerName,
                default_value=SamplerName.EULER,
                option_labels=SAMPLER_NAME_LABELS,
            ),
            SliderInput(
                "CFG Scale",
                minimum=1,
                default=7,
                maximum=20,
                controls_step=0.1,
                precision=1,
            ),
            EnumInput(
                ResizeMode,
                default_value=ResizeMode.JUST_RESIZE,
                option_labels=RESIZE_MODE_LABELS,
            ).with_id(10),
            SliderInput(
                "Width",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ).with_id(8),
            SliderInput(
                "Height",
                minimum=64,
                default=512,
                maximum=2048,
                slider_step=8,
                controls_step=8,
            ).with_id(9),
            BoolInput("Seamless Edges", default=False),
            TextInput("ControlNet0", placeholder="", default="").make_optional(),
            TextInput("ControlNet1", placeholder="", default="").make_optional(),
            TextInput("ControlNet2", placeholder="", default="").make_optional(),
            EnumInput(SD_MODELS),
        ]
        self.outputs = [
            ImageOutput(
                image_type="""def nearest_valid(n: number) = int & floor(n / 8) * 8;
                Image {
                    width: nearest_valid(Input8),
                    height: nearest_valid(Input9)
                }""",
                channels=3,
            ),
        ]

        self.category = ExternalStableDiffusionCategory
        self.name = "Image to Image"
        self.icon = "MdChangeCircle"
        self.sub = "Automatic1111"

    @cached
    def run(
        self,
        image: np.ndarray,
        prompt: Optional[str],
        negative_prompt: Optional[str],
        denoising_strength: float,
        seed: Seed,
        steps: int,
        sampler_name: SamplerName,
        cfg_scale: float,
        resize_mode: ResizeMode,
        width: int,
        height: int,
        tiling: bool,
        controlnet_0: str,
        controlnet_1: str,
        controlnet_2: str,
        model: Enum,
    ) -> np.ndarray:
        width, height = nearest_valid_size(
            width, height
        )  # This cooperates with the "image_type" of the ImageOutput
        load_sd_model(model)
        request_data = {
            "init_images": [encode_base64_image(image)],
            "prompt": prompt or "",
            "negative_prompt": negative_prompt or "",
            "denoising_strength": denoising_strength,
            "seed": seed.to_u32(),
            "steps": steps,
            "sampler_name": sampler_name.value,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "resize_mode": resize_mode.value,
            "tiling": tiling,
            "controlnet_units":([
                json.loads(cu) for cu in [controlnet_0, controlnet_1, controlnet_2] if (cu!="")
            ])
        }
        response = post(path="/controlnet/img2img", json_data=request_data)
        result = decode_base64_image(response["images"][0])
        h, w, _ = get_h_w_c(result)
        assert (w, h) == (
            width,
            height,
        ), f"Expected the returned image to be {width}x{height}px but found {w}x{h}px instead "
        return result

