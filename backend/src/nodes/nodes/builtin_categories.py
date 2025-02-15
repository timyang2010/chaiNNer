from .external_stable_diffusion import category as ExternalStableDiffusionCategory
from .image import category as ImageCategory
from .image_adjustment import category as ImageAdjustmentCategory
from .image_channel import category as ImageChannelCategory
from .image_dimension import category as ImageDimensionCategory
from .image_filter import category as ImageFilterCategory
from .image_utility import category as ImageUtilityCategory
from .material_textures import category as MaterialTexturesCategory
from .ncnn import category as NCNNCategory
from .onnx import category as ONNXCategory
from .pytorch import category as PyTorchCategory
from .sd_extension import category as SDExtensionCategory
from .utility import category as UtilityCategory

builtin_categories = [
    SDExtensionCategory,
    ImageCategory,
    ImageDimensionCategory,
    ImageAdjustmentCategory,
    ImageFilterCategory,
    ImageUtilityCategory,
    ImageChannelCategory,
    MaterialTexturesCategory,
    UtilityCategory,
    PyTorchCategory,
    NCNNCategory,
    ONNXCategory,
    ExternalStableDiffusionCategory,
]
category_order = [x.name for x in builtin_categories]
