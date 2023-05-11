from diffusion_webui.diffusion_models.controlnet import (
    StableDiffusionControlNetCannyGenerator,
    StableDiffusionControlNetHEDGenerator,
    StableDiffusionControlNetMLSDGenerator,
    StableDiffusionControlNetPoseGenerator,
    StableDiffusionControlNetScribbleGenerator,
    # StableDiffusionControlNetNormalGenerator,
    # StableDiffusionControlNetSegGenerator,
    # StableDiffusionControlNetDepthGenerator,
    # StableDiffusionControlNetPix2PixGenerator,
    # StableDiffusionControlNetShuffleGenerator,
    # StableDiffusionControlNetSoftEdgeGenerator,
    # StableDiffusionControlNetLineArtGenerator,
    # StableDiffusionControlNetLineArtAnimeGenerator,
)
# from diffusion_webui.diffusion_models.controlnet.controlnet_inpaint import (
#     StableDiffusionControlInpaintNetDepthGenerator,
#     StableDiffusionControlNetInpaintCannyGenerator,
#     StableDiffusionControlNetInpaintHedGenerator,
#     StableDiffusionControlNetInpaintMlsdGenerator,
#     StableDiffusionControlNetInpaintPoseGenerator,
#     StableDiffusionControlNetInpaintScribbleGenerator,
#     StableDiffusionControlNetInpaintSegGenerator,
# )
from diffusion_webui.diffusion_models.stable_diffusion import (
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
    StableDiffusionText2ImageGenerator,
)

__version__ = "2.4.0"
