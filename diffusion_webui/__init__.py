from diffusion_webui.diffusion_models.stable_diffusion import (
    StableDiffusionText2ImageGenerator,
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
)

from diffusion_webui.diffusion_models.controlnet.controlnet_inpaint import (
    StableDiffusionControlNetInpaintCannyGenerator,
    StableDiffusionControlInpaintNetDepthGenerator,
    StableDiffusionControlNetInpaintHedGenerator,
    StableDiffusionControlNetInpaintMlsdGenerator,
    StableDiffusionControlNetInpaintPoseGenerator,
    StableDiffusionControlNetInpaintScribbleGenerator,
    StableDiffusionControlNetInpaintSegGenerator,
)   

from diffusion_webui.diffusion_models.controlnet import (
    StableDiffusionControlNetCannyGenerator,
    StableDiffusionControlNetDepthGenerator,
    StableDiffusionControlNetHEDGenerator,
    StableDiffusionControlNetMLSDGenerator,
    StableDiffusionControlNetNormalGenerator,
    StableDiffusionControlNetPoseGenerator,
    StableDiffusionControlNetScribbleGenerator,
    StableDiffusionControlNetSegGenerator,
)


__version__ = "2.2.0"
