from diffusers import DiffusionPipeline, DDIMScheduler
from PIL import Image
import imageio
import torch

import gradio as gr

stable_model_list = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2",
    "stabilityai/stable-diffusion-2-base",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base"
]

stable_inpiant_model_list = [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting"
]

stable_prompt_list = [
        "a photo of a man.",
        "a photo of a girl."
    ]

stable_negative_prompt_list = [
        "bad, ugly",
        "deformed"
    ]


def resize(height,img):
  baseheight = height
  img = Image.open(img)
  hpercent = (baseheight/float(img.size[1]))
  wsize = int((float(img.size[0])*float(hpercent)))
  img = img.resize((wsize,baseheight), Image.Resampling.LANCZOS)
  return img

def img_preprocces(source_img, prompt, negative_prompt):
    imageio.imwrite("data.png", source_img["image"])
    imageio.imwrite("data_mask.png", source_img["mask"])
    src = resize(512, "data.png")
    src.save("src.png")
    mask = resize(512, "data_mask.png")  
    mask.save("mask.png")
    return src, mask

def stable_diffusion_inpaint(
    image_path:str,
    model_path:str,
    prompt:str,
    negative_prompt:str,
    guidance_scale:int,
    num_inference_step:int,
    ):

    image, mask_image = img_preprocces(image_path, prompt, negative_prompt)
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    output = pipe(
        prompt = prompt,
        image = image,
        mask_image=mask_image,
        negative_prompt = negative_prompt,
        num_inference_steps = num_inference_step,
        guidance_scale = guidance_scale,
    ).images

    return output[0]


def stable_diffusion_inpaint_app():
    with gr.Tab('Inpaint'):
        inpaint_image_file = gr.Image(
            source="upload", 
            type="numpy", 
            tool="sketch", 
            elem_id="source_container"
        )

        inpaint_model_id = gr.Dropdown(
            choices=stable_inpiant_model_list, 
            value=stable_inpiant_model_list[0], 
            label='Inpaint Model Id'
        )

        inpaint_prompt = gr.Textbox(
            lines=1, 
            value=stable_prompt_list[0], 
            label='Prompt'
        )

        inpaint_negative_prompt = gr.Textbox(
            lines=1, 
            value=stable_negative_prompt_list[0], 
            label='Negative Prompt'
        )

        with gr.Accordion("Advanced Options", open=False):
            inpaint_guidance_scale = gr.Slider(
                minimum=0.1, 
                maximum=15, 
                step=0.1, 
                value=7.5, 
                label='Guidance Scale'
            )

            inpaint_num_inference_step = gr.Slider(
                minimum=1, 
                maximum=100, 
                step=1, 
                value=50, 
                label='Num Inference Step'
            )

        inpaint_predict = gr.Button(value='Generator')
    
    variables = {
        "image_path": inpaint_image_file,
        "model_path": inpaint_model_id,
        "prompt": inpaint_prompt,
        "negative_prompt": inpaint_negative_prompt,
        "guidance_scale": inpaint_guidance_scale,
        "num_inference_step": inpaint_num_inference_step,
        "predict": inpaint_predict
    }

    return variables
