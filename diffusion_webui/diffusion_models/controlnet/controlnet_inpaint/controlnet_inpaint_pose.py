import gradio as gr
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel
from diffusion_webui.diffusion_models.controlnet.controlnet_inpaint.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from PIL import Image

from diffusion_webui.utils.model_list import (
    controlnet_pose_model_list,
    stable_inpiant_model_list,
)
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)

# https://github.com/mikonvergence/ControlNetInpaint


class StableDiffusionControlNetInpaintPoseGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self, stable_model_path, controlnet_model_path, scheduler):
        if self.pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=torch.float16
            )

            self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                pretrained_model_name_or_path=stable_model_path,
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16,
            )

        self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe
    
    def load_image(self, image_path):
        image = np.array(image_path)
        image = Image.fromarray(image_path)
        return image
    
    def controlnet_pose_inpaint(self, image_path: str):
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        image = image_path["image"].convert("RGB").resize((512, 512))
        image = np.array(image)
        image = openpose(image)

        return image

    def generate_image(
        self,
        image_path: str,
        stable_model_path: str,
        controlnet_model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        controlnet_conditioning_scale: int,
        scheduler: str,
        seed_generator: int,
    ):
        normal_image = image_path["image"].convert("RGB").resize((512, 512))
        mask_image = image_path["mask"].convert("RGB").resize((512, 512))
        
        normal_image = self.load_image(image_path=normal_image)
        mask_image = self.load_image(image_path=mask_image)
        
        controlnet_image = self.controlnet_pose_inpaint(image_path=image_path)

        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler=scheduler,
        )

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        output = pipe(
            prompt=prompt,
            image=normal_image,
            mask_image=mask_image,
            controlnet_image=controlnet_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images

        return output

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    controlnet_pose_inpaint_image_file = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                    )

                    controlnet_pose_inpaint_prompt = gr.Textbox(
                        lines=1, placeholder="Prompt", show_label=False
                    )

                    controlnet_pose_inpaint_negative_prompt = gr.Textbox(
                        lines=1,
                        show_label=False,
                        placeholder="Negative Prompt",
                    )
                    with gr.Row():
                        with gr.Column():
                            controlnet_pose_inpaint_stable_model_id = (
                                gr.Dropdown(
                                    choices=stable_inpiant_model_list,
                                    value=stable_inpiant_model_list[0],
                                    label="Stable Model Id",
                                )
                            )

                            controlnet_pose_inpaint_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )

                            controlnet_pose_inpaint_num_inference_step = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=50,
                                    label="Num Inference Step",
                                )
                            )
                            controlnet_pose_inpaint_num_images_per_prompt = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=1,
                                    label="Number Of Images",
                                )
                            )
                        with gr.Row():
                            with gr.Column():
                                controlnet_pose_inpaint_model_id = gr.Dropdown(
                                    choices=controlnet_pose_model_list,
                                    value=controlnet_pose_model_list[0],
                                    label="Controlnet Model Id",
                                )
                                controlnet_pose_inpaint_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[0],
                                    label="Scheduler",
                                )
                                controlnet_pose_inpaint_controlnet_conditioning_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.1,
                                    value=0.5,
                                    label="Controlnet Conditioning Scale",
                                )

                                controlnet_pose_inpaint_seed_generator = (
                                    gr.Slider(
                                        minimum=0,
                                        maximum=1000000,
                                        step=1,
                                        value=0,
                                        label="Seed Generator",
                                    )
                                )

                    controlnet_pose_inpaint_predict = gr.Button(
                        value="Generator"
                    )

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

            controlnet_pose_inpaint_predict.click(
                fn=StableDiffusionControlNetInpaintPoseGenerator().generate_image,
                inputs=[
                    controlnet_pose_inpaint_image_file,
                    controlnet_pose_inpaint_stable_model_id,
                    controlnet_pose_inpaint_model_id,
                    controlnet_pose_inpaint_prompt,
                    controlnet_pose_inpaint_negative_prompt,
                    controlnet_pose_inpaint_num_images_per_prompt,
                    controlnet_pose_inpaint_guidance_scale,
                    controlnet_pose_inpaint_num_inference_step,
                    controlnet_pose_inpaint_controlnet_conditioning_scale,
                    controlnet_pose_inpaint_scheduler,
                    controlnet_pose_inpaint_seed_generator,
                ],
                outputs=[output_image],
            )