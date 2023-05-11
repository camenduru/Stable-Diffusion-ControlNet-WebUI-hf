import gradio as gr
import numpy as np
import paddle
from controlnet_aux import MLSDdetector
from ppdiffusers import ControlNetModel
from PIL import Image

from diffusion_webui.diffusion_models.controlnet.controlnet_inpaint.pipeline_stable_diffusion_controlnet_inpaint import (
    StableDiffusionControlNetInpaintPipeline,
)
from diffusion_webui.utils.model_list import (
    controlnet_mlsd_model_list,
    stable_inpiant_model_list,
)
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)

# https://github.com/mikonvergence/ControlNetInpaint


class StableDiffusionControlNetInpaintMlsdGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self, stable_model_path, controlnet_model_path, scheduler):
        if self.pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path, paddle_dtype=paddle.float16
            )
            self.pipe = (
                StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    pretrained_model_name_or_path=stable_model_path,
                    controlnet=controlnet,
                    safety_checker=None,
                    paddle_dtype=paddle.float16,
                )
            )

        self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def load_image(self, image_path):
        image = np.array(image_path)
        image = Image.fromarray(image)
        return image

    def controlnet_inpaint_mlsd(self, image_path: str):
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        image = image_path["image"].convert("RGB").resize((512, 512))
        image = np.array(image)
        image = mlsd(image)

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

        control_image = self.controlnet_inpaint_mlsd(image_path=image_path)

        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler=scheduler,
        )

        if not seed_generator == -1:
            paddle.seed(seed_generator)

        output = pipe(
            prompt=prompt,
            image=normal_image,
            mask_image=mask_image,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        return output

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    controlnet_mlsd_inpaint_image_file = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                    )
                    controlnet_mlsd_inpaint_prompt = gr.Textbox(
                        lines=1, placeholder="Prompt", show_label=False
                    )
                    controlnet_mlsd_inpaint_negative_prompt = gr.Textbox(
                        lines=1,
                        show_label=False,
                        placeholder="Negative Prompt",
                    )
                    with gr.Row():
                        with gr.Column():
                            controlnet_mlsd_inpaint_stable_model_id = gr.Dropdown(
                                choices=stable_inpiant_model_list,
                                value=stable_inpiant_model_list[0],
                                label="Stable Model Id",
                            )
                            controlnet_mlsd_inpaint_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            controlnet_mlsd_inpaint_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Step",
                            )
                            controlnet_mlsd_inpaint_num_images_per_prompt = gr.Slider(
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=1,
                                label="Number Of Images",
                            )
                        with gr.Row():
                            with gr.Column():
                                controlnet_mlsd_inpaint_model_id = gr.Dropdown(
                                    choices=controlnet_mlsd_model_list,
                                    value=controlnet_mlsd_model_list[0],
                                    label="Controlnet Model Id",
                                )
                                controlnet_mlsd_inpaint_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[5],
                                    label="Scheduler",
                                )
                                controlnet_mlsd_inpaint_controlnet_conditioning_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.1,
                                    value=0.5,
                                    label="Controlnet Conditioning Scale",
                                )
                                controlnet_mlsd_inpaint_seed_generator = gr.Number(
                                    value=-1,
                                    label="Seed Generator",
                                )

                    controlnet_mlsd_inpaint_predict = gr.Button(
                        value="Generator"
                    )

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

            controlnet_mlsd_inpaint_predict.click(
                fn=StableDiffusionControlNetInpaintMlsdGenerator().generate_image,
                inputs=[
                    controlnet_mlsd_inpaint_image_file,
                    controlnet_mlsd_inpaint_stable_model_id,
                    controlnet_mlsd_inpaint_model_id,
                    controlnet_mlsd_inpaint_prompt,
                    controlnet_mlsd_inpaint_negative_prompt,
                    controlnet_mlsd_inpaint_num_images_per_prompt,
                    controlnet_mlsd_inpaint_guidance_scale,
                    controlnet_mlsd_inpaint_num_inference_step,
                    controlnet_mlsd_inpaint_controlnet_conditioning_scale,
                    controlnet_mlsd_inpaint_scheduler,
                    controlnet_mlsd_inpaint_seed_generator,
                ],
                outputs=[output_image],
            )
