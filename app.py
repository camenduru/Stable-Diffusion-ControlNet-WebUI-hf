
from utils.image2image import stable_diffusion_img2img
from utils.text2image import stable_diffusion_text2img
from utils.inpaint import stable_diffusion_inpaint

from controlnet.controlnet_canny import stable_diffusion_controlnet_img2img
from controlnet.controlnet_depth import stable_diffusion_controlnet_img2img
from controlnet.controlnet_hed import stable_diffusion_controlnet_img2img
from controlnet.controlnet_mlsd import stable_diffusion_controlnet_img2img
from controlnet.controlnet_pose import stable_diffusion_controlnet_img2img
from controlnet.controlnet_scribble import stable_diffusion_controlnet_img2img
from controlnet.controlnet_seg import stable_diffusion_controlnet_img2img


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
app = gr.Blocks()
with app:
    gr.Markdown("# **<h2 align='center'>Stable Diffusion WebUI<h2>**")
    gr.Markdown(
        """
        <h5 style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </h5>
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Tab('Text2Image'):
                text2image_model_id = gr.Dropdown(
                    choices=stable_model_list, 
                    value=stable_model_list[0], 
                    label='Text-Image Model Id'
                )

                text2image_prompt = gr.Textbox(
                    lines=1, 
                    value=stable_prompt_list[0], 
                    label='Prompt'
                )

                text2image_negative_prompt = gr.Textbox(
                    lines=1, 
                    value=stable_negative_prompt_list[0], 
                    label='Negative Prompt'
                )

                with gr.Accordion("Advanced Options", open=False):
                    text2image_guidance_scale = gr.Slider(
                        minimum=0.1, 
                        maximum=15, 
                        step=0.1, 
                        value=7.5, 
                        label='Guidance Scale'
                    )

                    text2image_num_inference_step = gr.Slider(
                        minimum=1, 
                        maximum=100, 
                        step=1, 
                        value=50, 
                        label='Num Inference Step'
                    )

                    text2image_height = gr.Slider(
                        minimum=128, 
                        maximum=1280, 
                        step=32, 
                        value=512, 
                        label='Tile Height'
                    )

                    text2image_width = gr.Slider(
                        minimum=128, 
                        maximum=1280, 
                        step=32, 
                        value=768, 
                        label='Tile Height'
                    )

                text2image_predict = gr.Button(value='Generator')


            with gr.Tab('Image2Image'):
                image2image2_image_file = gr.Image(label='Image')

                image2image_model_id = gr.Dropdown(
                    choices=stable_model_list, 
                    value=stable_model_list[0], 
                    label='Image-Image Model Id'
                )

                image2image_prompt = gr.Textbox(
                    lines=1, 
                    value=stable_prompt_list[0], 
                    label='Prompt'
                )

                image2image_negative_prompt = gr.Textbox(
                    lines=1, 
                    value=stable_negative_prompt_list[0], 
                    label='Negative Prompt'
                )

                with gr.Accordion("Advanced Options", open=False):
                    image2image_guidance_scale = gr.Slider(
                        minimum=0.1, 
                        maximum=15, 
                        step=0.1, 
                        value=7.5, 
                        label='Guidance Scale'
                    )

                    image2image_num_inference_step = gr.Slider(
                        minimum=1, 
                        maximum=100, 
                        step=1, 
                        value=50, 
                        label='Num Inference Step'
                    )

                image2image_predict = gr.Button(value='Generator')

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


            with gr.Tab('ControlNet'):
                with gr.Tab('Canny'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_model_list, 
                        value=stable_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_canny_predict = gr.Button(value='Generator')

                with gr.Tab('Hed'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list, 
                        value=stable_inpiant_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_hed_predict = gr.Button(value='Generator')

                with gr.Tab('MLSD line'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list, 
                        value=stable_inpiant_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_mlsd_predict = gr.Button(value='Generator')

                with gr.Tab('Segmentation'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list, 
                        value=stable_inpiant_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_seg_predict = gr.Button(value='Generator')

                with gr.Tab('Depth'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list, 
                        value=stable_inpiant_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_depth_predict = gr.Button(value='Generator')

                with gr.Tab('Scribble'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list, 
                        value=stable_inpiant_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_scribble_predict = gr.Button(value='Generator')

                with gr.Tab('Pose'):
                    controlnet_image_file = gr.Image(label='Image')

                    controlnet_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list, 
                        value=stable_inpiant_model_list[0], 
                        label='Stable Model Id'
                    )

                    controlnet_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_prompt_list[0], 
                        label='Prompt'
                    )

                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, 
                        value=stable_negative_prompt_list[0], 
                        label='Negative Prompt'
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        controlnet_guidance_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=15, 
                            step=0.1, 
                            value=7.5, 
                            label='Guidance Scale'
                        )

                        controlnet_num_inference_step = gr.Slider(
                            minimum=1, 
                            maximum=100, 
                            step=1, 
                            value=50, 
                            label='Num Inference Step'
                        )

                    controlnet_pose_predict = gr.Button(value='Generator')

    with gr.Tab('Generator'):
        with gr.Column():
            output_image = gr.Image(label='Image')
            
        text2image_predict.click(
            fn = stable_diffusion_text2img,
            inputs = [
                text2image_model_id,
                text2image_prompt, 
                text2image_negative_prompt, 
                text2image_guidance_scale,
                text2image_num_inference_step, 
                text2image_height,
                text2image_width,
            ],
            outputs = [output_image],
        )  

        image2image_predict.click(
            fn = stable_diffusion_img2img,
            inputs = [
                image2image2_image_file,
                image2image_model_id, 
                image2image_prompt, 
                image2image_negative_prompt,
                image2image_guidance_scale, 
                image2image_num_inference_step,
            ],
            outputs = [output_image],
        )  

        inpaint_predict.click(
            fn = stable_diffusion_inpaint,
            inputs = [
                inpaint_image_file,
                inpaint_model_id, 
                inpaint_prompt, 
                inpaint_negative_prompt,
                inpaint_guidance_scale, 
                inpaint_num_inference_step,
            ],
            outputs = [output_image],
        )  


        controlnet_canny_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  

        controlnet_hed_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  
        
        controlnet_mlsd_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  
        
        controlnet_seg_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  
        
        controlnet_depth_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  
        
        controlnet_scribble_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  
        
        controlnet_pose_predict.click(
            fn = stable_diffusion_controlnet_img2img,
            inputs = [
                controlnet_image_file,
                controlnet_model_id, 
                controlnet_prompt, 
                controlnet_negative_prompt,
                controlnet_guidance_scale, 
                controlnet_num_inference_step,
            ],
            outputs = [output_image],
        )  

app.launch()          