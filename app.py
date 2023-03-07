from diffusion_webui.controlnet.controlnet_canny import stable_diffusion_controlnet_canny_app, stable_diffusion_controlnet_canny
from diffusion_webui.controlnet.controlnet_depth import stable_diffusion_controlnet_depth_app, stable_diffusion_controlnet_depth
from diffusion_webui.controlnet.controlnet_hed import stable_diffusion_controlnet_hed_app, stable_diffusion_controlnet_hed
from diffusion_webui.controlnet.controlnet_mlsd import stable_diffusion_controlnet_mlsd_app, stable_diffusion_controlnet_mlsd
from diffusion_webui.controlnet.controlnet_pose import stable_diffusion_controlnet_pose_app, stable_diffusion_controlnet_pose
from diffusion_webui.controlnet.controlnet_scribble import stable_diffusion_controlnet_scribble_app, stable_diffusion_controlnet_scribble
from diffusion_webui.controlnet.controlnet_seg import stable_diffusion_controlnet_seg_app, stable_diffusion_controlnet_seg

from diffusion_webui.stable_diffusion.text2img_app import stable_diffusion_text2img_app, stable_diffusion_text2img
from diffusion_webui.stable_diffusion.img2img_app import stable_diffusion_img2img_app, stable_diffusion_img2img
from diffusion_webui.stable_diffusion.inpaint_app import stable_diffusion_inpaint_app, stable_diffusion_inpaint


import gradio as gr


app = gr.Blocks()
with app:
    gr.Markdown("# **<h1 align='center'>Stable Diffusion + ControlNet WebUI<h1>**")
    gr.Markdown(
        """
        <h4 style='text-align: center'>
        Follow me for more! 
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>
        </h4>
        """
    )
    with gr.Row():
        with gr.Column():
            text2image_app = stable_diffusion_text2img_app()
            img2img_app = stable_diffusion_img2img_app()
            inpaint_app = stable_diffusion_inpaint_app()
            
            with gr.Tab('ControlNet'):
                controlnet_canny_app = stable_diffusion_controlnet_canny_app()
                controlnet_hed_app = stable_diffusion_controlnet_hed_app()
                controlnet_mlsd_app = stable_diffusion_controlnet_mlsd_app()
                controlnet_depth_app = stable_diffusion_controlnet_depth_app()
                controlnet_pose_app = stable_diffusion_controlnet_pose_app()
                controlnet_scribble_app = stable_diffusion_controlnet_scribble_app()
                controlnet_seg_app = stable_diffusion_controlnet_seg_app()


        with gr.Tab('Output'):
            with gr.Column():
                output_image = gr.Image(label='Image')
                
        text2image_app['predict'].click(
            fn = stable_diffusion_text2img,
            inputs = [
                text2image_app['model_path'],
                text2image_app['prompt'], 
                text2image_app['negative_prompt'], 
                text2image_app['guidance_scale'],
                text2image_app['num_inference_step'], 
                text2image_app['height'],
                text2image_app['width'],
            ],
            outputs = [output_image],
        )  

        img2img_app['predict'].click(
            fn = stable_diffusion_img2img,
            inputs = [
                img2img_app['image_path'],
                img2img_app['model_path'], 
                img2img_app['prompt'], 
                img2img_app['negative_prompt'],
                img2img_app['guidance_scale'], 
                img2img_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  

        inpaint_app['predict'].click(
            fn = stable_diffusion_inpaint,
            inputs = [
                inpaint_app['image_path'],
                inpaint_app['model_path'], 
                inpaint_app['prompt'], 
                inpaint_app['negative_prompt'],
                inpaint_app['guidance_scale'], 
                inpaint_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  

        controlnet_canny_app['predict'].click(
            fn = stable_diffusion_controlnet_canny,
            inputs = [
                controlnet_canny_app['image_path'],
                controlnet_canny_app['model_path'], 
                controlnet_canny_app['prompt'], 
                controlnet_canny_app['negative_prompt'],
                controlnet_canny_app['guidance_scale'], 
                controlnet_canny_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  

        controlnet_hed_app['predict'].click(
            fn = stable_diffusion_controlnet_hed,
            inputs = [
                controlnet_hed_app['image_path'],
                controlnet_hed_app['model_path'], 
                controlnet_hed_app['prompt'], 
                controlnet_hed_app['negative_prompt'],
                controlnet_hed_app['guidance_scale'], 
                controlnet_hed_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  
        
        controlnet_mlsd_app['predict'].click(
            fn = stable_diffusion_controlnet_mlsd,
            inputs = [
                controlnet_mlsd_app['image_path'],
                controlnet_mlsd_app['model_path'], 
                controlnet_mlsd_app['prompt'], 
                controlnet_mlsd_app['negative_prompt'],
                controlnet_mlsd_app['guidance_scale'], 
                controlnet_mlsd_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  
        
        controlnet_depth_app['predict'].click(
            fn = stable_diffusion_controlnet_seg,
            inputs = [
                controlnet_depth_app['image_path'],
                controlnet_depth_app['model_path'], 
                controlnet_depth_app['prompt'], 
                controlnet_depth_app['negative_prompt'],
                controlnet_depth_app['guidance_scale'], 
                controlnet_depth_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  
        
        controlnet_pose_app['predict'].click(
            fn = stable_diffusion_controlnet_depth,
            inputs = [
                controlnet_pose_app['image_path'],
                controlnet_pose_app['model_path'], 
                controlnet_pose_app['prompt'], 
                controlnet_pose_app['negative_prompt'],
                controlnet_pose_app['guidance_scale'], 
                controlnet_pose_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  
        
        controlnet_scribble_app['predict'].click(
            fn = stable_diffusion_controlnet_scribble,
            inputs = [
                controlnet_scribble_app['image_path'],
                controlnet_scribble_app['model_path'], 
                controlnet_scribble_app['prompt'], 
                controlnet_scribble_app['negative_prompt'],
                controlnet_scribble_app['guidance_scale'], 
                controlnet_scribble_app['num_inference_step'],
            ],
            outputs = [output_image],
        )  
        
        controlnet_seg_app['predict'].click(
            fn = stable_diffusion_controlnet_pose,
            inputs = [
                controlnet_seg_app['image_path'],
                controlnet_seg_app['model_path'], 
                controlnet_seg_app['prompt'], 
                controlnet_seg_app['negative_prompt'],
                controlnet_seg_app['guidance_scale'], 
                controlnet_seg_app['num_inference_step'],
            ],
            outputs = [output_image],
        )

app.launch(debug=True)          