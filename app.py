import gradio as gr

from diffusion_webui.helpers import (
    StableDiffusionControlNetCannyGenerator,
    StableDiffusionControlNetDepthGenerator,
    StableDiffusionControlNetHEDGenerator,
    StableDiffusionControlNetMLSDGenerator,
    StableDiffusionControlNetPoseGenerator,
    StableDiffusionControlNetScribbleGenerator,
    StableDiffusionControlNetSegGenerator,
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
    StableDiffusionText2ImageGenerator,
    StableDiffusionControlInpaintNetCannyGenerator,
)


def main():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text2Img"):
                    StableDiffusionText2ImageGenerator.app()
                with gr.Tab("Img2Img"):
                    StableDiffusionImage2ImageGenerator.app()
                with gr.Tab("Inpaint"):
                    StableDiffusionInpaintGenerator.app()
                with gr.Tab("ControlNet"):
                    with gr.Tab("Canny"):
                        StableDiffusionControlNetCannyGenerator.app()
                    with gr.Tab("Depth"):
                        StableDiffusionControlNetDepthGenerator.app()
                    with gr.Tab("HED"):
                        StableDiffusionControlNetHEDGenerator.app()
                    with gr.Tab("MLSD"):
                        StableDiffusionControlNetMLSDGenerator.app()
                    with gr.Tab("Pose"):
                        StableDiffusionControlNetPoseGenerator.app()
                    with gr.Tab("Scribble"):
                        StableDiffusionControlNetScribbleGenerator.app()
                    with gr.Tab("Seg"):
                        StableDiffusionControlNetSegGenerator.app()
                with gr.Tab("ControlNet Inpaint"):
                    StableDiffusionControlInpaintNetCannyGenerator.app()
                    
    app.queue(concurrency_count=3)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    main()