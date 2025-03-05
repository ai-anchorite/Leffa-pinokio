import gradio as gr
import torch
import numpy as np
import os
import subprocess # os and subprocess for open folder
import datetime
from pathlib import Path
import gc
import warnings
import sys  # Add this import at the top with the others

# Suppress specific diffusers deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers.models.resnet')
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers.models.activations')
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers.models.downsampling')
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers.models.upsampling')
# Suppress PyTorch meshgrid warning
warnings.filterwarnings('ignore', message='torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument')
# Suppress persistant UNet config attributes warning (using exact message)
warnings.filterwarnings('ignore', category=UserWarning, message="The config attributes {'decay': 0.9999, 'inv_gamma': 1.0, 'min_decay': 0.0, 'optimization_step': 37000, 'power': 0.6666666666666666, 'update_after_step': 0, 'use_ema_warmup': False} were passed to UNet2DConditionModel, but are not expected and will be ignored. Please verify your config.json configuration file.")

from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


# Download base checkpoints (excluding pose transfer)
snapshot_download(
    repo_id="franciszzj/Leffa",
    local_dir="./ckpts",
    ignore_patterns=["*pose_transfer.pth"],  # Exclude pose transfer model
)


class LeffaPredictor(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize base components
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        # Initialize model holders as None
        self.vt_inference_hd = None
        self.vt_inference_dc = None
        self.pt_inference = None
        
        # Track current active model
        self.current_model = None
        
    def unload_all_models(self):
        """Completely unload all models from both GPU and RAM"""
        print("\nUnloading all models...")
        
        if self.vt_inference_hd is not None:
            self.vt_inference_hd.model.to("cpu")
            del self.vt_inference_hd.model
            del self.vt_inference_hd
            self.vt_inference_hd = None
            
        if self.vt_inference_dc is not None:
            self.vt_inference_dc.model.to("cpu")
            del self.vt_inference_dc.model
            del self.vt_inference_dc
            self.vt_inference_dc = None
            
        if self.pt_inference is not None:
            self.pt_inference.model.to("cpu")
            del self.pt_inference.model
            del self.pt_inference
            self.pt_inference = None
            
        self.current_model = None
        
        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()
        print("All models unloaded")
        
    def load_viton_hd(self):
        # Only unload if we're switching from a different model
        if self.current_model and self.current_model != "viton_hd":
            self.unload_all_models()
        
        if self.vt_inference_hd is None:
            print("Loading VITON-HD model...")
            vt_model_hd = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
                pretrained_model="./ckpts/virtual_tryon.pth",
                dtype="float16",
            )
            self.vt_inference_hd = LeffaInference(model=vt_model_hd)
            
        # Move to GPU if not already there
        if self.current_model != "viton_hd":
            print("Moving VITON-HD model to GPU...")
            self.vt_inference_hd.model.to(self.device)
            self.current_model = "viton_hd"
            print("VITON-HD model ready")
        
    def load_dress_code(self):
        # Only unload if we're switching from a different model
        if self.current_model and self.current_model != "dress_code":
            self.unload_all_models()
        
        if self.vt_inference_dc is None:
            print("Loading DressCode model...")
            vt_model_dc = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
                pretrained_model="./ckpts/virtual_tryon_dc.pth",
                dtype="float16",
            )
            self.vt_inference_dc = LeffaInference(model=vt_model_dc)
            
        # Move to GPU if not already there
        if self.current_model != "dress_code":
            print("Moving DressCode model to GPU...")
            self.vt_inference_dc.model.to(self.device)
            self.current_model = "dress_code"
            print("DressCode model ready")
        
    def load_pose_transfer(self):
        """Load pose transfer model, downloading if necessary"""
        # Only unload if we're switching from a different model
        if self.current_model and self.current_model != "pose_transfer":
            self.unload_all_models()
        
        if self.pt_inference is None:
            if not os.path.exists("./ckpts/pose_transfer.pth"):
                print("\nDownloading Pose Transfer model (20GB)...")
                snapshot_download(
                    repo_id="franciszzj/Leffa",
                    local_dir="./ckpts",
                    allow_patterns=["*pose_transfer.pth"],  # Only get pose transfer model
                )
                print("Download complete!")
                
            print("Loading Pose Transfer model...")
            pt_model = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
                pretrained_model="./ckpts/pose_transfer.pth",
                dtype="float16",
            )
            self.pt_inference = LeffaInference(model=pt_model)
            
        # Move to GPU if not already there
        if self.current_model != "pose_transfer":
            print("Moving Pose Transfer model to GPU...")
            self.pt_inference.model.to(self.device)
            self.current_model = "pose_transfer"
            print("Pose Transfer model ready")
        
    def cleanup_after_inference(self):
        """Cleanup after inference completion"""
        print("\nCleaning up after inference...")
        # Just clear CUDA cache, don't unload model
        torch.cuda.empty_cache()
        print("Memory cleaned up after inference")

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=-1,  # Default to random
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False
    ):
        # Use random seed if seed is -1
        if seed == -1:
            seed = torch.randint(0, 2147483647, (1,)).item()
            print(f"Using random seed: {seed}")
            
        # Open and resize the source image.
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # For virtual try-on, optionally preprocess the garment (reference) image.
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                # preprocess_garment_image returns a 768x1024 image.
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            # Otherwise, load the reference image.
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            elif vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            elif vt_model_type == "dress_code":
                inference = self.vt_inference_dc
        elif control_type == "pose_transfer":
            inference = self.pt_inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment, autosave=True):
        # Load appropriate model based on type
        if vt_model_type == "viton_hd":
            self.load_viton_hd()
        else:
            self.load_dress_code()
            
        try:
            gen_image, mask, densepose = self.leffa_predict(
                src_image_path,
                ref_image_path,
                "virtual_tryon",
                ref_acceleration,
                step,
                scale,
                seed,
                vt_model_type,
                vt_garment_type,
                vt_repaint,
                preprocess_garment,
            )
            if autosave:
                save_generated_image(gen_image, "virtual_tryon")
            return gen_image, mask, densepose
        finally:
            self.cleanup_after_inference()

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, autosave=True):
        # Load pose transfer model
        self.load_pose_transfer()
        
        try:
            gen_image, mask, densepose = self.leffa_predict(
                src_image_path,
                ref_image_path,
                "pose_transfer",
                ref_acceleration,
                step,
                scale,
                seed,
            )
            if autosave:
                save_generated_image(gen_image, "pose_transfer")
            return gen_image, mask, densepose
        finally:
            self.cleanup_after_inference()

def save_generated_image(image_array, process_type):
    """Save the generated image with datetime and process type prefix"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{process_type}_{timestamp}.png"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    if isinstance(image_array, np.ndarray):
        Image.fromarray(image_array.astype(np.uint8)).save(output_path)
    else:
        image_array.save(output_path)
    return str(output_path)
    
def open_output_folder():
    """Open the outputs folder in the system's file explorer"""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', output_dir])
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', output_dir])
        else:  # Linux
            subprocess.run(['xdg-open', output_dir])
    except Exception as e:
        print(f"Failed to open folder: {str(e)}")

def open_examples_folder(folder_type):
    """Open the examples folder in the system's file explorer"""
    example_dir = Path(f"./ckpts/examples/{folder_type}")
    example_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', example_dir])
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', example_dir])
        else:  # Linux
            subprocess.run(['xdg-open', example_dir])
    except Exception as e:
        print(f"Failed to open folder: {str(e)}")

# error handling for missing inputs
def validate_vt_inputs(src_image, ref_image):
    """Validate inputs for virtual try-on"""
    if src_image is None:
        print("Error: Please upload a person image")
        return False
    if ref_image is None:
        print("Error: Please upload a garment image")
        return False
    return True

def validate_pt_inputs(src_image, ref_image):
    """Validate inputs for pose transfer"""
    if ref_image is None:
        print("Error: Please upload a person image")
        return False
    if src_image is None:
        print("Error: Please upload a target pose image")
        return False
    return True

def vt_generate(*args):
    """Generate virtual try-on output with input validation"""
    if not validate_vt_inputs(args[0], args[1]):
        return None, None, None
    print("Generating virtual try-on image...")
    return leffa_predictor.leffa_predict_vt(*args)

def pt_generate(*args):
    """Generate pose transfer output with input validation"""
    if not validate_pt_inputs(args[0], args[1]):
        return None, None, None
    print("Generating pose transfer image...")
    return leffa_predictor.leffa_predict_pt(*args)


if __name__ == "__main__":
    leffa_predictor = LeffaPredictor()
    example_dir = "./ckpts/examples"
    person1_images = list_dir(f"{example_dir}/person1")
    person2_images = list_dir(f"{example_dir}/person2")
    garment_images = list_dir(f"{example_dir}/garment")

    # a couple of newlines ensuring the launch URL appears on a clean line
    print("\n\n")
    
    title = "## Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation"
    link = """[📚 Paper](https://arxiv.org/abs/2412.08486) - [🤖 Code](https://github.com/franciszzj/Leffa) - [🔥 Demo](https://huggingface.co/spaces/franciszzj/Leffa) - [🤗 Model](https://huggingface.co/franciszzj/Leffa)  
           
           """
    news = """More info can be found in the [GitHub repository](https://github.com/franciszzj/Leffa). Please leave a Star ⭐ if you like it!
           """
    description = "Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer)."
    note = "Note: The models used in the demo are trained solely on academic datasets. Virtual try-on uses VITON-HD/DressCode, and pose transfer uses DeepFashion."

    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
        with gr.Tab("Virtual Try-on  (32G RAM | 12GB VRAM)"):
            with gr.Row():
                with gr.Column():
                    vt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )
                    with gr.Row():
                        vt_open_folder = gr.Button("📂 Open Output Folder")   
                    with gr.Row():   
                        with gr.Accordion("Person Examples", open=False) as person_accordion:
                            gr.Examples(
                                inputs=vt_src_image,
                                examples_per_page=20,
                                examples=person1_images,
                            )
                            person1_add_btn = gr.Button("➕ Add More [requires restart to appear]", size="sm")
                            person1_add_btn.click(fn=lambda: open_examples_folder("person1"), inputs=[], outputs=[])

                with gr.Column():
                    vt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Garment Image",
                        width=512,
                        height=512,
                    )
                    preprocess_garment_checkbox = gr.Checkbox(
                        label="Preprocess Garment Image (PNG only)",
                        value=False
                    )
                    with gr.Accordion("Garment Examples", open=False) as garment_accordion:
                        gr.Examples(
                            inputs=vt_ref_image,
                            examples_per_page=20,
                            examples=garment_images,
                        )
                        garment_add_btn = gr.Button("➕ Add More", size="sm")
                        garment_add_btn.click(fn=lambda: open_examples_folder("garment"), inputs=[], outputs=[])

                with gr.Column():
                    vt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )
                    with gr.Row():
                        vt_gen_button = gr.Button("Generate", variant="primary")
                        with gr.Group():
                            autosave_checkbox = gr.Checkbox(label="Autosave", value=False)
                            save_button = gr.Button("Save Current Image", variant="huggingface")

                    with gr.Accordion("Advanced Options", open=False):
                        vt_model_type = gr.Radio(
                            label="Model Type",
                            choices=[("VITON-HD (Recommended)", "viton_hd"),
                                     ("DressCode (Experimental)", "dress_code")],
                            value="viton_hd",
                        )
                        vt_garment_type = gr.Radio(
                            label="Garment Type - lower/dress limited support",
                            choices=[("Upper", "upper_body"),
                                     ("Lower", "lower_body"),
                                     ("Dress", "dresses")],
                            value="upper_body",
                        )
                        vt_ref_acceleration = gr.Radio(
                            label="Accelerate Reference UNet (may slightly reduce performance)",
                            choices=[("True", True), ("False", False)],
                            value=False,
                        )
                        vt_repaint = gr.Radio(
                            label="Repaint Mode",
                            choices=[("True", True), ("False", False)],
                            value=False,
                        )
                        vt_step = gr.Number(
                            label="Inference Steps", minimum=30, maximum=100, step=1, value=30)
                        vt_scale = gr.Number(
                            label="Guidance Scale", minimum=0.1, maximum=5.0, step=0.1, value=2.5)
                        vt_seed = gr.Number(
                            label="Random Seed (-1 for random)", minimum=-1, maximum=2147483647, step=1, value=-1)

                    with gr.Accordion("Debug", open=False):
                        vt_mask = gr.Image(
                            label="Generated Mask",
                            width=256,
                            height=256,
                        )
                        vt_densepose = gr.Image(
                            label="Generated DensePose",
                            width=256,
                            height=256,
                        )

                    def save_current_image(image):
                        if image is not None:
                            save_generated_image(image, "virtual_tryon")
                    
                    vt_gen_button.click(
                        fn=vt_generate,
                        inputs=[
                            vt_src_image, vt_ref_image, vt_ref_acceleration,
                            vt_step, vt_scale, vt_seed, vt_model_type,
                            vt_garment_type, vt_repaint, preprocess_garment_checkbox,
                            autosave_checkbox
                        ],
                        outputs=[vt_gen_image, vt_mask, vt_densepose]
                    )
                    
                    save_button.click(
                        fn=save_current_image,
                        inputs=[vt_gen_image],
                        outputs=[]
                    )

        with gr.Tab("Pose Transfer  (64GB RAM | 16GB VRAM)"):
            with gr.Row():
                with gr.Column():
                    pt_ref_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Person Image",
                        width=512,
                        height=512,
                    )
                    with gr.Row():
                        pt_open_folder = gr.Button("📂 Open Output Folder")   
                    with gr.Row():    
                        with gr.Accordion("Person Examples", open=False) as pt_person_accordion:
                            gr.Examples(
                                inputs=pt_ref_image,
                                examples_per_page=20,
                                examples=person1_images,
                            )
                            pt_person1_add_btn = gr.Button("➕ Add More", size="sm")
                            pt_person1_add_btn.click(fn=lambda: open_examples_folder("person1"), inputs=[], outputs=[])
                with gr.Column():
                    pt_src_image = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Target Pose Person Image",
                        width=512,
                        height=512,
                    )
                    pt_status = gr.Textbox(
                    value="⚠️ A 20GB model will be downloaded on first use\n⚠️ 64GB RAM | 16GB VRAM required!\n⬅️ Check terminal for progress\n⬇️ Hugging Face Demo link below to assess quality\n\n💣 This textbox will self-destruct on next start after download",
                        label="Optional Pose model missing",
                        lines=6,
                        interactive=False,
                        visible=not os.path.exists("./ckpts/pose_transfer.pth")
                    )
                    with gr.Accordion("Pose Examples", open=False) as pose_accordion:
                        gr.Examples(
                            inputs=pt_src_image,
                            examples_per_page=20,
                            examples=person2_images,
                        )
                        pose_add_btn = gr.Button("➕ Add More", size="sm")
                        pose_add_btn.click(fn=lambda: open_examples_folder("person2"), inputs=[], outputs=[])
                with gr.Column():
                    pt_gen_image = gr.Image(
                        label="Generated Image",
                        width=512,
                        height=512,
                    )
                    with gr.Row():
                        pose_transfer_gen_button = gr.Button("Generate", variant="primary")
                        with gr.Group():
                            pt_autosave_checkbox = gr.Checkbox(label="Autosave", value=False)
                            pt_save_button = gr.Button("Save Current Image", variant="huggingface")

                    with gr.Accordion("Advanced Options", open=False):
                        pt_ref_acceleration = gr.Radio(
                            label="Accelerate Reference UNet (may slightly reduce performance)",
                            choices=[("True", True), ("False", False)],
                            value=False,
                        )
                        pt_step = gr.Number(
                            label="Inference Steps", minimum=30, maximum=100, step=1, value=30)
                        pt_scale = gr.Number(
                            label="Guidance Scale", minimum=0.1, maximum=5.0, step=0.1, value=2.5)
                        pt_seed = gr.Number(
                            label="Random Seed (-1 for random)", minimum=-1, maximum=2147483647, step=1, value=-1)

                    with gr.Accordion("Debug", open=False):
                        pt_mask = gr.Image(
                            label="Generated Mask",
                            width=256,
                            height=256,
                        )
                        pt_densepose = gr.Image(
                            label="Generated DensePose",
                            width=256,
                            height=256,
                        )

                    def save_pt_image(image):
                        if image is not None:
                            save_generated_image(image, "pose_transfer")

                    pose_transfer_gen_button.click(
                        fn=pt_generate,
                        inputs=[pt_src_image, pt_ref_image, pt_ref_acceleration, pt_step, pt_scale, pt_seed, pt_autosave_checkbox],
                        outputs=[pt_gen_image, pt_mask, pt_densepose]
                    )

                    pt_save_button.click(
                        fn=save_pt_image,
                        inputs=[pt_gen_image],
                        outputs=[]
                    )

        # Move folder button handlers to the end of the interface
        for btn in [vt_open_folder, pt_open_folder]:
            btn.click(
                fn=open_output_folder,
                inputs=[],
                outputs=None
            )
     
        gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')
        gr.Markdown(title)
        gr.Markdown(description)
        gr.Markdown(note)
        gr.Markdown(news)
        gr.Markdown(link)

        demo.launch(share=False, server_port=7860, allowed_paths=["./ckpts/examples"])
