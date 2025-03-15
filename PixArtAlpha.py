import torch
import os
import math
import folder_paths
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download
import numpy as np

# Try to import SASolverScheduler, but handle the case if it's missing
try:
    from .sa_solver_diffusers import SASolverScheduler
except ModuleNotFoundError:
    # If SASolverScheduler is not available, we'll handle this in the code
    print("SASolverScheduler not found, only DPM-Solver will be available")
    SASolverScheduler = None

device = "cuda" if torch.cuda.is_available() else "cpu"


class PA_BaseModelLoader_fromhub:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "PixArt-alpha/PixArt-XL-2-1024-MS"})
            }
        }

    RETURN_TYPES = ("PAMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "PixArtAlpha"
  
    def load_model(self, base_model_path):
        print(f"Loading PixArtAlpha model from: {base_model_path}")
        
        # Load with explicit full precision to avoid type mismatches
        model = PixArtAlphaPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,  # Force everything to be float32
            low_cpu_mem_usage=True,
        ).to(device)
        
        print("Model loaded successfully with float32 precision")
        return [model]


class PA_Generation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        schedulers = ["DPM-Solver"]
        if SASolverScheduler is not None:
            schedulers.append("SA-Solver")
            
        return {
            "required": {
                "model": ("PAMODEL",),
                "positive": ("STRING", {"multiline": True, "forceInput": True}),
                "negative": ("STRING", {"multiline": True, "forceInput": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 3072, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 3072, "step": 32}), 
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0, "max": 20}),
                "schedule": (schedulers,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "PixArtAlpha"
                       
    def generate_image(self, model, positive, negative, steps, guidance_scale, seed, width, height, schedule):
        # Configure scheduler
        if schedule == 'DPM-Solver':
            if not isinstance(model.scheduler, DPMSolverMultistepScheduler):
                model.scheduler = DPMSolverMultistepScheduler()
        elif schedule == "SA-Solver" and SASolverScheduler is not None:
            if not isinstance(model.scheduler, SASolverScheduler):
                model.scheduler = SASolverScheduler.from_config(model.scheduler.config, algorithm_type='data_prediction', tau_func=lambda t: 1 if 200 <= t <= 800 else 0, predictor_order=2, corrector_order=2)
        else:
            if schedule != "DPM-Solver":
                print(f"Warning: SA-Solver not available, using DPM-Solver instead")
                model.scheduler = DPMSolverMultistepScheduler()
        
        # Make sure dimensions are valid
        width = max(64, width - (width % 8))
        height = max(64, height - (height % 8))
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Enable CPU offloading for memory efficiency
        try:
            # Try to use the CPU offloading features
            if hasattr(model, "enable_model_cpu_offload"):
                print("Enabling model CPU offloading")
                model.enable_model_cpu_offload()
            elif hasattr(model, "enable_sequential_cpu_offload"):
                print("Enabling sequential CPU offload")
                model.enable_sequential_cpu_offload()
            else:
                # Manual CPU offloading
                print("Applying manual CPU memory management")
                # Move non-essential components to CPU
                if hasattr(model, "safety_checker") and model.safety_checker is not None:
                    model.safety_checker = model.safety_checker.to("cpu")
                if hasattr(model, "feature_extractor") and model.feature_extractor is not None:
                    model.feature_extractor = model.feature_extractor.to("cpu")
        except Exception as e:
            print(f"CPU offload setup error: {e}")
        
        # Enable other memory-saving features
        try:
            model.enable_attention_slicing(slice_size="auto")
            if hasattr(model, "enable_vae_slicing"):
                model.enable_vae_slicing()
            if hasattr(model, "enable_vae_tiling"):
                model.enable_vae_tiling()
        except Exception as e:
            print(f"Memory optimization error: {e}")
        
        # Create generator
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Define custom callback for step-by-step memory management
        def memory_efficient_callback(step, timestep, latents):
            # Move intermediate outputs to CPU at regular intervals
            if step % 5 == 0 and step > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            return latents
        
        # Try direct generation with memory optimizations
        try:
            print(f"Generating at full resolution ({width}x{height}) with RAM assistance")
            output = model(
                prompt=positive,
                negative_prompt=negative,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                callback=memory_efficient_callback,
                callback_steps=1
            )
            
            if isinstance(output, tuple):
                images_list = output[0]
            else:
                images_list = output.images
                
        except (torch.cuda.OutOfMemoryError, RuntimeError, AttributeError) as e:
            print(f"Error even with CPU offloading: {e}")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # As a last resort, try stage-by-stage processing
            print("Using stage-by-stage processing to maximize RAM usage")
            
            try:
                # Move components back to GPU for direct access
                if hasattr(model, "unet"):
                    model.unet = model.unet.to(device)
                if hasattr(model, "text_encoder"):
                    model.text_encoder = model.text_encoder.to(device)
                if hasattr(model, "vae"):
                    model.vae = model.vae.to(device)
                
                # Generate half resolution but with full parameters
                scaled_w = width // 2
                scaled_h = height // 2
                
                # Ensure multiples of 8
                scaled_w = scaled_w - (scaled_w % 8)
                scaled_h = scaled_h - (scaled_h % 8)
                
                print(f"Generating at {scaled_w}x{scaled_h} with highest quality parameters")
                
                # Increase steps for better quality
                higher_steps = min(steps + 10, 50)
                
                output = model(
                    prompt=positive,
                    negative_prompt=negative,
                    num_inference_steps=higher_steps,  # More steps for higher quality
                    generator=generator,
                    guidance_scale=guidance_scale * 1.1,  # Slightly higher guidance for better detail
                    width=scaled_w,
                    height=scaled_h,
                    callback=memory_efficient_callback,
                    callback_steps=1
                )
                
                if isinstance(output, tuple):
                    images_list = output[0]
                else:
                    images_list = output.images
                    
                # High-quality upscaling
                from PIL import Image
                upscaled_images = []
                
                print(f"Performing high-quality upscaling from {scaled_w}x{scaled_h} to {width}x{height}")
                
                for img in images_list:
                    # First resize to slightly larger than target
                    oversized = img.resize((width + 100, height + 100), Image.LANCZOS)
                    
                    # Then crop to target size (helps preserve details)
                    left = (oversized.width - width) // 2
                    top = (oversized.height - height) // 2
                    upscaled_img = oversized.crop((left, top, left + width, top + height))
                    
                    upscaled_images.append(upscaled_img)
                
                images_list = upscaled_images
                
            except Exception as final_e:
                print(f"Final fallback failed: {final_e}")
                raise RuntimeError(f"Failed to generate image: {final_e}")
        
        # Convert to tensor
        images_tensors = []
        for img in images_list:
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).float() / 255.
            
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
                
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)
        
        # Concatenate if multiple images
        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]
        
        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "PA_BaseModelLoader_fromhub": PA_BaseModelLoader_fromhub,
    "PA_Generation": PA_Generation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PA_BaseModelLoader_fromhub": "PixArtAlpha ModelLoader",
    "PA_Generation": "PixArtAlpha Generation"
}