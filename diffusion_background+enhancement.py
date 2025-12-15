"""
Lightweight Background Image Generator with Auto-Enhancement
Optimized for GPUs with 4-6GB VRAM
Automated: Generate -> Enhance -> Save

Requirements:
pip install diffusers transformers accelerate torch torchvision safetensors pillow opencv-python realesrgan basicsr
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import gc
import cv2
import numpy as np
import traceback

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class LightweightBackgroundGenerator:
    def __init__(self, model_id="segmind/small-sd", device=None):
        """
        Initialize lightweight background generator.
        
        Models optimized for low VRAM:
        - "segmind/small-sd" (2GB VRAM, fast) - RECOMMENDED
        - "segmind/tiny-sd" (1.5GB VRAM, fastest)
        - "OFA-Sys/small-stable-diffusion-v0" (2GB VRAM)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading lightweight model on {self.device}...")
        print(f"Model: {model_id}")
        
        # Load the pipeline with memory optimizations
        # Try safetensors first (safer and faster), fall back to .bin if not available
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True  # Prefer safetensors format
            )
            print("✓ Loaded model using safetensors format")
        except Exception as e:
            if "safetensors" in str(e).lower():
                print("⚠️ Safetensors not available, falling back to PyTorch format")
                print("   Note: With PyTorch 2.9.1+, this is safer than older versions")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=False  # Fall back to .bin files
                )
            else:
                raise
        
        self.pipe = self.pipe.to(self.device)
        
        # Critical memory optimizations for low VRAM
        if self.device == "cuda":
            self.pipe.enable_attention_slicing(1)  # Maximum slicing
            self.pipe.enable_vae_slicing()  # Slice VAE
            
            # Try to enable memory efficient attention
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xformers enabled for extra memory savings")
            except:
                print("✓ Using standard attention (install xformers for more savings)")
        
        print("Model loaded successfully!")
        print(f"Estimated VRAM usage: ~2-3GB")
    
    def generate_background(self, 
                          prompt, 
                          negative_prompt="blurry, low quality, pixelated, distorted",
                          width=512,
                          height=512,
                          num_inference_steps=25,
                          guidance_scale=7.5,
                          seed=None):
        """
        Generate a background image.
        
        IMPORTANT: For 6GB GPU, keep dimensions at 512x512 or lower!
        """
        # Validate dimensions for low VRAM
        max_pixels = width * height
        if max_pixels > 512 * 512 and self.device == "cuda":
            print(f"⚠️  Warning: {width}x{height} may cause OOM on 6GB GPU")
            print(f"   Recommended: 512x512 or smaller")
        
        # Clear GPU cache before generation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Set random seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"\nGenerating: '{prompt}'")
        print(f"Size: {width}x{height}, Steps: {num_inference_steps}")
        
        try:
            # Generate with autocast for memory efficiency
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            image = result.images[0]
            print("✓ Generation complete!")
            
            # Clear cache after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return image
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n❌ GPU Out of Memory!")
                print("Solutions:")
                print("1. Reduce image size (try 256x256 or 384x384)")
                print("2. Reduce inference steps (try 15-20)")
                print("3. Use tiny-sd model (even smaller)")
                raise
            else:
                raise
    
    def save_image(self, image, filename="background.png", output_dir="outputs"):
        """Save generated image"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"✓ Saved to: {filepath}")
        return filepath


class ImageEnhancer:
    def __init__(self, model_path="weights/RealESRGAN_x4plus.pth", device=None):
        """Initialize Real-ESRGAN enhancer"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.upsampler = None
        self.use_realesrgan = False
        
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ RealESRGAN model not found at: {model_path}")
                print("   Falling back to bicubic upscaling")
                return
            
            print(f"\nLoading RealESRGAN from: {model_path}")
            
            # RRDBNet config for RealESRGAN_x4plus
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            
            # Use half precision only on CUDA
            half = self.device.type == "cuda"
            
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,          # no tiling; set >0 if VRAM issues
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
            
            self.use_realesrgan = True
            print(f"✓ RealESRGAN ready on {self.device} (half={half})")
            
        except Exception as e:
            print(f"⚠️ Could not initialize RealESRGAN, using bicubic fallback")
            print(f"   Error: {e}")
    
    def enhance(self, image):
        """
        Enhance image using Real-ESRGAN or fallback to bicubic
        Args:
            image: PIL Image or numpy array
        Returns:
            PIL Image (enhanced)
        """
        print("\n" + "="*60)
        print("Enhancing image...")
        print("="*60)
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        
        try:
            if self.use_realesrgan and self.upsampler is not None:
                print("Using Real-ESRGAN 4x enhancement...")
                output, _ = self.upsampler.enhance(img_array, outscale=4)
                print(f"✓ Enhanced to {output.shape[1]}x{output.shape[0]}")
            else:
                print("Using bicubic 4x upscaling (fallback)...")
                h, w = img_array.shape[:2]
                output = cv2.resize(
                    img_array,
                    (w * 4, h * 4),
                    interpolation=cv2.INTER_CUBIC
                )
                print(f"✓ Upscaled to {output.shape[1]}x{output.shape[0]}")
            
            # Convert back to PIL RGB
            enhanced_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            print("✓ Enhancement complete!")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return enhanced_image
            
        except Exception as e:
            print(f"⚠️ Enhancement failed: {e}")
            print("Returning original image")
            traceback.print_exc()
            return image if isinstance(image, Image.Image) else Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))


def main():
    """Automated workflow: Generate -> Enhance -> Save"""
    
    print("\n" + "="*60)
    print("AI Background Generator with Auto-Enhancement")
    print("="*60)
    
    # Get prompt from user
    print("\nEnter your image description:")
    prompt = input("Prompt: ").strip()
    
    if not prompt:
        print("No prompt provided. Using default.")
        prompt = "beautiful mountain landscape, sunset, detailed"
    
    print(f"\n✓ Prompt: {prompt}")
    
    # Automated settings
    model_id = "segmind/small-sd"  # Small-SD (Recommended)
    width, height = 768, 512  # 768x512 size
    steps = 20  # 20 inference steps
    filename = "background.png"  # Final filename
    
    print(f"✓ Model: Small-SD (Recommended)")
    print(f"✓ Size: {width}x{height}")
    print(f"✓ Steps: {steps}")
    
    # Initialize generator
    print("\n" + "="*60)
    print("STEP 1: Initializing Generator")
    print("="*60)
    generator = LightweightBackgroundGenerator(model_id=model_id)
    
    # Generate image
    print("\n" + "="*60)
    print("STEP 2: Generating Image")
    print("="*60)
    try:
        generated_image = generator.generate_background(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps
        )
        
        # Save original (optional, for comparison)
        os.makedirs("outputs", exist_ok=True)
        original_path = "outputs/original_" + filename
        generated_image.save(original_path)
        print(f"✓ Original saved: {original_path}")
        
    except Exception as e:
        print(f"\n❌ Generation Error: {e}")
        return
    
    # Initialize enhancer
    print("\n" + "="*60)
    print("STEP 3: Initializing Enhancer")
    print("="*60)
    enhancer = ImageEnhancer()
    
    # Enhance image
    print("\n" + "="*60)
    print("STEP 4: Enhancing Image")
    print("="*60)
    try:
        enhanced_image = enhancer.enhance(generated_image)
        
    except Exception as e:
        print(f"\n❌ Enhancement Error: {e}")
        print("Using original image instead")
        enhanced_image = generated_image
    
    # Save final enhanced image
    print("\n" + "="*60)
    print("STEP 5: Saving Final Image")
    print("="*60)
    final_path = generator.save_image(enhanced_image, filename)
    
    # Show final image
    print("\nOpening enhanced image...")
    enhanced_image.show()
    
    # Memory info
    if torch.cuda.is_available():
        print(f"\nGPU Memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print(f"Original: {original_path}")
    print(f"Enhanced: {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()
