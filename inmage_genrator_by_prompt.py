"""
Lightweight Background Image Generator
Optimized for GPUs with 4-6GB VRAM
Automated version - only prompts user for description

Requirements:
pip install diffusers transformers accelerate torch torchvision safetensors pillow
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import gc

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
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
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


def main():
    """Automated usage - only asks for prompt"""
    
    print("\n" + "="*60)
    print("Lightweight Background Generator (Automated)")
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
    filename = "background.png"  # Default filename
    
    print(f"✓ Model: Small-SD (Recommended)")
    print(f"✓ Size: {width}x{height}")
    print(f"✓ Steps: {steps}")
    print(f"✓ Filename: {filename}")
    
    # Initialize generator
    print("\nInitializing generator...")
    generator = LightweightBackgroundGenerator(model_id=model_id)
    
    # Generate
    try:
        image = generator.generate_background(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps
        )
        
        # Save
        generator.save_image(image, filename)
        
        # Show
        print("\nOpening image...")
        image.show()
        
        # Memory info
        if torch.cuda.is_available():
            print(f"\nGPU Memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        print("\n✓ All done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you got an OOM error, the script will need to use smaller dimensions.")


if __name__ == "__main__":
    main()
