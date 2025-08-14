# predict.py
from cog import BasePredictor, Input, Path
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

# Define the model cache location
MODEL_CACHE = "sdxl-cache"
# The specific base model version we want to use
BASE_VERSION = "stabilityai/stable-diffusion-xl-base-1.0"
# Your Hugging Face LoRA model
LORA_MODEL = "brucetan91/my-sdxl-lora"


class Predictor(BasePredictor):
    def setup(self):
        """Loads the model into memory to make running multiple predictions efficient"""
        # Load the VAE separately for stability
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        )
        
        # Load the base SDXL pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            BASE_VERSION,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=MODEL_CACHE,
        )
        
        # *** THIS IS THE KEY STEP ***
        # Load your LoRA weights from Hugging Face into the pipeline
        self.pipe.load_lora_weights(LORA_MODEL)
        
        self.pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", default=7.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        output_path = "/tmp/output.png"
        image.save(output_path)

        return Path(output_path)
