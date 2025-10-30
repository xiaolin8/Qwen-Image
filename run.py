import argparse
import torch
import os
from diffusers import DiffusionPipeline
from datetime import datetime

def main():
    """
    A flexible runner script for Qwen-Image generation that accepts all
    necessary parameters from the command line.
    """
    parser = argparse.ArgumentParser(description="A flexible runner for Qwen-Image generation.")

    # --- Core Arguments ---
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the local model directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")

    # --- Pipeline & Generation Arguments ---
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Text prompt to guide the model away from generating certain things.")
    parser.add_argument("--height", type=int, default=1328, help="The height in pixels of the generated image.")
    parser.add_argument("--width", type=int, default=1328, help="The width in pixels of the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="The number of denoising steps.")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible generation.")

    args = parser.parse_args()

    # --- 1. Initialize Pipeline ---
    print(f"Loading pipeline from local directory: {args.model_dir}")
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    try:
        pipe = DiffusionPipeline.from_pretrained(args.model_dir, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"Error loading the pipeline: {e}")
        print("Please ensure the --model_dir path is correct and contains the necessary model files.")
        return

    print(f"Pipeline loaded successfully on device: {device}")

    # --- 2. Prepare Generator for Reproducibility ---
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # --- 3. Generate Image ---
    print(f"Generating image for prompt: \"{args.prompt}\" ")
    try:
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator
        ).images[0]
    except Exception as e:
        print(f"Error during image generation: {e}")
        return

    # --- 4. Save Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a somewhat unique and descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize prompt for use in filename (take first 30 chars)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in args.prompt[:30])
    filename = f"{timestamp}_{safe_prompt}.png"
    output_path = os.path.join(args.output_dir, filename)

    image.save(output_path)
    print(f"Image successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
