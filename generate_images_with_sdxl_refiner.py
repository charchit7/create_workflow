import argparse
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import json
import os
import sys

def check_directory(path):
    if os.path.exists(path):
        # Check if the directory contains any PNG files
        if any(file.endswith('.png') for file in os.listdir(path)):
            print(f"Error: The directory '{path}' already exists and contains PNG files.")
            print("To prevent overwriting existing images, the script will now exit.")
            print("Please choose a different directory or remove existing images if you want to proceed.")
            sys.exit(1)


def generate_images(prompts_path, save_path):

    # Check if the save directory already exists and contains PNG files
    check_directory(save_path)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    generator = torch.Generator("cuda").manual_seed(42)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")


    # Load the prompts from json file
    with open(prompts_path, 'r') as file:
        prompts_data = json.load(file)

    prompts = prompts_data['prompts']

    # Generate and save images
    for val in prompts:
        image = base(
        prompt=val['text'],
        num_inference_steps=40,
        denoising_end=0.8,
        generator=generator,
        output_type="latent",
        ).images
        image = refiner(
        prompt=val['text'],
        num_inference_steps=40,
        generator=generator,
        denoising_start=0.8,
        image=image,
        ).images[0]
        image_path = os.path.join(save_path, f"image_{val['id']}.png")
        image.save(image_path)
        print(f"Generated and saved image for prompt {val['id']} at {image_path}")

def generate_image_v2(prompts_path, save_path):

    # Check if the save directory already exists and contains PNG files
    check_directory(save_path)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    generator = torch.Generator("cuda").manual_seed(42)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")


    # Load the prompts from json file
    with open(prompts_path, 'r') as file:
        prompts_data = json.load(file)

    prompts = prompts_data['prompts']

    # Generate and save images
    for val in prompts:
        
        #image to image_setting
        image = base(prompt=val['text'], output_type="latent").images[0]
        image = refiner(prompt=val['text'], image=image[None, :]).images[0]

        image_path = os.path.join(save_path, f"image_{val['id']}.png")
        image.save(image_path)
        print(f"Generated and saved image for prompt {val['id']} at {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompts using a specified model.")
    parser.add_argument("prompts_path", type=str, help="Path to the JSON file containing prompts")
    parser.add_argument("save_path", type=str, help="Path to save the generated images")
    parser.add_argument("version", type=str, help="which refiner variant to run")
    args = parser.parse_args()

    if args.version == 'v1':
        generate_images(args.prompts_path, args.save_path)
    elif args.version == 'v2':
        generate_image_v2(args.prompts_path, args.save_path)
    else:
        print('Invalid version')