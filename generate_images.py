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


def generate_images(model_name, prompts_path, save_path):

    # Check if the save directory already exists and contains PNG files
    check_directory(save_path)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_name, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    generator = torch.Generator("cuda").manual_seed(42)

    # Load the prompts from json file
    with open(prompts_path, 'r') as file:
        prompts_data = json.load(file)

    prompts = prompts_data['prompts']

    # Generate and save images
    for val in prompts:
        image = pipeline(
            val['text'],
            generator=generator,
        ).images[0]
        image_path = os.path.join(save_path, f"image_{val['id']}.png")
        image.save(image_path)
        print(f"Generated and saved image for prompt {val['id']} at {image_path}")

def generate_image_juggernaut(prompts_path, save_path):
    
    # Check if the save directory already exists and contains PNG files
    check_directory(save_path)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    generator = torch.Generator("cuda").manual_seed(42)

    # Load the prompts from json file
    with open(prompts_path, 'r') as file:
        prompts_data = json.load(file)

    prompts = prompts_data['prompts']

    # Generate and save images
    for val in prompts:
        image = pipeline(
            val['text'],
            generator=generator,
        ).images[0]
        image_path = os.path.join(save_path, f"image_{val['id']}.png")
        image.save(image_path)
        print(f"Generated and saved image for prompt {val['id']} at {image_path}")


def generate_image_lcmlora(prompts_path, save_path):

    # Check if the save directory already exists and contains PNG files
    check_directory(save_path)

    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    # "RunDiffusion/Juggernaut-XL-v9",
    variant="fp16",
    torch_dtype=torch.float16).to("cuda")

    # set scheduler
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    # load LCM-LoRA
    pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")

    generator = torch.Generator("cuda").manual_seed(42)

    # Load the prompts from json file
    with open(prompts_path, 'r') as file:
        prompts_data = json.load(file)

    prompts = prompts_data['prompts']

    # Generate and save images
    for val in prompts:
        image = pipeline(
        prompt=val['text'], num_inference_steps=4, generator=generator, guidance_scale=1.0
        ).images[0]
        image_path = os.path.join(save_path, f"image_{val['id']}.png")
        image.save(image_path)
        print(f"Generated and saved image for prompt {val['id']} at {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompts using a specified model.")
    # parser.add_argument("model_name", type=str, help="Name or path of the model to use")
    parser.add_argument("prompts_path", type=str, help="Path to the JSON file containing prompts")
    parser.add_argument("save_path", type=str, help="Path to save the generated images")
    parser.add_argument("version", type=str, help="Which model you want to test : sd1.5, sdxl, juggernaut")    
    args = parser.parse_args()

    if args.version == "sd1.5":
        model_name = "runwayml/stable-diffusion-v1-5"
        generate_images(model_name, args.prompts_path, args.save_path)
    elif args.version == "sdxl":
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        generate_images(model_name, args.prompts_path, args.save_path)
    elif args.version == "juggernaut":
        generate_image_juggernaut(args.prompts_path, args.save_path)
    elif args.version == "lcmlora":
        generate_image_lcmlora(args.prompts_path, args.save_path)
        