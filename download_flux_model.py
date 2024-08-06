# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir="/lustre/shared/charchit/FLUX_dev")


import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("/lustre/shared/charchit/FLUX_sch", torch_dtype=torch.bfloat16).to("cuda")
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=2,
    max_sequence_length=256,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
image.save("/home/charchit/create_workflow/flux-schnell.png")