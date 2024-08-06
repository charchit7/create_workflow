
# Installation to run the code

Install pytorch - `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

Follow the instructions given here to install diffusers :
https://huggingface.co/docs/diffusers/en/installation

# Installation for Upscalar

The codebase for Real-ESRGAN has old dependeicies of python which I fixed in this code-base: 

```
pip install basicsr
pip install facexlib
pip install gfpgan
python setup.py develop
```

Download the weights 
`wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights`

Put the weight in the folder : `/create_workflow/super_resolution/Real-ESRGAN/weights`

Then you can follow the code below to run the Upscalar - basically Real-ESRGAN




# Instructions to run the code

Folder_structure : 

- Simple_promts.json : contains json file with key as prompt number and value as prompts. (total 10 prompts)
- Full_body_prompts.json : contains json file with key as prompt number and value as prompts. (total 10 prompts)
- Complex_prompts.json : contains json file with key as prompt number and value as prompts. (total 10 prompts)
- Generate_images.py : this python file contains code to run Stable Diffusion SD1.5, SDXL, and civit.ai Juggernaut, and SDXL - LCM-LORA.

```
Format to run code : 
python file_name.py path_to_prompt.json_file folder_to_save_results model_variant_to_run

For model_variant_to_run replace with 
-  sd1.5 for Stable Diffusion
- sdxl for stable diffusion XL
- juggernaut for civit-ai juggernaut model
- lcmlora : for stable diffusion XL - LCM-LORA

Example CLI command: 
python generate_images.py simple_prompts.json /home/charchit/create_workflow/temp_test/ sdxl

```

- generate_images_with_sdxl_refiner.py contains code to run refiner version of the SDXL
```
Format to run code:
python file_name.py path_to_promt.json folder_to_save model_variant
available variants :
- v1 
- v2

Example CLI  command:
python generate_images_with_sdxl_refiner.py simple_prompts.json /home/charchittemp_test_2 v1
```

- clip_similarity_score.py : contains function

```
compare_text_image : this function is used to generate similarity score between generate image and it's promts.

Usage :
--
from clip_smilarity_score import similarity_score
image_path = '/path/to/your/image.jpg'
text = "A description of the image"
similarity_score = compare_text_image(image_path, text)

```

- compare_two_texts.py : this code is used to compare two texts. The idea is that we are evaluating prompt and genereated captions from the florence-2 model.

```
Usage:
--
from compare_two_texts import compare_texts

text1 = 'The image shows a woman with long brown hair wearing a red t-shirt, smiling at the camera against a red background.'
text2 = "A smiling young woman with long brown hair, wearing a red t-shirt, standing against a plain white wall."

similarity_score = compare_texts(text1, text2)
```

- florence_2_captions.py : this code is used to generate captions for an image. 

```
Usage : 
--
result = analyze_image('/path/to/your/image.png', '<DETAILED_CAPTION>')
print(result)
```


# Instruction to run the Upscalar
go to the folder :

`cd super_resolution/Real-ESRGAN`

Then use the below CLI command to run the upscalar

```
Usage:
python inference_realesrgan.py -n RealESRGAN_x4plus -i images_folder -s upscalar_value --face_enchance -o output_folder
```
If your generate image is from SD1.5 then use -s 4 if your image is generate from SDXL and it's variants (LCM, juggernaut, etc) ise -s 2

```
Example CLI command: 
python inference_realesrgan.py -n RealESRGAN_x4plus -i /home/charchit/create_workflow/standing_prompts_lcmLORA/ -s 2 --face_enhance -o standing_prompts_lcmLORA

```

Drive Link for generated Images : `https://drive.google.com/drive/folders/1V5kUqhor6w-O1WwMIVrJ3scyXXlAvEBe?usp=drive_link`
