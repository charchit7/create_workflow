import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import clip

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def compare_text_image(image_path, text):
    """
    Calculate the similarity score between an image and a text description using CLIP.
    
    Args:
    image_path (str): Path to the image file.
    text (str): Text description to compare with the image.
    
    Returns:
    float: Similarity score between the image and text.
    """
    # Open the image using PIL
    image = Image.open(image_path)

    # Prepare the inputs for the model
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    # Calculate the cosine similarity between the image and text embeddings
    similarity = torch.cosine_similarity(image_embeds, text_embeds)
    return similarity.item()


# Example usage
if __name__ == "__main__":
    image_path = '/path/to/your/image.jpg'
    text = "A description of the image"
    similarity_score = compare_text_image(image_path, text)
    print(f"Similarity between image and text: {similarity_score}")