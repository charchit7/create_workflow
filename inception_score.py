import torch
from torch import nn
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import os

class InceptionScore:
    def __init__(self, batch_size=32, resize=True, splits=10):
        self.batch_size = batch_size
        self.resize = resize
        self.splits = splits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained Inception-v3 model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        
        # Remove the last fully connected layer
        self.inception_model.fc = nn.Identity()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)) if resize else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_images(self, image_folder):
        images = []
        for filename in os.listdir(image_folder):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_folder, filename)
                img = Image.open(img_path).convert('RGB')
                img = self.preprocess(img)
                images.append(img)
        print('images folder size',len(images))
        return torch.stack(images)

    @torch.no_grad()
    def get_predictions(self, images):
        preds = []
        n_batches = int(np.ceil(len(images) / self.batch_size))
        for i in tqdm(range(n_batches)):
            batch = images[i * self.batch_size: (i + 1) * self.batch_size].to(self.device)
            pred = self.inception_model(batch)
            preds.append(pred.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def calculate_inception_score(self, preds, eps=1e-16):
        # Calculate the mean KL-divergence
        kl_divergence = []
        preds = preds.clip(min=eps)  # To avoid log(0)
        preds = preds / preds.sum(axis=1, keepdims=True)  # Normalize

        for i in range(self.splits):
            part = preds[i * (len(preds) // self.splits): (i + 1) * (len(preds) // self.splits), :]
            py = part.mean(axis=0)
            scores = []
            for j in range(part.shape[0]):
                pyx = part[j, :]
                scores.append(entropy(pyx, py))
            kl_divergence.append(np.mean(scores))

        # Calculate the Inception Score and its standard deviation
        is_score = np.exp(np.mean(kl_divergence))
        is_std = np.exp(np.std(kl_divergence))

        return is_score, is_std

    def compute(self, image_folder):
        print("Loading and preprocessing images...")
        images = self.load_images(image_folder)
        
        print("Calculating Inception predictions...")
        preds = self.get_predictions(images)
        
        print("Computing Inception Score...")
        inception_score, inception_score_std = self.calculate_inception_score(preds)
        
        return inception_score, inception_score_std

# Example usage
if __name__ == "__main__":
    image_folder = "/lustre/shared/charchit/generated_images/simple_prompts_sdxl/"
    inception_score_calculator = InceptionScore()
    score, std = inception_score_calculator.compute(image_folder)
    print(f"Inception Score: {score:.3f} ± {std:.3f}")