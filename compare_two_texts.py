from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(texts, model, tokenizer):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

def compare_texts(text1, text2, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Get embeddings
    embeddings = get_embedding([text1, text2], model, tokenizer)

    # Compute cosine similarity
    similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))

    return similarity.item()

# Example usage
if __name__ == "__main__":
    text1 = 'The image shows a woman with long brown hair wearing a red t-shirt, smiling at the camera against a red background.'
    text2 = "A smiling young woman with long brown hair, wearing a red t-shirt, standing against a plain white wall."
    
    similarity_score = compare_texts(text1, text2)
    print(f"Similarity between the texts: {similarity_score:.4f}")