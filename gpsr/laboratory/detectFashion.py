import itertools
import open_clip
from PIL import Image
import torch

# Assuming the necessary OpenFashionClip setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('module/CLIP/openfashionclip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Lists to be detected
color_list = ["blue", "yellow", "black", "white", "red", "orange", "gray"]
clothe_list = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]
# Generate all color-clothes combinations
combined_clothes = [f"{color} {clothe}" for color, clothe in itertools.product(color_list, clothe_list)]

def detect_fashion_combinations(image_path):
    img = Image.open(image_path)
    img = preprocess(img).to(device)

    prompt_prefix = "a photo of a"
    
    # Process color-clothes combinations
    full_text_inputs = [prompt_prefix + " " + item for item in combined_clothes]
    tokenized_prompt = tokenizer(full_text_inputs).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(img.unsqueeze(0))
        text_features = clip_model.encode_text(tokenized_prompt)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    text_probs_percent = text_probs * 100
    text_probs_percent_np = text_probs_percent.cpu().numpy()
    
    # Sort the results and pick the top 4
    top_indices = text_probs_percent_np[0].argsort()[-4:][::-1]
    results = {}
    for idx in top_indices:
        item = combined_clothes[idx]
        results[item] = "{:.2f}%".format(text_probs_percent_np[0][idx])
    
    return results

# Example usage
image_path = 'examples/shoes.jpg'
detections = detect_fashion_combinations(image_path)

for item, probability in detections.items():
    print(f"{item:<20}: {probability}")