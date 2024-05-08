import open_clip
from PIL import Image
import torch
import time

start_time = time.time()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('weights/openfashionclip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')


img = Image.open('examples/shoes.jpg')
img = preprocess(img).to(device)

# Different sets of text inputs to iterate over
text_input_categories = [
    ["man", "woman"],
    ["20s", "30s", "40s"],
    ["wearing shoes", "bare foot", "wearing socks"],
    ["holding drinks", "no drinks"]
]

# Prompt prefix
prompt = "a photo of a"

# Process each category of text inputs
for text_inputs in text_input_categories:
    # Measure the time for this category
    category_start_time = time.time()
    
    # Prepare text inputs with the prompt
    full_text_inputs = [prompt + " " + t for t in text_inputs]
    tokenized_prompt = tokenizer(full_text_inputs).to(device)
    
    # Perform the model inference
    with torch.no_grad():
        image_features = clip_model.encode_image(img.unsqueeze(0))
        text_features = clip_model.encode_text(tokenized_prompt)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    # Convert probabilities to percentages
    text_probs_percent = text_probs * 100
    text_probs_percent_np = text_probs_percent.cpu().numpy()
    
    # Format the output
    formatted_probs = ["{:.2f}%".format(value) for value in text_probs_percent_np[0]]
    
    # Print the results for this category
    print(f"Text Inputs: {text_inputs}")
    print("Labels probabilities in percentage:", formatted_probs)
    
    # Category execution time
    category_end_time = time.time()
    print(f"Execution time for this category: {category_end_time - category_start_time:.2f} seconds\n")

# Total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
