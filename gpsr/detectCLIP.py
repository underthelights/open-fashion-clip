import itertools
import open_clip
import torch

def init_clip():
    # Assuming the necessary OpenFashionClip setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
    state_dict = torch.load('module/CLIP/openfashionclip.pt', map_location=device)
    clip_model.load_state_dict(state_dict['CLIP'])
    clip_model = clip_model.eval().requires_grad_(False).to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    return clip_model, preprocess, tokenizer, device

def detectTopClothe(img, clip_model, preprocess, tokenizer, device):
    clothe_list = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]
    prompt_prefix = "a photo of a human who is wearing a "
    clothe_prompts = [prompt_prefix + " " + clothe for clothe in clothe_list]
    
    tokenized_prompt = tokenizer(clothe_prompts).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(img.unsqueeze(0))
        text_features = clip_model.encode_text(tokenized_prompt)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    text_probs_percent = text_probs * 100
    text_probs_percent_np = text_probs_percent.cpu().numpy()
    
    top_index = text_probs_percent_np[0].argmax()
    top_clothe = clothe_list[top_index]
    top_clothe_prob = "{:.2f}%".format(text_probs_percent_np[0][top_index])
    
    return top_clothe, top_clothe_prob
