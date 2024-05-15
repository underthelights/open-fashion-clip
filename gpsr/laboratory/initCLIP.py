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
