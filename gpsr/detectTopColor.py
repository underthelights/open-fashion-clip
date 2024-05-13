import torch

def detectTopColor(img, clip_model, preprocess, tokenizer, device):
    color_list = ["blue", "yellow", "black", "white", "red", "orange", "gray"]
    prompt_prefix = "A photo of person who is wearing a top colored "
    color_prompts = [prompt_prefix + " " + color for color in color_list]
    
    tokenized_prompt = tokenizer(color_prompts).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(img.unsqueeze(0))
        text_features = clip_model.encode_text(tokenized_prompt)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    text_probs_percent = text_probs * 100
    text_probs_percent_np = text_probs_percent.cpu().numpy()
    
    top_index = text_probs_percent_np[0].argmax()
    top_color = color_list[top_index]
    top_color_prob = "{:.2f}%".format(text_probs_percent_np[0][top_index])
    
    return top_color, top_color_prob
