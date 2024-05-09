import open_clip
from PIL import Image
import torch
import time

# start_time = time.time()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('module/CLIP/openfashionclip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')


img = Image.open('examples/shoes.jpg')
img = preprocess(img).to(device)

prompt = "a photo of a"
text_inputs = ["wearing shoes", "bare foot"]
text_inputs = [prompt + " " + t for t in text_inputs]
tokenized_prompt = tokenizer(text_inputs).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(img.unsqueeze(0)) #Input tensor should have shape (b,c,h,w)
    text_features = clip_model.encode_text(tokenized_prompt)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


# 확률을 퍼센트로 변환
text_probs_percent = text_probs * 100

# 텐서를 CPU로 이동시키고 NumPy 배열로 변환
text_probs_percent_np = text_probs_percent.cpu().numpy()

# 포맷팅을 사용하여 소수점 아래 두 자리까지 출력
formatted_probs = ["{:.2f}%".format(value) for value in text_probs_percent_np[0]]

print("Labels probabilities in percentage:", formatted_probs)


# print("Labels probs:", text_probs)
# 여기에 시간을 측정하고 싶은 코드 블록을 넣습니다.

# end_time = time.time()
# total_time = end_time - start_time
# print(f"Total execution time: {total_time:.2f} seconds")
