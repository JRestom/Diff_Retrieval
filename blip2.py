import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

raw_image = Image.open("/home/jose.viera/cv806/CoVR/datasets/CIRR/images/train/0/train-42-0-img0.png").convert("RGB")
caption = "a large fountain spewing water into the air"

#raw_image.resize((596, 437))

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
sample = {"image": image}

features_image = model.extract_features(sample, mode="image")
print(features_image.image_embeds.shape)
print(features_image.image_embeds_proj.shape)