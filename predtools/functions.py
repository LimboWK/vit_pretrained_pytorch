from PIL import Image
import matplotlib.pyplot as plt
import torch 
import torch.functional as F
import torch.nn as nn

def extract_feature_vector(img_path, model=None, tsfm=None, show=False, device="cpu", **kwargs):
    # feature_vector = torch.tensor([0]*768, dtype=torch.float).to(device) # 768 dim
    img = Image.open(img_path)
    if show:
        fig = plt.figure(figsize=(6,6))
        fig.add_subplot(1,1,1)
        plt.imshow(img)
    # Preprocess image
    img = tsfm(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img.to(device), logits=True).squeeze(0) # model must be vit body modified version!
    return F.normalize(outputs, dim=0)


def make_fewshots_pred_ft_all(img_path, ground_truth:dict, model_ft=None, **kwargs):
    device = kwargs["device"]
    # test_vector = extract_feature_vector(img_path, show=True)
    # print(pmatrix.shape)
    # prediction = nn.Softmax(dim=0)(torch.matmul(pmatrix, test_vector))
    img = Image.open(img_path)
    # Preprocess image
    tsfm = kwargs["tsfm"]
    img = tsfm(img).unsqueeze(0)
    prediction = nn.Softmax(dim=1)(model_ft(img.to(device))).squeeze(0)
    print("-"*20)
    print("prediction with \"vit + mlp\" finetuning")
    for key, value in ground_truth.items():
        print(f"{value}:{prediction[key]*100:.04f}%")
    print("-"*20)
    
    
def make_fewshots_pred_noft(img_path, ground_truth:dict, pmatrix, **kwargs):
    test_vector = extract_feature_vector(img_path, show=True)
    # print(pmatrix.shape)
    prediction = nn.Softmax(dim=0)(torch.matmul(pmatrix, test_vector))
    print("-"*20)
    print("prediction with no-finetuning")
    for key, value in ground_truth.items():
        print(f"{value}:{prediction[key]*100:.04f}%")
    
    
    
def make_fewshots_pred_ft(img_path, ground_truth:dict, **kwargs):
    test_vector = extract_feature_vector(img_path, model=kwargs["model"], tsfm=kwargs["tsfm"], device=kwargs["device"], show=False)
    # print(pmatrix.shape)
    # prediction = nn.Softmax(dim=0)(torch.matmul(pmatrix, test_vector))
    prediction = nn.Softmax(dim=1)(cls_head(test_vector.unsqueeze(0))).squeeze()
    print("-"*20)
    print("prediction with head-finetuning")
    for key, value in ground_truth.items():
        print(f"{value}:{prediction[key]*100:.04f}%")