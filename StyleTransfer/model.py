# Import lib
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.optim as optim
import requests
from torchvision import transforms, models

# Get the features
vgg = models.vgg19(pretrained=True).features

# Optimize target img
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

"""
    Data preprocessing
"""

def load_image(img_path, max_size=400, shape=None):
    if "http" in img_path:
        response = requests.get(img_path) # Get img by http url
        image = Image.open(BytesIO(response.content)).convert('RGB') 
    else: 
        image = Image.open(img_path).convert("RGB")
        
    if max(image.size) > max_size:
        size = max_size # resize in case of large img
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape 
    
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
   
    # get rid of transparent, alpha channel (:3) and batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

"""
    Load img
"""
content = load_image('images/octopus.jpg').to(device)

style = load_image('images/hockney.jpg', shape=content.shape[-2:]).to(device)

"""
    Create helper function
"""
def im_convert(tensor):
    # Unnormalize and display tensor as img
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze() 
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)
    
    return image

"""
    View image
"""
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

"""
    Map layer names to names in the content & style rep
"""
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': "conv1_1",
                 '5': 'conv2_1',
                 '10': 'conv3_1',
                 '19': 'conv4_1',
                 '21': 'conv4_2', # Where the content is represented
                 '28': 'conv5_1'}
    
    features = {}
    x = image
    
    # model._module is a dict containing the modules in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
        
    return features

"""
    Create gram matrix --> tensor * reshaped tensor
"""
def gram_matrix(tensor):
    # batch_size, depth, height, width
    _, d, h, w = tensor.size()
    
    # reshape 
    tensor = tensor.view(d, h * w)
    
    # calc
    gram = torch.mm(tensor, tensor.t())
    
    return gram

# Apply features on the loaded img
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calc gram matrix for each layer in style rep
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# let target be clone of content img --> then apply the style
target = content.clone().requires_grad_(True).to(device)

"""
    Define weights 
"""
style_weights = {'conv1_1': 1.,
                'conv2_1': 0.75,
                'conv3_1': 0.2,
                'conv4_1': 0.2,
                "conv5_1": 0.2}

content_weight = 1 
style_weight = 1e6

"""
    Define and update loss
"""
# Show img after 400 steps
show_every = 400 

# Iter hyperparams
optimizer = optim.Adam([target], lr = 0.003)
steps = 2000

for ii in range(1, steps+1):
    # get features from the target img
    target_features = get_features(target, vgg)
    
    # calc content loss from conv4_2 where all the content is rep
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    # define style loss
    style_loss = 0
    for layer in style_weights:
        # Get feature in each layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        
        # get style in each layer
        style_gram = style_grams[layer]
        
        # calc loss for each layer with the appropriate weight from each convo layer
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        
        # update loss counter
        style_loss += layer_style_loss / (d * h * w)
    
    
    # calc total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # now update target img 
    optimizer.zero_grad() # clear grad for any optimized var
    total_loss.backward() # backprop
    optimizer.step() # update weights
    
    # display img
    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()
    
""" 
    Display finalized target img
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

# Display side by side 
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))