import torch
from torch import nn
import torchvision
from torchinfo import summary

# load pretrained vgg network from torchvision
model = torchvision.models.vgg11_bn(weights="IMAGENET1K_V1")
summary(model)

# change model classifier to output 
model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=10, bias=True),
)
# initialize weight(xavier initialization)
torch.nn.init.xavier_normal_(model.classifier[0].weight)

# freeze some early layers for parameter reduction
intermediate_params = []
last_params = []
for name, para in model.named_parameters():
    if name in ["features.11.weight",
                "features.11.bias",
                "features.12.weight",
                "features.12.bias",
                "features.15.weight",
                "features.15.bias",
                "features.16.weight",
                "features.16.bias",
                "features.18.weight",
                "features.18.bias",
                "features.19.weight",
                "features.19.bias",
                "features.22.weight",
                "features.22.bias",
                "features.23.weight",
                "features.23.bias",
                "features.25.weight",
                "features.25.bias"
                "features.26.weight"
                "features.26.bias"
                ]:
        # print("feature abstraction layer:", name)
        para.requires_grad = True
        intermediate_params.append(para)
    elif "classifier" in name:
        # print("classifier layer:", name)
        para.requires_grad = True
        last_params.append(para)
    else:
        # print("non trainable layer:", name)
        para.requires_grad = False

summary(model)
# we can check that trainable parameters changed from 132,868,840 to 9,103,626
# it is mainly due to the change of fully connected layer not freezing the early layers.