import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()

        def conv_block(in_channels, out_channels, pool=False):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


    #def forward(self, x):
        # return self.model(x)
       # pass

# Load model
def load_model(model_path):
    model = ResNet9(in_channels=3, num_classes=4)  # or your real config
    model.load_state_dict(torch.load("model/apple2.pth", map_location='cpu'))
    model.eval()
    return model

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to model input size
    transforms.ToTensor(),                  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize (ImageNet stats)
                         [0.229, 0.224, 0.225])
])

# Predict function
def predict(model, image_file):
    class_names = ['Apple___Cedar_apple_rust','Apple___Black_rot','Apple___Apple_scab', 'Apple___healthy']
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(image)
        print("Model output:", outputs)  # 🔍 debug print
        predicted_class = torch.argmax(outputs, dim=1).item()
    return class_names[predicted_class]