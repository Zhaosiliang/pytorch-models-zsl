import os

import torch

from models.unet import UNet
from utils.utils import keep_image_size_open
from utils.utils import keep_image_size_open
from torchvision import transforms
from torchvision.utils import save_image

transform = transforms.Compose([
    transforms.ToTensor()
])

net = UNet().cuda()
weight_path = 'params/VOC'
epoch = 1
i = 0
weights = f'{weight_path}/UNet{epoch}-{i}.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('success')
else:
    print('no loading')

_input = input('please input image path:')

img = keep_image_size_open(_input)
img_data = transform(img).cuda()
img_data = torch.unsqueeze(img_data, 0)
out = net(img_data)
name = _input.split('/')[-1]
save_image(out, f'result/result{name}')
print(out)
# /home/zsl/dataWorkspace/VOCdata/VOCdevkit/VOC2007/JPEGImages/000001.jpg
