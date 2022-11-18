import os.path

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data.VOCdataset import ZslVOCDataset
from models.unet import UNet
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/VOC'
data_path = '/home/zsl/dataWorkspace/VOCdata/VOCdevkit/VOC2007'
save_path = 'train_image'
if __name__ == '__main__':
    dataLoader = DataLoader(ZslVOCDataset(data_path), batch_size=2, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(os.path.join(weight_path, 'unet.pth')):
        net.load_state_dict(torch.load(weight_path))
        print('load weights successfully from {weight_path}'.format(weight_path=weight_path))
    else:
        print('failed')
    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()
    epoch = 1
    while True:
        for i, (image, segment_image) in enumerate(dataLoader):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch} - {i} - train_loss ===>> {train_loss.item()}')
            if i % 50 == 0:
                torch.save(net.state_dict(), f'{weight_path}/UNet{epoch}-{i}.pth')
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        epoch = epoch + 1
