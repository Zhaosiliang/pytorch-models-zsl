import os

from torch.utils.data import Dataset
from utils.utils import keep_image_size_open
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class ZslVOCDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.names = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        segment_name = self.names[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)

