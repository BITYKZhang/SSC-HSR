import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def filter(imgs, loader, min_size=None):
    if min_size is None:
        return imgs
    filtered_imgs = []
    for img in imgs:
        w, h = loader(img).size
        if w < min_size or h < min_size:
            continue
        filtered_imgs.append(img)
    return filtered_imgs


def default_loader(path):
    return Image.open(path).convert('RGB')


def gray_loader(path):
    return Image.open(path).convert('L')


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, min_size=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        imgs = filter(imgs, loader=loader, min_size=min_size)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class ImageFolderDataset(ImageFolder):
    def __init__(
            self, root, common_transform=None, input_transform=None, target_transform=None, loader=default_loader,
            min_size=None):
        super().__init__(root, loader=loader, min_size=min_size)
        self.common_transform = common_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = super().__getitem__(index)
        input = output = self.common_transform(img)
        if self.input_transform is not None:
            input = self.input_transform(input)
        if self.target_transform is not None:
            output = self.target_transform(output)
        return input, output
