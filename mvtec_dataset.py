from PIL import Image

import os
import os.path
import sys
import torch
import torch.utils.data as data


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


class MVTec_AD(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def make_dataset(self, dir, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if self.phase == 'test':
            gt_dir = os.path.join(dir, 'ground_truth')
        dir = os.path.join(dir, self.phase)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self.phase == 'test':
                        if target == 'good':
                            gt_path = None
                        else:
                            gt_fname = fname.split('.')[0] + '_mask.png'
                            gt_path = os.path.join(gt_dir, target, gt_fname)
                    if is_valid_file(path):
                        if self.phase == 'test':
                            item = (path, gt_path, class_to_idx[target])
                        else:
                            item = (path, class_to_idx[target])
                        images.append(item)

        return images

    def __init__(self, root, transform=None,
                 mask_transform=None, extensions=IMG_EXTENSIONS,
                 is_valid_file=None, phase='train'):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        if phase not in ('train', 'test'):
            raise (RuntimeError(
                'phase of MVTec_AD dataset must be "train" or "test".'))
        self.phase = phase
        data_dir = os.path.join(self.root, phase)
        classes, class_to_idx = self._find_classes(data_dir)
        samples = self.make_dataset(
            self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + data_dir + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.extensions = extensions
        self.transform = transform
        self.mask_transform = mask_transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.imgs = self.samples
        self.targets = [s[1] for s in samples]

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(
                dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.phase == 'train':
            path, target = self.samples[index]
            sample = self.pil_loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            return sample, target
        else:
            path, gt_path, target = self.samples[index]
            sample = self.pil_loader(path)
            if gt_path is None:
                gt_mask = Image.new('L', sample.size)
            else:
                gt_mask = Image.open(gt_path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.mask_transform is not None:
                gt_mask = self.mask_transform(gt_mask)
            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            return sample, gt_mask, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    imH = 512
    imW = 512
    class_dir = 'leather/'
    test_dataset_dir = '/home/cly/data_disk/MVTec_AD/data/' + class_dir
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    trans = transforms.Compose([
        # transforms.RandomCrop((imH, imW)),
        transforms.Resize((imH, imW)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trans2 = transforms.Compose([
        # transforms.RandomCrop((imH, imW)),
        transforms.Resize((imH, imW), Image.NEAREST),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    test_dataset = MVTec_AD(test_dataset_dir, transform=trans,
                            mask_transform=trans2, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    img, gt_mask, _ = next(iter(test_dataloader))
    print(img.shape)
    print(gt_mask.shape)
