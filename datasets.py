# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms
import torch

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


class TTFSWrapper(object):
    """
    Dataset wrapper that converts image tensor values into time-to-first-spike (TTFS)
    spike times. Assumes the wrapped dataset returns (img_tensor, label) where img_tensor
    is a torch.Tensor produced by `transforms.ToTensor()` followed by `transforms.Normalize(mean, std)`.

    The wrapper will reverse the normalization using the provided `mean` and `std`, clamp
    the resulting pixel values to [0,1], and map intensity I -> spike time t via
        t = t_min + (1 - I) * (t_max - t_min)

    Parameters
    - dataset: the base dataset (e.g., torchvision CIFAR/ ImageFolder)
    - mean: sequence of channel means used in normalization
    - std: sequence of channel stds used in normalization
    - t_min: earliest spike time (float)
    - t_max: latest spike time (float)
    - return_original: if True, wrapper returns (spike_times, original_tensor, label)
      otherwise returns (spike_times, label)
    """

    def __init__(self, dataset, t_min=0.0, t_max=1.0, return_original=False):
        self.dataset = dataset
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.return_original = return_original

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import torch as _torch

        img, label = self.dataset[idx]
        # if the dataset's transform returned a PIL image (no transform applied), try to convert
        if not isinstance(img, _torch.Tensor):
            # let the original dataset handle converting; raise for clarity
            raise TypeError("TTFSWrapper expects dataset to return a torch.Tensor image.")

        # Ensure float dtype and clamp to [0,1] (ToTensor should produce this range,
        # but resizing/interpolation may produce tiny out-of-bound values).
        img = img.to(dtype=_torch.float32)
        img_clamped = img.clamp(0.0, 1.0)

        # Map intensity to spike times: t = t_min + (1 - I) * (t_max - t_min)
        span = self.t_max - self.t_min
        spike_times = self.t_min + (1.0 - img_clamped) * span

        if self.return_original:
            return spike_times, img_clamped, label
        return spike_times, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "CIFAR":
        # root = args.data_path if is_train else args.eval_data_path
        # dataset = datasets.ImageFolder(root, transform=transform)
        # nb_classes = args.nb_classes


        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)         # dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10


        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    # Optionally wrap dataset to perform TTFS encoding (time-to-first-spike)
    if getattr(args, 'ttfs_convert', True):
        # Use dataset-specific mean/std for reversing normalization before TTFS encoding
        # if getattr(args, 'data_set', None) == 'CIFAR':
        #     # CIFAR mean/std (common convention for CIFAR-10/100)
        #     mean = [0.4914, 0.4822, 0.4465]
        #     std = [0.2023, 0.1994, 0.2010]
        # else:
        #     imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        #     mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        #     std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        t_min = getattr(args, 'ttfs_tmin', 0.0)
        t_max = getattr(args, 'ttfs_tmax', 1.0)
        return_original = getattr(args, 'ttfs_return_original', False)
        dataset = TTFSWrapper(dataset, t_min=t_min, t_max=t_max, return_original=return_original)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    # Use dataset-specific mean/std for CIFAR-10/100 if requested
 
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            # color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
     
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        # If TTFS encoding is requested, ensure we do NOT normalize here: TTFS
        # expects raw intensities in [0,1]. Remove Normalize if present.
        if getattr(args, 'ttfs_convert', False):
            if isinstance(transform, tuple):
                for trans in transform:
                    trans.transforms = [t for t in trans.transforms if not isinstance(t, transforms.Normalize)]
            else:
                transform.transforms = [t for t in transform.transforms if not isinstance(t, transforms.Normalize)]
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
