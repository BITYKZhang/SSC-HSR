import numpy as np

from .functional import hflip, vflip, brightness_change, multiplicative_color_change, contrast_change, rotate


# assume channel first

class Augment:
    def __call__(self, img):
        raise NotImplementedError


class Sequential(Augment):
    def __init__(self, augments):
        self.augments = augments

    def __call__(self, img):
        for augment in self.augments:
            img = augment(img)
        return img


class FlipHorizontal(Augment):
    def __init__(self):
        self.enable = np.random.randint(0, 2) > 0.5

    def __call__(self, img):
        if self.enable:
            return hflip(img)
        return img


class FlipVertical(Augment):
    def __init__(self):
        self.enable = np.random.randint(0, 2) > 0.5

    def __call__(self, img):
        if self.enable:
            return vflip(img)
        return img


class BrightnessChange(Augment):
    def __init__(self):
        self.strength = np.random.normal(loc=0, scale=0.02)

    def __call__(self, img):
        return brightness_change(img, self.strength)


class MultiplicativeColorChange(Augment):
    def __init__(self):
        self.strength = np.random.uniform(0.9, 1.1)

    def __call__(self, img):
        return multiplicative_color_change(img, self.strength)


class Contrast(Augment):
    def __init__(self):
        self.alpha = np.random.uniform(-0.3, 0.3)

    def __call__(self, img):
        return contrast_change(img, self.alpha)


# TODO: Fix
class Rotate(Augment):
    def __init__(self):
        self.k = np.random.randint(0, 4)

    def __call__(self, img):
        return rotate(img, self.k)


