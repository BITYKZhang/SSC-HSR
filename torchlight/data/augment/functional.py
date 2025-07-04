import numpy as np

def vflip(img):
    return img[..., ::-1, :]


def hflip(img):
    return img[..., :, ::-1]


def brightness_change(img, strength):
    return img + strength


def multiplicative_color_change(img, strength):
    return img * strength


def grayscale(img):
    dst = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
    dst[0, :, :] = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    dst = np.repeat(dst, 3, axis=0)
    return dst


def blend(img1, img2, alpha=0.5):
    return img1 * alpha + img2 * (1-alpha)


def contrast_change(img, alpha):
    gs_2 = grayscale(img)
    img = blend(gs_2, img, alpha=alpha)
    return img


def rotate(img, k):
    return np.rot90(img, k)

