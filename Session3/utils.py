from __future__ import print_function
import os, sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Color:
    GRAY = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    CRIMSON = 38


def colorize(num, string, bold=False, highlight=False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))


def generate_image_patches_db(in_directory, out_directory, patch_size=64):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    total = 2688
    count = 0
    for split_dir in os.listdir(in_directory):
        if not os.path.exists(os.path.join(out_directory, split_dir)):
            os.makedirs(os.path.join(out_directory, split_dir))

        for class_dir in os.listdir(os.path.join(in_directory, split_dir)):
            if not os.path.exists(os.path.join(out_directory, split_dir, class_dir)):
                os.makedirs(os.path.join(out_directory, split_dir, class_dir))

            for imname in os.listdir(os.path.join(in_directory, split_dir, class_dir)):
                count += 1
                print('Processed images: ' + str(count) + ' / ' + str(total), end='\r')
                im = Image.open(os.path.join(in_directory, split_dir, class_dir, imname))
                # patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size), max_patches=256/patch_size)
                # patches = extract_patches(np.array(im), (patch_size, patch_size), overlap_allowed=0, crop_fraction_allowed=0)
                # patches = image.extract_patches(np.array(im), patch_size, extraction_step=patch_size)
                patches = create_patches(im, patch_size)
                for i, patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(
                        os.path.join(out_directory, split_dir, class_dir, imname.split(',')[0] + '_' + str(i) + '.jpg'))
    print('\n')


def create_patches(image, patch_size):
    imgwidth, imgheight = [256, 256]
    patches = []
    for i in range(0, imgheight, patch_size):
        for j in range(0, imgwidth, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            a = image.crop(box)
            patches.append(np.array(a))
    return patches
