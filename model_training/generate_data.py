#! python3

import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

path2 = os.getcwd() + "/model_training/"

backgrounds = os.listdir(path2 + "data_backgrounds")
foregrounds = os.listdir(path2 + "data_persons")

for i, foreground_name in enumerate(foregrounds):

    for j, background_name in enumerate(backgrounds):

        background = Image.open(path2 + "data_backgrounds/" + background_name)
        foreground = Image.open(path2 + "data_persons/" + foreground_name)
        empty_as_background = Image.fromarray(np.zeros_like(background))

        mask_transparency = Image.fromarray(np.array(foreground)[:,:,3])

        offset_width = int(-foreground.width/2 + background.width/2 + np.random.uniform(-background.width/4, background.width/4))
        offset_height = int(-foreground.height/2 + background.height/2 + np.random.uniform(-background.height/4, background.height/4))
        offset = tuple([offset_width, offset_height])

        background.paste(foreground, offset, foreground)
        empty_as_background.paste(mask_transparency, offset, mask_transparency)

        empty_as_background = (np.asarray(empty_as_background)/255.0).copy()
        empty_as_background[empty_as_background > 0.5] = 1
        empty_as_background[empty_as_background < 0.5] = 0
        empty_as_background = rgb2gray(empty_as_background)
        empty_as_background = Image.fromarray(np.uint8(empty_as_background)*255, mode = "L")

        background.save(f"{path2}/prepared_data/images/{str(i)}_{str(j)}_original.jpeg")
        empty_as_background.save(f"{path2}/prepared_data/masks/{str(i)}_{str(j)}_mask.jpeg")