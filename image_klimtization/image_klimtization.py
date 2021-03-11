#! python3

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
from skimage.transform import rescale

from neural_transfer import klimtize_background

# load the pretrained model trained on ADE20k dataset
model = pspnet_101_voc12()
model.summary()

input_image = "monti_draghi"
path_2_image = os.getcwd() + "/image_klimtization/" + input_image + ".jpeg"
img = Image.open(path_2_image)
original = np.asarray(img)

# -------------------------------------------------------------------------------
# - Semantic segmentation

out = model.predict_segmentation(
    inp=path_2_image,
    out_fname="out" + input_image + ".png"
)

# Select the pixels that belong to the person
id_unique = 15 # Corresponds to people
face_mask = (out == np.ones_like(out)*id_unique).astype(int)

# How much we need to upscale the output of the semantic segmentation
upscale_height_face = original.shape[0]/face_mask.shape[0]
upscale_width_face = original.shape[1]/face_mask.shape[1]

# Upscale the segmented image to the shape of the original
image_rescaled = rescale(face_mask,
                        (upscale_height_face, upscale_width_face),
                        anti_aliasing=True,
                        preserve_range = True)

face_mask_rescaled = (image_rescaled >= np.ones_like(image_rescaled)*0.5)

# Add alpha channel to the original image
a_channel = Image.new('L', img.size, 255)
img.putalpha(a_channel)

# Make the pixels that do not belong to the person segment, transparent
original_copy = np.copy(np.asarray(img))
original_copy[~face_mask_rescaled,3] = 0

face_original = Image.fromarray(original_copy)

# -------------------------------------------------------------------------------
# - Style transfer

# Convert the original image into Klimt style
background_klimt = klimtize_background(path_2_background = path_2_image)
background_klimt = np.asarray(background_klimt)

# How much we need to upscale the output of the style transfer output
upscale_height_background = original.shape[0]/background_klimt.shape[0]
upscale_width_background = original.shape[1]/background_klimt.shape[1]

image_rescaled = rescale(background_klimt,
                        (upscale_height_background, upscale_width_background, 1),
                        anti_aliasing=True,
                        preserve_range = True)/255.0

image_rescaled = Image.fromarray(np.uint8(image_rescaled*255))

# Take the segment belonging to the person taken from the original image and
# paste it on the background that is now in the style of Klimt
image_rescaled.paste(face_original, (0,0), face_original)
image_rescaled.show()
image_rescaled.save("klimtized_" + input_image + ".jpeg")
