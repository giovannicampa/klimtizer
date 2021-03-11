# Klimtizer
This project uses machine learning tools to transform an input image into a Klimt-style painting.

## Conversion procedure
The steps are the following:
1. A mask selecting pixels belonging to the human shapes will be produced through semantic segmentation 
2. The background will be rendered into the style of a Klimt painting through neural style transfer
3. The mask of __Point 1.__ wil be used to select the human shapes of the original image and overlap them to the _klimtized_ image

## Semantic segmentation
For the part of the semantic segmentation two approaches will be used.
A. Use a pretrained network
B. Train an own segmentation network

### Training of the segmentation network
To train the segmentation network we need labelled training data. This is generated automatically, by pasting images of human faces with transparent background, in random locations on a new background.
The pixel labels will be automatically generated with help of the transparency mask of the human image. All the pixels that belong to the human will be of class 1, while every other pixel, will have the label 0.  
