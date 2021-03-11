#! /usr/bin/python3

import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import tensorflow_hub as hub

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


def klimtize_background(path_2_background):


  def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

  content_name = "Eating"
  style_name = "klimt"

  content_path = path_2_background
  style_path = '/home/giovanni/PyProjects/coursera_ml_tensorflow_advanced/course_4_generative_deep_learning/styles/' + style_name + '.jpeg'


  def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


  def imshow(image, title=None):
    if len(image.shape) > 3:
      image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
      plt.title(title)


  content_image = load_img(content_path)
  style_image = load_img(style_path)


  height = content_image.shape[1]
  width = content_image.shape[2]


  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

  content_layers = ['block5_conv2'] 

  style_layers = ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1', 
                  'block4_conv1', 
                  'block5_conv1']

  num_content_layers = len(content_layers)
  num_style_layers = len(style_layers)

  def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


  def vgg_layers_custom_dimension(layer_names, height, width):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    input_layer = tf.keras.layers.Input(shape = (height, width, 3), name="input_2")
    x = tf.keras.layers.Lambda( 
      lambda image: tf.image.resize( 
          image, (224, 224), 
          method = tf.image.ResizeMethod.BICUBIC,
          # align_corners = True, # possibly important
          preserve_aspect_ratio = True
      )
    )(input_layer)
    
    for i, layer in enumerate(vgg.layers):
      if i == 0:
        continue
      elif i != len(vgg.layers)-1:
        x = layer(x)
      else:
        output_layer = layer(x)

    expanded_model = tf.keras.Model(input_layer, output_layer)
    expanded_model.trainable = False

    # expanded_model.compile(optimizer= "adam", loss = "sparse_categorical_crossentropy", metrics= ["acc"])


    # model.get_layer("vgg19").get_layer('block1_conv1')
    outputs = [expanded_model.get_layer(name).output for name in layer_names]
    
    model = tf.keras.Model([expanded_model.input], outputs)
    return model



  # style_extractor = vgg_layers_custom_dimension(style_layers, height, width)
  style_extractor = vgg_layers(style_layers)
  style_outputs = style_extractor(style_image*255)

  #Look at the statistics of each layer's output
  for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()


  def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


  class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
      super(StyleContentModel, self).__init__()
      self.vgg =  vgg_layers(style_layers + content_layers)
      self.style_layers = style_layers
      self.content_layers = content_layers
      self.num_style_layers = len(style_layers)
      self.vgg.trainable = False

    def call(self, inputs):
      "Expects float input in [0,1]"
      inputs = inputs*255.0
      preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
      outputs = self.vgg(preprocessed_input)
      style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])

      style_outputs = [gram_matrix(style_output)
                      for style_output in style_outputs]

      content_dict = {content_name:value 
                      for content_name, value 
                      in zip(self.content_layers, content_outputs)}

      style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

      return {'content':content_dict, 'style':style_dict}

  extractor = StyleContentModel(style_layers, content_layers)

  results = extractor(tf.constant(content_image))
  
  # -------------------------------------------------------------------------------------
  # Run gradient descent
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  image = tf.Variable(content_image)

  def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  style_weight=1e-2
  content_weight=1e4

  # Coursera values
  # style_weight =  1e-4
  # content_weight = 1e-32

  def style_content_loss(outputs):
      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
      style_loss *= style_weight / num_style_layers

      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                              for name in content_outputs.keys()])
      content_loss *= content_weight / num_content_layers
      loss = style_loss + content_loss
      return loss

  @tf.function()
  def train_step(image):
    with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

  train_step(image)
  train_step(image)
  train_step(image)
  plt.show(plt.imshow(tensor_to_image(image)))

  import time
  start = time.time()

  epochs = 2
  steps_per_epoch = 30

  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image)
      print(".", end='')
    display.clear_output(wait=True)
    img = tensor_to_image(image)
    plt.show(plt.imshow(img))
    # img.save(content_name + "_" + style_name + "_" + str(n) + ".jpeg")
    print("Train step: {}".format(step))

  end = time.time()
  print("Total time: {:.1f}".format(end-start))
  return img
