## Biliotecas

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

## Parte 01 Pré-Processamento

## Funções para plotar as imagens

def display(display_list):
  plt.figure(figsize=(20, 20))

  title = ["Input Image", "True Mask", "Predicted Mask"]

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis("on")
  plt.show()

def display(display_list):
    plt.figure(figsize=(20, 20))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])

        # Use imshow diretamente em objetos Image
        plt.imshow(display_list[i])

        plt.axis("on")
    plt.show()

## Funções pré-processamentos das imagens

def resize(input_image, input_mask):
    #input_image = tf.image.resize(input_image, (128, 128), method="nearest")
    #input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
    input_image = input_image.resize((128, 128))
    input_mask = input_mask.resize((128, 128))

    return input_image, input_mask

def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    #input_mask -= 1
    #input_mask = tf.cast(input_mask, tf.float32) / 255.0

    return input_image, input_mask

def load_image_train(input_image, input_mask):
    input_image, input_mask = resize(input_image, input_mask)
    #input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(input_image, input_mask):
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

## Diretórios

train_image_dir = 'Cafe/Treinamento/train/image/'
train_mask_dir = 'Cafe/Treinamento/train/mask/'

validation_image_dir = 'Cafe/Treinamento/validation/image/'
validation_mask_dir = 'Cafe/Treinamento/validation/mask/'


## Função para processar as imagens em um diretório

def process_images(image_dir, mask_dir, arrayIMG, process_function):

    i = 0
    AX = np.empty((len(arrayIMG), 128, 128, 3), dtype=np.float32)
    AY = np.empty((len(arrayIMG), 128, 128), dtype=np.bool)
    for name in arrayIMG:

      # Constrói o caminho completo para a imagem e a máscara
      image_path = os.path.join(image_dir, f"{name}x.jpg")
      mask_path = os.path.join(mask_dir, f"{name}y.png")

      # Carrega a imagem e a máscara usando PIL
      input_image = Image.open(image_path)
      input_mask = Image.open(mask_path)

      #np.array(input_mask).shape
      #display([input_image, input_mask])

      # Corrige a orientação da imagem se necessário
      if name != '1553':
        input_image = input_image.rotate(-90, expand=True)

      #display([input_image, input_mask])

      # Aplica a função de processamento
      processed_image, processed_mask = process_function(input_image, input_mask)
      AX[i] = processed_image
      AY[i] = processed_mask

      sample_image, sample_mask = AX[i], AY[i]
      #display([sample_image, sample_mask])
      i=i+1
    return AX, AY

## Aplica a função de processamento aos conjuntos de treinamento e validação
namesTrain = ['39','101','107','1209','1307','1553','1558','1904','2221','2299']
namesValidation = ['51','2258']
ax, ay = process_images(train_image_dir, train_mask_dir, namesTrain, process_function=load_image_train)
vx, vy = process_images(validation_image_dir, validation_mask_dir, namesValidation, process_function=load_image_test)

## Parte 02 Montar a U-Net

## Modelo 01

def double_conv_block(x, n_filters):

    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

def build_unet_model():

    # inputs
    inputs = layers.Input(shape=(128,128,3))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model

unet_model = build_unet_model()
unet_model.summary()
tf.keras.utils.plot_model(unet_model, show_shapes=True)

## Modelo 02

# Create a function for a convolution block
def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu",
                               kernel_initializer="he_normal", padding="same")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), activation="relu",
                               kernel_initializer="he_normal", padding="same")(x)
    return x

# Create a function for the expanding path
def upsample_block(inputs, conv_prev, num_filters):
    up = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    concat = tf.keras.layers.concatenate([up, conv_prev])
    conv = conv_block(concat, num_filters)
    return conv

# Inputs
inputs = tf.keras.layers.Input((128, 128, 3))

# Normalization
s = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)

# Contraction path
c1 = conv_block(s, 16)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = conv_block(p1, 32)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = conv_block(p2, 64)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = conv_block(p3, 128)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5 = conv_block(p4, 256)

# Expansive path
c6 = upsample_block(c5, c4, 128)
c7 = upsample_block(c6, c3, 64)
c8 = upsample_block(c7, c2, 32)
c9 = upsample_block(c8, c1, 16)

# Output layer
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model2 = tf.keras.Model(inputs, outputs, name = "Unet2")

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model2.summary()

## Treinando a rede 01

## Model 01

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   steps_per_epoch = 5E,
                   loss=keras.losses.BinaryCrossentropy(),
                   metrics=[
                            "accuracy",
                            keras.metrics.FalseNegatives(),
                        ])
history=unet_model.fit(ax, ay, batch_size=2, epochs=10, verbose=2, validation_data=(vx,vy))

## Model 02

results = model2.fit(ax, ay, batch_size = 2, epochs = 15, verbose=2,validation_data=(vx,vy))

## Testando a rede

def binarize_images(images, threshold=0.18):
    binary_images = []

    for img in images:
        # Binarize the image based on the threshold
        binary_img = np.where(img < threshold, 0, 1)
        binary_images.append(binary_img)

    return binary_images

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset):
    for i in range(dataset.shape[0]):
        pred_mask = model2.predict(np.expand_dims(dataset[i], axis=0))
        display([dataset[i], create_mask(pred_mask)])

def display_images_with_predictions(images, predictions):

    predictions = binarize_images(predictions)
    plt.figure(figsize=(15, 5))

    for i in range(len(images)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis("on")

        plt.subplot(2, len(predictions), len(predictions) + i + 1)
        plt.imshow(predictions[i], cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("on")

    plt.show()

testNames = ['372','422','480','1056', '1399','1512','1535','1666']

root = '/content/drive/MyDrive/Colab Notebooks/Cafe/'

TX = []

i = 0
for image in  testNames:
      # Constrói o caminho completo para a imagem e a máscara
      image_path = os.path.join(root, f"{image}.jpg")

      # Carrega a imagem e a máscara usando PIL
      input_image = Image.open(image_path)

      #display([input_image])

      input_image = tf.image.resize(input_image, (128, 128), method="nearest")
      input_image = tf.cast(input_image, tf.float32) / 255.0
      #plt.imshow(input_image)
      #plt.show()

      TX.append(input_image)
      i=i+1

TX = np.array(TX, dtype=np.float32)
resultado = unet_model.predict(TX)
display_images_with_predictions(TX, resultado)


show_predictions(TX)