import tensorflow as tf
import pathlib
import urllib.request
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers.legacy import RMSprop
import os
import numpy as np
import urllib.request
import zipfile
from keras.applications.inception_v3 import InceptionV3


# print tensorflow version
print(tf.__version__)


# custom Keras callback
class KerasCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling  training!")
            self.model.stop_training = True


# function for printing horse or human based on result
def cat_or_dog(filename, result):
    if result > 0.5:
        print(f'{filename} is a cat')
    else:
        print(f'{filename} is a dog')

dataset_dir = 'datasets/PetImages'

# download the dataset
dataset_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
data_dir = tf.keras.utils.get_file('kagglecatsanddogs_5340.zip', origin=dataset_url, extract=True, cache_dir='./')
data_dir = pathlib.Path(dataset_dir).with_suffix('')

# total images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# filter-out corrupt images
num_skipped = 0
should_rewrite_image = True
for data_dir in ('Cat', 'Dog'):
    folder_path = os.path.join(dataset_dir, data_dir)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        is_jfif = True
        should_remove = False

        with open(fpath, "rb") as fobj:
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)

        try:
            img = tf.io.read_file(fpath)
            if not tf.io.is_jpeg(img):
                should_remove = True

            img = tf.image.decode_image(img)

            if img.ndim != 3:
                should_remove = True
        except:
            should_remove = True

        if (not is_jfif) or should_remove:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
        elif should_rewrite_image:
            tmp = tf.io.encode_jpeg(img)
            tf.io.write_file(fpath, tmp)

print('Deleted %d images' % num_skipped)

# generate training and validation dataset
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
BATCH_SIZE = 200

# training split
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# validation split
validation_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# print the labels / class names
class_names = train_ds.class_names
print(class_names)

# create test dataset
val_batches = tf.data.experimental.cardinality(validation_ds)
test_ds = validation_ds.take(val_batches // 5)
validation_ds = validation_ds.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

# configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_ds.prefetch(buffer_size=AUTOTUNE)

# normalization
normalization = tf.keras.layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    normalization,
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
validation_ds = validation_ds.map(lambda x, y: (normalization(x, training=True), y))

# create base model
base_model = InceptionV3(
    weights='imagenet',
    include_top=False
)

# layers must not be trainable
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# add a final sigmoid layer for classification
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

initial_epochs = 50

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=initial_epochs
)
