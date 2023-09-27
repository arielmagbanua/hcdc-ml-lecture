import tensorflow as tf
import pathlib
import urllib.request
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers.legacy import RMSprop
import os

# print tensorflow version
print(tf.__version__)


# custom Keras callback
class KerasCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.98:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# function for printing horse or human based on result
def cat_or_dog(filename, result):
    if result > 0.5:
        print(f'{filename} is a cat')
    else:
        print(f'{filename} is a dog')


# download the dataset
dataset_dir = 'datasets/PetImages'
dataset_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
data_dir = tf.keras.utils.get_file('kagglecatsanddogs_5340.zip', origin=dataset_url, extract=True, cache_dir='./')
data_dir = pathlib.Path(dataset_dir).with_suffix('')

# total images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# filter-out corrupt images
num_skipped = 0

for data_dir in ('Cat', 'Dog'):
    folder_path = os.path.join(dataset_dir, data_dir)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, 'rb')
            is_jfif = tf.compat.as_bytes('JFIF') in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print('Deleted %d images' % num_skipped)

# generate training and validation dataset
image_height = 150
image_width = 150
image_size = (image_height, image_width)
batch_size = 200

# split the dataset into training and validation
train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="both",
    seed=1337,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

# augment the dataset
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Resizing(image_height, image_width),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
])

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# download the weights and pretrained model
weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

# create the pre-trained model
pre_trained_model = InceptionV3(
    input_shape=(image_height, image_width, 3),
    include_top=False,
    weights=None
)

# load the weights
pre_trained_model.load_weights(weights_file)

# layers must not be trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# create and compile the model
model = tf.keras.Model(pre_trained_model.input, x)

# compile the model
model.compile(
    optimizer=RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# retrain the model
history = model.fit(
    augmented_train_ds,
    validation_data=validation_ds,
    epochs=125
)

# save model
model.save('cats-or-dogs.keras')
