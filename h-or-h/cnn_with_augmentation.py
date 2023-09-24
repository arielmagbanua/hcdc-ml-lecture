import tensorflow as tf
import urllib.request
import zipfile
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers.legacy import RMSprop
import numpy as np

# training dataset url
training_url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
# training file name and directory
training_file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
# download the training dataset
urllib.request.urlretrieve(training_url, training_file_name)

# extract to directory
zip_ref = zipfile.ZipFile(training_file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

# validation dataset url
validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
# validation file name and directory
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
# download the validation data set
urllib.request.urlretrieve(validation_url, validation_file_name)

# extract to directory
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

# all images will be rescaled by 1./255 and augment then images
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

# all images will be rescaled by 1./255
validation_datagen = ImageDataGenerator(
    rescale=1/255.
)
validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

# create the model
model = tf.keras.models.Sequential([
    # note the input shape is the desired size of the image 300x300 with 3 bytes color
    # this is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # the second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # the third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # the fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # the fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

# compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# train the model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)
print(history.history.keys())

training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

print(f'training loss: {training_loss}')
print(f'training accuracy: {training_accuracy}')
print(f'validation loss: {validation_loss}')
print(f'validation accuracy: {validation_accuracy}')


# function for printing horse or human based on result
def horse_or_human(filename, result):
    if result > 0.5:
        print(f'{filename} is a human')
    else:
        print(f'{filename} is a horse')


# download Actual Images
actual_images_dl = 'https://drive.google.com/uc?export=download&id=1guy6SPfNId625-dDk43w5bxKeZX0luwt'
filename = "actual_samples.zip"
dir = 'actual_samples'
urllib.request.urlretrieve(actual_images_dl, filename)

# extract to directory
zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall(dir)
zip_ref.close()

# image1.jpg
image1 = image.load_img(dir + '/image1.jpg', target_size=(300, 300))
# convert image to tensor
image1_np = image.img_to_array(image1) / 255.
image1_np = np.expand_dims(image1_np, axis=0)  # extend to make 3D array
image1_tensor = np.vstack([image1_np])
# make inference
classes = model.predict(image1_tensor)
result = classes[0]
horse_or_human('image1.jpg', result)

# image2.jpg
image2 = image.load_img(dir + '/image2.jpg', target_size=(300, 300))
# convert image to tensor
image2_np = image.img_to_array(image2) / 255.
image2_np = np.expand_dims(image2_np, axis=0)  # extend to make 3D array
image2_tensor = np.vstack([image2_np])
# make inference
classes = model.predict(image2_tensor)
result = classes[0]
horse_or_human('image2.jpg', result)

# image3.jpg
image3 = image.load_img(dir + '/image3.jpg', target_size=(300, 300))
# convert image to tensor
image3_np = image.img_to_array(image3) / 255.
image3_np = np.expand_dims(image3_np, axis=0)  # extend to make 3D array
image3_tensor = np.vstack([image3_np])
# make inference
classes = model.predict(image3_tensor)
result = classes[0]
horse_or_human('image3.jpg', result)
