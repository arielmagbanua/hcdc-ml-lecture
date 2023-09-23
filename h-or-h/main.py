import tensorflow as tf
import urllib.request
import zipfile


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

