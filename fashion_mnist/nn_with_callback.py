import tensorflow as tf
import numpy as np


# custom keras callback
class KerasCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# load and split the dataset
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# normalize
training_images = training_images / 255.0
test_images = test_images / 255.0

# create / compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# create the callback
callback = KerasCallback()

# train the model
model.fit(training_images, training_labels, epochs=50, callbacks=[callback])

# evaluate the model
model.evaluate(test_images, test_labels)

# inference / predict
classifications = model.predict(test_images)
prediction = np.argmax(classifications[1]) # get the index of highest value
print(prediction)  # prediction
print(test_labels[1])  # true value label
