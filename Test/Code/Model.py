'''
## Licence:

This repository contains a variety of content; some developed by VARUN, and some from third-parties.
The third-party content is distributed under the license provided by those parties.
The content developed by VARUN is distributed under the following license:
I am providing code and resources in this repository to you under an open source license.
Because this is my personal repository, the license you receive to my code and resources is from me.

More about Licence at [link](https://github.com/t-varun/Face-Recognition/blob/master/LICENSE).
'''

# Import the requirements
import numpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

# Load the data
data_name = '../training_data_cleaned.npy'
Name = 'Face-Recognition_Dense-128x1_{}'.format(int(time.time()))

tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(Name))

data = np.load(data_name)

img = []
label = []

for item in data:
    img.append(item[0])
    label.append(item[1])

train_images = img[-500:]
train_labels = label[-500:]

test_images = img[:500]
test_labels = label[:500]

train_images = np.asarray(train_images)
test_images = np.asarray(test_images)

class_names = ['VARUN', 'BUNNY']

# Exploration of the data
# print(np.shape(train_images))
# print(len(train_labels))
# print(np.shape(test_images))
# print(len(test_labels))

# Process the data
train_images = train_images / 255.0

test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(60, 80)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(train_images, train_labels, epochs=5, callbacks=[tensorboard])
## tensorboard --logdir=logs/ --host=127.0.0.1

model.save('FR-TensorModel.model')

# Accuracy of the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Make Predictions
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
