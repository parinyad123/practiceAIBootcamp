import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


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
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


'''
Step 0: Access the Fashion MNIST directly from TensorFlow
'''

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Since the class names are not included with the dataset, store them here to use later when plotting the images.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
Step 1: Let's explore the format of the dataset before training the model.
'''

print("Shape of training images: {}".format(train_images.shape))
print("Number of labels in the training set: {}".format(len(train_labels)))
print("Labels of training set: {}".format(train_labels))

print("Shape of testing images: {}".format(test_images.shape))
print("Number of labels in the testing set: {}\n".format(len(test_labels)))

'''
Step 2: The data must be preprocessed before training the network. 
If we inspect the first image in the training set, we will see that the pixel values fall in the range of 0 to 255
'''

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Let's scale these values to a range of 0 to 1 by dividing the values by 255 !
train_images = train_images / 255.0
test_images = test_images / 255.0


'''
Step 3: Display the first 25 images from the training set and display the class name below each image. 
Verify that the data is in the correct format and we're ready to build and train the network.
'''

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


'''
Step 4: Building the neural network requires configuring the layers of the model, then compiling the model. 
'''

# Dense means the densely-connected, or fully-connected, neural layer.
# The first Dense layer has 128 nodes (or neurons).
# The second (and last) layer is a 10-node softmax layer. This returns an array of 10 probability scores that sum to 1.
# Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
]) # Sequential = บอกให้เอา layer มาต่อกัน
 # Dense = fully connected = เชื่อมกันทุกตัว


# Now, we compile the model.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
Step 5: Train the model. 
'''

# The model is "fit" to the training data.
# As the model trains, the loss and accuracy metrics are displayed.
model.fit(train_images, train_labels, epochs=5)
# run ครบทุกตัวเรียก 1 epochs


'''
Step 6: Compare how the model performs on the test dataset and observe that 
the accuracy on the test dataset is a little less than the accuracy on the training dataset. 
This gap between training accuracy and test accuracy is an example of overfitting. 
'''

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc, '\n')

'''
Step 7: Make predictions. 
'''

predictions = model.predict(test_images)

# A prediction is an array of 10 numbers. These describe the "confidence" of the model that
# the image corresponds to each of the 10 different articles of clothing.
print("The first prediction: {}".format(predictions[0]))
print("The highest confidence appears at label#{} with actual label: {}\n".format(np.argmax(predictions[0]), test_labels[0]))

# Let's plot several images with their predictions.
# Correct prediction labels are blue and incorrect prediction labels are red.
# The number gives the percent (out of 100) for the predicted label.
# Note that it can be wrong even when very confident.

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()
