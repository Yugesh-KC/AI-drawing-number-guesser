import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Decide if to load an existing model or to train a new one
train_new_model = False

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    cv2.imwrite('output_image0.png', X_test[0])

    X_train = np.where(X_train > 125, 0, 255)
    X_test = np.where(X_test > 125, 0,255 )

    # Reshape to add channel dimension for CNN input
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    
    cv2.imwrite('output_image0.png', X_test[0])
    cv2.imwrite('output_image1.png', X_test[1])
    cv2.imwrite('output_image2.png', X_test[2])




    # Create a neural network model with CNN layers
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test,verbose=2)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    # Saving the model
    model.save('handwritten_digits_cnn.keras')


def guesser(image_path):
    # Load the CNN model
    model = tf.keras.models.load_model('handwritten_digits_cnn.keras')

    # Preprocess the image for prediction
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, (28, 28))  # Resize to match model input shape
    img = img.reshape(1, 28, 28, 1)  # Reshape and normalize

    # Make prediction using the loaded model
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    probability = prediction[0][predicted_digit]

    # # Display the preprocessed image
    # plt.imshow(img[0], cmap='gray')
    # plt.title(f"Predicted Digit: {predicted_digit}, Probability: {probability:.2f}")
    # plt.axis('off')
    # plt.show()
    cv2.imwrite('themodel.png', img[0])
    print(probability)

    return predicted_digit, probability

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.where(X_train > 1, 255, X_train)
X_test = np.where(X_test > 1, 255, X_test)


# for i in range (2):
#     img=X_test[i]
#     plt.title(i)
#     plt.imshow(img, cmap='gray')
#     # plt.title(f"Predicted Digit: {predicted_digit}, Probability: {probability:.2f}")
#     plt.axis('off')
#     plt.show()
#     print(X_test.shape)



# print(guesser('digits/digit4.png'))
# print(guesser('zero.png'))