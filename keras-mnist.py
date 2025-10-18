import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# 1. Load the MNIST dataset
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# 2. Preprocess the data
xTrain = xTrain.reshape((60000, 28 * 28)).astype("float32") / 255  # Normalizes the data (0, 1)
xTest = xTest.reshape((10000, 28 * 28)).astype("float32") / 255  # Normalizes the data (0, 1)
yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)

# 3. Build the model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes (0–9)

# 4. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
model.fit(xTrain, yTrain, epochs=5, batch_size=32, validation_split=0.1)

# 6. Evaluate on test data
testLoss, testAcc = model.evaluate(xTest, yTest)
print(f"Test accuracy: {testAcc:.4f}")

while True:
    print("Use the following commands \n \
        t - test an image\n \
        q - to quit \n")

    userInput = input()
    if(userInput == 'q'):
        break

    elif(userInput == 't'):
        userInput = input()
        try:
            img = Image.open(userInput).convert('L') # Converting to grayscale
        except:
            print("Couldn't find image")

        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        imgArray = np.array(img) # Converting to numpy array
        imgArray = 255 - imgArray # Inverts the collors
        imgArray = imgArray.astype('float32') / 255 # Normalizes the data (0, 1)

        imgArray = imgArray.reshape(1, 28 * 28) # 1 - batch size, 28 * 28 - data

        prediction = model.predict(imgArray)
        print(np.argmax(prediction))

    else:
        print("unknown command")