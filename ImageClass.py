import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import MaxPool2D

data = keras.datasets.cifar10

EPOCHS = 100
BATCH_SIZE = 32
val_split = 0.2

(train_images, train_labels), (test_images, test_labels) = data.load_data()

classes = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
<<<<<<< HEAD
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
=======
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
>>>>>>> e98e6a95822d79e0db1f45a711e888a424609e8f

def def_model():
    model = keras.Sequential([
        Conv2D(64, (3, 3), input_shape=(
            32, 32, 3), activation='relu'),
        Conv2D(64, (3, 3), input_shape=(
            32, 32, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(input_shape=()),
        #Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.L2(0.05)),
        #Dense(512, activation='relu', kernel_regularizer=keras.regularizers.L1(0.05)),
        #Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

def train_model(model):
    history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_split=val_split, batch_size=BATCH_SIZE, callbacks=callback)
    return history, model

def load_model():
<<<<<<< HEAD
    tf.saved_model.load('saved_image_model')
=======
    model = tf.keras.models.load_model('saved_image_model')
    return model
>>>>>>> e98e6a95822d79e0db1f45a711e888a424609e8f

def plot_model(history):
    plt.figure(figsize=(32, 32))
    plt.plot(history.history['accuracy'])

<<<<<<< HEAD
if __name__ == '__main__':
    try:
        model = load_model()
    except:
        model = def_model()
        try:
            history, model = train_model(model)
        except:
            print("An error occured")
        try:
            tf.saved_model.save(model, 'saved_image_model')
        except:
            print('Unable to write to file')
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    predictions = model.predict(test_images)


=======

try:
    model = load_model()
except:
    model = def_model()
    try:
        history, model = train_model(model)
    except:
        print("An error occured")
    try:
        model.save('saved_image_model')
    except:
        print('Unable to write to file')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

predictions = model.predict(test_images)
for i in range(5):
    plt.figure()
    plt.imshow(test_images[i])
    plt.title("Predicted " + classes[np.argmax(predictions[i])] + ", should have been " + classes[test_labels[i][0]])
    plt.xlabel(f"{round(np.max(predictions[i])*100)} % certain")
>>>>>>> e98e6a95822d79e0db1f45a711e888a424609e8f
