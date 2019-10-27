from src.config import models_path, results_path, MNIST_model_names, MNIST_datasets, training_images_file, training_labels_file_name, data_path

from src.data_processing.MNIST import prepare_for_model_training
from src.util.fileio import load_pkl_file, load_training_labels, show_image
from src.data_processing.number_extraction import extract_k_numbers
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, BatchNormalization
import keras
from src.config import MNIST_PIXEL

if __name__ == '__main__':
    training_images_file_path = os.path.join(data_path, training_images_file)
    training_labels_file_path = os.path.join(data_path, training_labels_file_name)

    x_test = load_pkl_file(training_images_file_path)
    y_test = load_training_labels(training_labels_file_path)

    X = np.empty((x_test.shape[0] * 6, 28, 3*28))
    Y = np.empty(y_test.shape[0] * 6).astype(int)
    for i in range(0, x_test.shape[0]):
        three_num = extract_k_numbers(x_test[i])
        Y[i*6:i*6+6] = y_test[i]
        X[i*6] = np.hstack((three_num[0], three_num[1], three_num[2]))
        X[i*6+1] = np.hstack((three_num[0], three_num[2], three_num[1]))
        X[i*6+2] = np.hstack((three_num[1], three_num[2], three_num[0]))
        X[i*6+3] = np.hstack((three_num[1], three_num[0], three_num[2]))
        X[i*6+4] = np.hstack((three_num[2], three_num[0], three_num[1]))
        X[i*6+5] = np.hstack((three_num[2], three_num[1], three_num[0]))

    X = prepare_for_model_training(X)
    Y = keras.utils.to_categorical(Y, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
                     input_shape=(MNIST_PIXEL, 3 * MNIST_PIXEL, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.20))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    for i in range(10):
        print("Epoch", i)

        model.fit(X[0:40000 * 6], Y[0:40000 * 6], epochs=1, verbose=1)

        results = model.evaluate(X[0:40000 * 6], Y[0:40000 * 6], verbose=0)

        print("\t\tEpoch " + str(i + 1) + "/10: training accuracy=" + str(results[1]),
              ", training loss=" + str(results[0]))

        results = model.evaluate(X[40000 * 6:], Y[40000 * 6:], verbose=0)

        print("\t\tEpoch " + str(i + 1) + "/10: validation accuracy=" + str(results[1]),
              ", validation loss=" + str(results[0]))
