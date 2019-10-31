import keras
import matplotlib as plt
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
from keras.models import Model,Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping
from math import ceil
import os
from keras.preprocessing.image import ImageDataGenerator

from src.config import data_path, training_labels_file_name, training_images_file, models_path, results_path, testing_images_file
from src.util.fileio import load_training_labels, load_pkl_file, dictionary_to_json, save_kaggle_results
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from src.util.fileio import save_confusion_matrix


def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Conv2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = keras.layers.add([res,out])

    return out

#Define the model


def MiniModel(input_shape):
    images = Input(input_shape)
    net = Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1], padding="same")(images)
    net = Unit(net,32)
    net = Unit(net,32)
    net = Unit(net,32)

    net = Unit(net,64,pool=True)
    net = Unit(net,64)
    net = Unit(net,64)

    net = Unit(net,128,pool=True)
    net = Unit(net,128)
    net = Unit(net,128)

    net = Unit(net, 256,pool=True)
    net = Unit(net, 256)
    net = Unit(net, 256)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    net = Dense(units=10,activation="softmax")(net)

    model = Model(inputs=images,outputs=net)

    return model


def train():
    # Get the file paths of the training data
    training_images_file_path = os.path.join(data_path, training_images_file)
    training_labels_file_path = os.path.join(data_path, training_labels_file_name)

    # Load the training data
    X = load_pkl_file(training_images_file_path)
    Y = load_training_labels(training_labels_file_path)

    (train_x, test_x, train_y, test_y) = train_test_split(X, Y, test_size=0.15, random_state=1, shuffle=True, stratify=Y)

    del X
    del Y
    #normalize the data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    #Subtract the mean image from both train and test set
    train_x = train_x - train_x.mean()
    test_x = test_x - test_x.mean()

    #Divide by the standard deviation
    train_x = train_x / train_x.std(axis=0)
    test_x = test_x / test_x.std(axis=0)

    train_x = train_x.reshape(train_x.shape + (1,))
    test_x = test_x.reshape(test_x.shape + (1,))

    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
    )

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_x)

    #Encode the labels to vectors
    train_y = keras.utils.to_categorical(train_y,10)
    test_y_og = test_y.copy()
    test_y = keras.utils.to_categorical(test_y,10)

    #define a common unit


    input_shape = (128,128,1)
    model = MiniModel(input_shape)

    #Print a Summary of the model

    model.summary()
    #Specify the training components
    model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])


    epochs = 50
    steps_per_epoch = ceil(50000/128)

    mc = ModelCheckpoint(os.path.join(models_path, 'RNN.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128),
                        validation_data=datagen.flow(test_x, test_y),
                        epochs=epochs,steps_per_epoch=steps_per_epoch, verbose=2, callbacks=[mc])

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('ResNN accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(results_path,'RNN_10deg_Acc.png'))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('ResNN loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(results_path, "RNN_10deg" + '_Loss.png'))

    model.load_weights(os.path.join(models_path, 'RNN.h5'))

    y_pred = model.predict(test_x).argmax(axis=1)
    cm = confusion_matrix(test_y_og,y_pred)

    results = model.evaluate(x=test_x, y=test_y, batch_size=128)

    r = {}
    r['acc'] = results[1]
    r['loss'] = results[0]
    dictionary_to_json(os.path.join(results_path, "RNN_results.json"), r)

    save_confusion_matrix(cm, fig_file_path=os.path.join(results_path, "Confusion_RNN.png"), classes=['1', '2', '3', '4', '5', '6', '7', '8', '9'], title="RNN Confusion matrix on unprocessed data")


    test_images_file_path = os.path.join(data_path, testing_images_file)
    x_test = load_pkl_file(test_images_file_path)

    x_test = x_test.astype('float32') / 255
    x_test = x_test - x_test.mean()

    x_test = x_test / x_test.std(axis=0)

    x_test = x_test.reshape(x_test.shape + (1,))

    pred = model.predict(x_test).argmax(axis=1)
    save_kaggle_results(os.path.join(results_path, "predictions_rnn.csv"), pred)

if __name__ == '__main__':
    train()