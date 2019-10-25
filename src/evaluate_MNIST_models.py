# This scripts will either load the models or retrain them
import os
from keras.models import Model

from src.data_processing.MNIST import get_MNIST
from src.models.mnist_predictor import get_model
from src.config import retrain_models, models_path
from src.util.fileio import load_model, save_model_weights


def evaluate_MNIST_model(model_str: str, dataset: str):
    """
    Evaluate the input model for the accuracy metric
    The results such as the confusion matrix will be saved to the results folder
    :param model_str: String code for the model to evaluate (CNN, RNN)
    :param dataset: String code for which dataset to train on (MNIST, PROC_MNIST)
    """
    (x_test, y_test) = get_MNIST(dataset)[1]

    if not retrain_models:
        try:
            model = get_model(model_str)
            model_path = os.path.join(models_path, model_str + ".h5")
            load_model(model_path, model)
        except:
            print("The model file cannot be found at " + model_path + " so it will be retrained.")
            model = train_model(model_str, dataset)

    else:
        model = train_model(model_str, dataset)

    print(model.evaluate(x_test, y_test))


def train_model(model_str: str, dataset: str, generate_results = True, show_graphs: bool = False):
    """
    Train the model, generate graphs of training and validation error and loss per epoch
    :param model_str: String code for the model to evaluate (CNN, RNN)
    :param dataset: Dataset code for what data to train on (MNIST, PROC_MNIST)
    :param generate_results: If true, the results of the training are saved in the results folder
    :param show_graphs: If true, the graphs are shown to the user
    :return: The optimal model
    """
    (x_train, y_train), (x_test, y_test) = get_MNIST(dataset)

    # Keep track of the validation and training accuracies as well as loss
    accuracy = {'training': [], 'validation': []}
    loss = {'training': [], 'validation': []}

    model: Model = get_model(model_str)
    model_path = os.path.join(models_path, model_str + "_" + dataset + ".h5")

    best_accuracy = 0.
    for i in range(10):
        # Perform one epoch
        model.fit(x=x_train, y=y_train, epochs=1)

        # Evaluate the model
        results = model.evaluate(x_train, y_train)

        loss['training'].append(results[0])
        accuracy['training'].append(results[1])

        print("Epoch " + str(i+1) + ": training accuracy=" + str(results[1]), ", training loss=" + str(results[0]))

        results = model.evaluate(x_test, y_test)

        loss['validation'].append(results[0])
        accuracy['validation'].append(results[1])

        print("Epoch " + str(i+1) + ": validation accuracy=" + str(results[1]), ", validation loss=" + str(results[0]))

        if best_accuracy < results[1]:
            save_model_weights(model_path, model)
            best_accuracy = results[1]

    load_model(model_path, model)

    print(accuracy['training'])
    print(accuracy['validation'])

    return model


if __name__ == '__main__':
    train_model("CNN", "MNIST")
