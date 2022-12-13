import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, activations
from keras.applications import ResNet50
from keras.layers.core.activation import Activation
import tensorflow_datasets as tfds
from keras.models import load_model

import tempfile
import os


def load_train_ds(model_name: str, batch_size: int = 32):
    # use imagenette model for testing purposes
    ds = tfds.load(
        model_name,
        split=["train"],
        as_supervised=True,
        batch_size=batch_size,
        shuffle_files=True,
        with_info=False
    )
    return ds[0]


def load_val_ds(model_name: str, batch_size: int = 32):
    # use imagenette model for testing purposes
    ds = tfds.load(
        model_name,
        split=["validation"],
        as_supervised=True,
        batch_size=batch_size,
        with_info=False
    )
    return ds[0]


def resize_image(images, size):
    """
        Resizes given images to size
    """
    return tf.image.resize(images, size)


def change_activation_function(model: Model, new_activation_function):
    """
        Sets new activation function to all 'Activation' layer in a model
        This function works in-place and does not return anything.

        new_activation_function needs to be one of keras.activations
    """

    # TODO: do we want to change all activation function (Conv2D too) or just from the activation layer?
#    for layer in model.layers:
#        # check if layer is an activation layer - if so, then replace activation function
#        if isinstance(layer, Activation):
#            layer.activation = new_activation_function

    # is this line needed???
    model.activation = activations.tanh

    for layer in model.layers:
        # check if layer is an activation layer - if so, then replace activation function
        if hasattr(layer, 'activation'):
            layer.activation = new_activation_function


def apply_modifications(model, custom_objects=None):
    """
    Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.

    Copied from: https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py#L95
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)


def visualize_training(history, epochs, file_save_name : str = None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    if file_save_name is not None:
        plt.savefig(file_save_name)
    else:
        plt.show()
