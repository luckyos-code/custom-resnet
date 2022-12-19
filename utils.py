import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, activations
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
    model.activation = new_activation_function

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


    THIS FUNCTION DOES NOT SEEMS TO BE NEEDED ANYMORE
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


def visualize_training(history, epochs, file_save_name: str = None):
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


def get_image_dataset_resizer(img_size: int) -> tf.keras.Sequential:
    """
        Returns a keras sequential layer to resize dataset to a same length and width provided by *img_size*
    """
    resizer = tf.keras.Sequential([
        tf.keras.layers.Resizing(img_size, img_size),
    ])
    return resizer


def get_image_dataset_augmentater() -> tf.keras.Sequential:
    """
        Runs data augmentation tasks to prevent overfitting
        Tasks run are:
            - Random Horizontal Flip
            - Random Vertical Flip
            - Random Rotation
            - Random Zoom
    """

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

    return data_augmentation


def prepare_dataset(ds: tf.data.Dataset, img_size: int, apply_resnet50_preprocessing: bool, batch_size: int = None, shuffle: bool = False, augment: bool = False) -> tf.data.Dataset:
    """
        Prepares datasets for training and validation for the ResNet50 model.
        This function applies image resizing, resnet50-preprocessing to the dataset. Optionally the data can be shuffled or further get augmented (random flipping, etc.)
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Resize and rescale all datasets.
    resizer = get_image_dataset_resizer(img_size)
    ds = ds.map(lambda x, y: (resizer(x), y),
                num_parallel_calls=AUTOTUNE)

    if apply_resnet50_preprocessing:
        ds = ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
                    num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    if batch_size is not None:
        # Batch all datasets.
        ds = ds.batch(batch_size)

    augmenter = get_image_dataset_augmentater()
    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


def load_model_weights(filedir: str, img_shape: tuple) -> tf.keras.Model:
    """
        Loads a ResNet50 model instance with weights provided from *filedir*.
    """
    model: tf.keras.Model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=filedir,
        input_shape=img_shape,
        pooling=None,
    )

    return model
