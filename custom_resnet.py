import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List, Tuple
from keras.applications import ResNet50
from keras import Model
from keras.layers.core.activation import Activation
import matplotlib.pyplot as plt

# TODO: check default parameter for random dataset augmentations
# TODO: change activation function of all layers that use an activation function, or only those of Activation layer?
# TODO: how to prepend/ append more layers like input or output layer


class NotInitializedError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PreTrainResNet:

    def __init__(self, batch_size: int = 32, image_size: int = 224, run_name: str = "resnet-training"):
        self.batch_size: int = batch_size
        self.image_size: int = image_size

        self.run_name: str = run_name

        self.ds_val: tf.data.Dataset = None
        self.ds_train: tf.data.Dataset = None
        self.resnet_model: Model = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def load_and_prepare_tfds_dataset(self, dataset_name: str = "imagenette"):
        print("Loading tfds dataset")
        self.ds_train, self.ds_val = load_tfds(dataset_name, self.batch_size)

        print("Preparing tfds dataset")
        self.ds_train = prepare_dataset(self.ds_train, img_size=self.image_size, batch_size=self.batch_size, apply_resnet50_preprocessing=True, shuffle=True, augment=False)
        self.ds_val = prepare_dataset(self.ds_val, img_size=self.image_size, batch_size=self.batch_size, apply_resnet50_preprocessing=True)

    def load_and_prepare_external_dataset(self, ds_train: tf.data.Dataset, ds_val: tf.data.Dataset):
        """
            Takes an external training and validation dataset and applies the augmentation and resizing pipeline to it.
            The modified datasets are then set in this class for further steps.
        """
        print("Loading and preparing external datasets")
        self.ds_train = prepare_dataset(ds_train, img_size=self.image_size, batch_size=self.batch_size, apply_resnet50_preprocessing=True, shuffle=True, augment=False)
        self.ds_val = prepare_dataset(ds_val, img_size=self.image_size, batch_size=self.batch_size, apply_resnet50_preprocessing=True)

    def load_resnet_model(self, custom_model: Model = None, pooling: str = None, pre_trained_weights_location: str = None, idx_unfreezed_layer: int = 0):
        """
             Loads a ResNet model and sets it for this class.
             If no custom model is given to this function. the default keras ResNet50 is used as a model.
             This default model is loaded without a top layer and initialized with random weights, unless the filename of pretrained weights is given.
             If a filename with pretrained weights is given, then these weights are loaded in. After this all layers except the last 'idx_unfreezed_layer'-layers are freezed, and are therefore not trainable anymore.
        """

        print("Loading ResNet model")
        if custom_model is not None:
            print("Using custom model")
            self.resnet_model: Model = custom_model
        else:
            print("Using Keras ResNet50 model")
            self.resnet_model: Model = ResNet50(
                include_top=False,
                weights=None,
                input_shape=(self.image_size, self.image_size, 3),
                pooling=pooling,  # pooling mode for when include_top is False
            )

        if pre_trained_weights_location is not None:
            self.resnet_model.load_weights(pre_trained_weights_location)
            # freeze layers until 'unfreeze_layers' index of layers (so the last x layers can still be trained)
            for layer in (self.resnet_model.layers) if not idx_unfreezed_layer else (self.resnet_model.layers[:-idx_unfreezed_layer]):
                layer.trainable = False

    def append_layers_to_resnet_model(self, layers: tf.keras.Sequential):
        self.resnet_model = Model(self.resnet_model, layers)

    def change_resnet_models_activation_function(self, new_activation_function: Activation):
        """
            Sets new activation function to all 'Activation' layer in a model
            This function works in-place and does not return anything.

            new_activation_function needs to be one of keras.activations

            Requires the ResNet model to be initialized.
        """
        self._check_resnet_model_initialized()

        # for layer in self.resnet_model.layers:
        #    # check if layer is an activation layer - if so, then replace activation function
        #    if isinstance(layer, Activation):
        #        layer.activation = new_activation_function

        # this code sets all activation functions to our new activation function
        # so not only the Activation layer are affected by this, but also conv2d activations
        for layer in self.resnet_model.layers:
            # check if layer is an activation layer - if so, then replace activation function
            if hasattr(layer, 'activation'):
                layer.activation = new_activation_function

    def compile_resnet_model(self, optimizer: tf.keras.optimizers.Optimizer, loss_function: tf.keras.losses.Loss, optimizer_learning_rate: float = 0.001, metrics: List[str] = ["accuracy"]):
        """
            Compiles the resnet model with the given optimizer and loss function.
            This step is required before actually training the resnet model.
        """
        print("Compiling ResNet Model")
        optimizer.learning_rate = optimizer_learning_rate
        self.resnet_model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics)

    def train_resnet_model(self, epochs: int, use_checkpoints: bool = True, save_model: bool = True, save_training_metrics_figure: bool = True):
        """
            Loads a dataset, either given by name to be downloaded from tfds or passed as parameter to this function.
        """

        self._check_datasets_initialized()
        self._check_resnet_model_initialized()

        cp_callback = None

        if use_checkpoints:
            # checkpoint setup
            checkpoint_path = f"training/{self.run_name}/{self.run_name}.ckpt"
            # this saves the weights for us, which we can reload later on
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        print(f"Training Model with {epochs} epochs")
        history = self.resnet_model.fit(
            self.ds_train,
            validation_data=self.ds_val,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[cp_callback]
        )

        if save_model:
            print("Saving model")
            self.resnet_model.save(f"saved_model/{self.run_name}.h5")

        if save_training_metrics_figure:
            visualize_training(history, epochs, f"Metrics-{self.run_name}.png")

    def get_model(self):
        pass

    def _check_resnet_model_initialized(self):
        if self.resnet_model is None:
            raise NotInitializedError("ResNet model was not initialized")

    def _check_datasets_initialized(self):
        if self.ds_val is None or self.ds_train is None:
            raise NotInitializedError("Validation/ Training dataset was not initialized")


def load_tfds(model_name: str, batch_size: int = 32, split: List[str] = ["train", "validation"], shuffle_files: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
        Loads a tfds dataset by name with a given batch size and split
    """
    print("Loading tfds dataset")

    ds_train, ds_test = tfds.load(
        model_name,
        split=split,
        as_supervised=True,
        batch_size=batch_size,
        shuffle_files=shuffle_files,
        with_info=False
    )

    return (ds_train, ds_test)


def prepare_dataset(ds: tf.data.Dataset, img_size: int, apply_resnet50_preprocessing: bool, batch_size: int, shuffle: bool = False, shuffle_seed: int = 42, augment: bool = False, random_rotation: float = 0.2, random_zoom: float = 0.1, random_flip: str = "horizontal_and_vertical") -> tf.data.Dataset:
    """
        Prepares datasets for training and validation for the ResNet50 model.
        This function applies image resizing, resnet50-preprocessing to the dataset. Optionally the data can be shuffled or further get augmented (random flipping, etc.)
    """
    AUTOTUNE = tf.data.AUTOTUNE

    resizer = tf.keras.Sequential([
        tf.keras.layers.Resizing(img_size, img_size),
    ])

    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(random_flip),
        tf.keras.layers.RandomRotation(random_rotation),
        tf.keras.layers.RandomZoom(random_zoom),
    ])

    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resizer(x), y),
                num_parallel_calls=AUTOTUNE)

    if apply_resnet50_preprocessing:
        ds = ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
                    num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(shuffle_seed)

    if batch_size is not None:
        # Batch all datasets.
        ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.cache().prefetch(buffer_size=AUTOTUNE)


def visualize_training(history, epochs, file_save_name: str = None):
    """
        Visualized the accurarcy metrics for a model's training process.
        If a filename is specified, the figure is not shown but directly saved as a file.
    """
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
