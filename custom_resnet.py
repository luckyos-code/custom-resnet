import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List, Tuple, Callable, Optional
from keras.applications import ResNet50
from keras import Model
import matplotlib.pyplot as plt

# TODO: check default parameter for random dataset augmentations
# TODO: change activation function of all layers that use an activation function, or only those of Activation layer?
# TODO: how to prepend/ append more layers like input or output layer
# TODO: put prepcrocessing/ resizing/ augentation on GPU


class NotInitializedError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PreTrainResNet:

    def __init__(self, batch_size: int = 32, image_size: int = 224, run_name: str = "resnet-training"):
        self.batch_size: int = batch_size
        self.image_size: int = image_size
        self.resnet_input_shape: Tuple[int, int, int] = (self.image_size, self.image_size, 3)

        self.run_name: str = run_name

        self.ds_val: tf.data.Dataset | None = None
        self.ds_train: tf.data.Dataset | None = None
        self.resnet_model: Model | None = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # we normally use the softmax activation at the end, so we disable logits in loss function

    def load_resnet_model(self, custom_model: Optional[Model] = None, pooling: Optional[str] = None, pre_trained_weights_location: Optional[str] = None, idx_unfreezed_layer: int = 0):
        """
             Loads a ResNet model and sets it for this class.
             If no custom model is given to this function. the default keras ResNet50 is used as a model.
             This default model is loaded without a top layer and initialized with random weights, unless the filename of pretrained weights is given.
             If a filename with pretrained weights is given, then these weights are loaded in. After this all layers except the last 'idx_unfreezed_layer'-layers are freezed, and are therefore not trainable anymore.


            Pooling:
            Optional pooling mode
            None -  means that the output of the model will be the 4D tensor output of the last convolutional block.
            avg - means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
            max - means that global max pooling will be applied.
        """

        print("Loading ResNet model")
        if custom_model is not None:
            print("Using custom model")
            self.resnet_model = custom_model
        else:
            print("Using Keras ResNet50 model")
            self.resnet_model = ResNet50(
                include_top=False,
                weights=None,
                input_shape=self.resnet_input_shape,
                pooling=pooling,  # pooling mode for when include_top is False
            )

        if pre_trained_weights_location is not None:
            self.resnet_model.load_weights(pre_trained_weights_location)
            # freeze layers until 'unfreeze_layers' index of layers (so the last x layers can still be trained)
            for layer in (self.resnet_model.layers) if not idx_unfreezed_layer else (self.resnet_model.layers[:-idx_unfreezed_layer]):
                layer.trainable = False

    def append_layer_to_resnet_model(self, layer: tf.keras.layers.Layer):
        self._check_resnet_model_initialized()
        output = layer(self.resnet_model.output)
        self.resnet_model = Model(inputs=self.resnet_model.layers[0].input, outputs=output)

    def change_resnet_models_activation_function(self, new_activation_function: Callable[[tf.Tensor], tf.Tensor]):
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

    def compile_resnet_model(self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None, loss_function: Optional[tf.keras.losses.Loss] = None, metrics: List[str] = ["accuracy"]):
        """
            Compiles the resnet model with the given optimizer and loss function.
            This step is required before actually training the resnet model.
            If no optimizer or loss function is passed to this function, the classses default Adam optimizer and standard loss function 'SparseCategoricalCrossentropy' is used.
        """

        self._check_resnet_model_initialized()

        print("Compiling ResNet Model")

        if optimizer is None:
            optimizer = self.optimizer

        if loss_function is None:
            loss_function = self.loss_function

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

    def load_prepare_tfds_train_dataset(self, dataset_name: str, resnet50_preprocessing: bool, shuffle: bool, augment: bool, split: List[str] = ["train"], shuffle_seed: int = 42, random_rotation: float = 0.2, random_zoom=0.1, random_flip: str = "horizontal_and_vertical"):
        print("Loading tfds train dataset")
        self.ds_train = load_tfds(ds_name=dataset_name, split=split, batch_size=self.batch_size)

        print("Preparing tfds train dataset")
        self.ds_train = prepare_dataset(self.ds_train, img_size=self.image_size, batch_size=None, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed, random_rotation=random_rotation, random_zoom=random_zoom, random_flip=random_flip)

    def load_prepare_tfds_validation_dataset(self, dataset_name: str, resnet50_preprocessing: bool, shuffle: bool, augment: bool, split: List[str] = ["validation"], shuffle_seed: int = 42, random_rotation: float = 0.2, random_zoom=0.1, random_flip: str = "horizontal_and_vertical"):
        print("Loading tfds validation dataset")
        self.ds_val = load_tfds(ds_name=dataset_name, split=split, batch_size=self.batch_size)

        print("Preparing tfds validation dataset")
        self.ds_val = prepare_dataset(self.ds_train, img_size=self.image_size, batch_size=None, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed, random_rotation=random_rotation, random_zoom=random_zoom, random_flip=random_flip)

#    def load_prepare_external_train_dataset(self, ds: tf.data.Dataset, resnet50_preprocessing: bool, shuffle: bool, augment: bool):
#        print("Loading and preparing external train datasets")
#        self.ds_train = prepare_dataset(ds, img_size=self.image_size, batch_size=self.batch_size, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment)
#
#    def load_prepare_external_validation_dataset(self, ds: tf.data.Dataset, resnet50_preprocessing: bool, shuffle: bool, augment: bool):
#        print("Loading and preparing external validation datasets")
#        self.ds_val = prepare_dataset(ds, img_size=self.image_size, batch_size=self.batch_size, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment)


def load_tfds(ds_name: str, batch_size: int, split: List[str] = ["train"], shuffle_files: bool = False) -> tf.data.Dataset:
    """
        Loads a tfds dataset by name with a given batch size and split
        Batch size of -1 means loading the complete dataset at once
    """
    ds = tfds.load(
        ds_name,
        split=split,
        as_supervised=True,
        batch_size=batch_size,
        shuffle_files=shuffle_files,
        with_info=False
    )
    return ds[0]


def prepare_dataset(ds: tf.data.Dataset, img_size: int, resnet50_preprocessing: bool, batch_size: int | None, shuffle: bool = False, shuffle_seed: int = 42, augment: bool = False, random_rotation: float = 0.2, random_zoom: float = 0.1, random_flip: str = "horizontal_and_vertical") -> tf.data.Dataset:
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

    if resnet50_preprocessing:
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


def visualize_training(history, epochs, file_save_name: Optional[str] = None):
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
