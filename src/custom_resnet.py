import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List, Tuple, Callable, Optional
from keras.applications import ResNet50
from keras import Model
import matplotlib.pyplot as plt
import os

# TODO: check default parameter for random dataset augmentations
# TODO: change activation function of all layers that use an activation function, or only those of Activation layer?


class PreTrainResNet:
    """
        PreTrainResNet class
        Can be used for transfer learning with a ResNet50 model.

        This class allows to load a default resnet50 model with random weights, or load a custom model.
        This model can be trained by train/ validation datasets provided by tfds or a custom dataset.


        Attributes
        -----------
        dir_saved_weights (str) - path, in which the model's weights can be saved
        dir_saved_model (str) - path, in which the model can be saved
        dir_checkpoints (str) - path, in which the model's checkpoints can be saved during training
        dir_plt_path (str) - path, in which the metrics created during training can be saved

    """

    dir_saved_weights: str = "weights"
    dir_saved_model: str = "models"
    dir_checkpoints: str = "checkpoints"
    dir_plt_path: str = "figures"

    def __init__(self, batch_size: int = 32, image_size: int = 224, run_name: str = "resnet-training"):
        self.batch_size: int = batch_size
        self.image_size: int = image_size
        self.resnet_input_shape: Tuple[int, int, int] = (self.image_size, self.image_size, 3)

        self.run_name: str = run_name
        """ Specifies the name for the current setup/ run """

        self.ds_val: tf.data.Dataset | None = None
        self.ds_train: tf.data.Dataset | None = None
        self.model: Model | None = None
        """ The classes ground-model. By default it is a ResNet50. """

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # we normally use the softmax activation at the end, so we disable logits in loss function

        self._setup_folders()

    def load_weights_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
            Loads weights from a checkpoint path.
            If no path is given, the default checkpoint path '/checkpoints/{run_name}' is used.
            The model itself needs to be loaded before calling this function in order to properly load the weights.

            Parameter
            -----------
            checkpoint_path (Optional[str]) - checkpoint path to restore model from
        """
        self._check_resnet_model_initialized()

        checkpoint = tf.train.Checkpoint(self.resnet_model)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.dir_checkpoints, self.run_name)

        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint.restore(checkpoint_path)

    def load_model_from_h5_file(self, h5_file_path: Optional[str] = None) -> None:
        """
            Loads a complete model (layers + weights + optimizer + loss function) from a hdf5 file.
            If no file path is given, the default path '/models/{run_name}' is used.

            Parameter
            -----------
            h5_file_path (Optional[str]) - file path of hdf5 file
        """
        if h5_file_path is None:
            h5_file_path = os.path.join(self.dir_saved_model, f"{self.run_name}.h5")
        self.resnet_model = tf.keras.models.load_model(h5_file_path)

    def load_resnet_model(self, custom_model: Optional[Model] = None, pooling: Optional[str] = None, pre_trained_weights_location: Optional[str] = None, freeze_layers: bool = True, idx_unfreezed_layer: int = 0):
        """
            Loads a ResNet model and sets it for this class.
            If no custom model is given to this function. the default keras ResNet50 is used as a model.
            This default model is loaded without a top layer.
            It is initialized with random weights, unless the filename of pretrained weights is given.
            If a filename with pretrained weights is given, then these weights are loaded in.

            Parameter
            -----------
            custom_model (Optional[tf.keras.Model]) - a custom model that can be loaded if specifed
            pooling (Optional[str]) - Optional pooling mode, None - means that the output of the model will be the 4D tensor output of the last convolutional block, avg - means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor, max - means that global max pooling will be applied.
            pre_trained_weights_location (Optional[str]) - defines the path from which pretrained weights can be loaded for the ResNet50 model
            freeze_layer (bool) - if True, sets the model's layers to not-trainable
            idx_unfreezed_layer (int) - If freeze_layers is True, all layers except the last 'idx_unfreezed_layer'-layers are freezed
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
            # expect_partial - Silence warnings about incomplete checkpoint restores
            # this prints warnings about unused optimizer variables or missing layers, we want to ignore them here
            self.resnet_model.load_weights(pre_trained_weights_location).expect_partial()

        if freeze_layers:
            # freeze all layers except the last 'idx_unfreezed_layer' layers
            for layer in (self.resnet_model.layers) if not idx_unfreezed_layer else (self.resnet_model.layers[:-idx_unfreezed_layer]):
                layer.trainable = False

    def append_layers_to_resnet_model(self, x: tf.Tensor):
        """
           Uses Keras' Functional Model API to append layers to the class' model.
           Requires the ResNet model to be initialized.

           Parameter
           -----------
           x (tf.Tensor) - Layer tensor to be appended to the existing model.
        """
        self._check_resnet_model_initialized()
        output = x(self.resnet_model.output)
        self.resnet_model = Model(inputs=self.resnet_model.layers[0].input, outputs=output)

    def prepend_layers_to_resnet_model(self, x: tf.Tensor):
        """
            Uses Keras' Functional Model API to prepend layers to the class' model.
            Requires the ResNet model to be initialized.

           Parameter
           -----------
           x (tf.Tensor) - Layer tensor to be prepended to the existing model.
        """
        self._check_resnet_model_initialized()
        self.resnet_model = Model(inputs=x, outputs=self.resnet_model(x))

    def change_resnet_models_activation_function(self, new_activation_function: Callable[[tf.Tensor], tf.Tensor]):
        """
            Sets new activation function to all 'Activation' layer in a model
            Requires the ResNet model to be initialized.

            Parameter
            -----------
            new_activation_function (Callable[[tf.Tensor], tf.Tensor]) - new activation set for the model, needs to be one of 'tf.keras.activations'

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
            Requires the ResNet model to be initialized.

            Parameter
            -----------
            optimizer (Optional[tf.keras.optimizers.Optimizer]) - Optimizer to be used when compiling the model
            loss_function (Optional[tf.keras.losses.Loss]) - Loss function to be used when compiling the model
            metrics (List[str]) - metrics as list of strings to be tracked when training the model (https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
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

    def train_model(self, epochs: int, use_checkpoints: bool = True, save_weights: bool = True, save_model: bool = True, save_training_metrics_figure: bool = True):
        """
            Trains the model.
            The model and train/evaluation datasets need to be initialized before calling this function

            Parameter
            -----------
            epochs (int) - Number of epochs to train the model
            use_checkpoints (bool) - if True, creates checkpoints after every epoch to save the current train weights, these are saved in '/checkpoints/{run_name}'
            save_weights (bool) - if True, saves the model's weights after training has finished, weight data is saved in '/weights/{run_name}'
            save_model (bool) - if True, saves the complete model when training has finished as a hdf5 file, file located in '/models/{run_name}'
            save_training_metrics_figure (bool) - if True, creates and saves a figure with the previously specified metrics created during training, file location of figure is '/figures/'
        """

        self._check_datasets_initialized()
        self._check_resnet_model_initialized()

        cp_callback = None

        if use_checkpoints:
            # checkpoint setup
            checkpoint_path = os.path.join(self.dir_checkpoints, self.run_name)
            print(f"Using checkpoints under: {checkpoint_path}")
            # this saves the weights for us, which we can reload later on
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True)

        print(f"Training Model with {epochs} epochs")
        history = self.resnet_model.fit(
            self.ds_train,
            validation_data=self.ds_val,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[cp_callback]
        )

        if save_weights:
            save_weights_path = os.path.join(self.dir_saved_weights, f"{self.run_name}")
            print(f"Saving model weights in: {save_weights_path}")
            self.resnet_model.save_weights(save_weights_path)

        if save_model:
            save_model_path = os.path.join(self.dir_saved_model, f"{self.run_name}.h5")
            print(f"Saving model in: {save_model_path}")
            self.resnet_model.save(save_model_path)

        if save_training_metrics_figure:
            figure_path = os.path.join(self.dir_plt_path, f"Metrics-{self.run_name}.png")
            visualize_training(history, epochs, figure_path)

    def evaluate_model(self, ds_eval: tf.data.Dataset, batch_size: int = 32):
        """
            Calls the models evaluation function on the given dataset.

            Parameter
            -----------
            ds_eval (tf.data.Dataset) - dataset to be evaluated by model
            batch_size (int) - batch size to be used for evaluating data
        """
        self._check_resnet_model_initialized()
        print("Evaluate on test data")
        results = self.resnet_model.evaluate(ds_eval, batch_size=batch_size)
        print("test loss, test acc:", results)

    def predict_single_input(self, value: tf.Tensor):
        """
            Prints the model's output for a single input tensor value.

            Parameter
            -----------
            value (tf.Tensor) - value for which a prediction should be done
        """
        self._check_resnet_model_initialized()
        prediction = self.resnet_model(value, training=False)
        print(f"Prediction: {prediction}")

    def load_prepare_tfds_train_dataset(self, dataset_name: str, resnet50_preprocessing: bool, shuffle: bool, augment: bool, split: List[str] = ["train"], shuffle_seed: int = 42, random_rotation: float = 0.2, random_zoom=0.1, random_flip: str = "horizontal_and_vertical"):
        """
            Loads and prepares a tfds training dataset.
            The dataset is automatically downloaded by the tfds module and only requires a dataset name.
            Preparation can include data shuffling, augmentation and resnet50-preprocessing.
            The dataset is then set as a training dataset for this class' model.

            Parameter
            -----------
            dataset_name (str) - Name of the tfds dataset to be used
            resnet50_preprocessing (bool) - If true, applies resnet50 specific preprocessing to the dataset
            shuffle (bool) - If true, shuffles the data
            augment (bool) - If true, applies the following augmentation steps to the data (rotation, zoom, flip)
            split (List[str]) - the split to be used for training dataset
            shuffle_seed (int) - Seed to be applied for the random shuffling
            random_rotation (float) - amount of random rotation applied to the data during augmentation
            random_zoom (float) - amount of random zooming applied to the data during augmentation
            random_flip (str) - specifies in which direction the data should be flipped during augmentation
        """
        print("Loading tfds train dataset")
        self.ds_train = load_tfds(ds_name=dataset_name, split=split, batch_size=self.batch_size)

        print("Preparing tfds train dataset")
        self.ds_train = prepare_dataset(self.ds_train, img_size=self.image_size, batch_size=None, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed, random_rotation=random_rotation, random_zoom=random_zoom, random_flip=random_flip)

    def load_prepare_tfds_validation_dataset(self, dataset_name: str, resnet50_preprocessing: bool, shuffle: bool, augment: bool, split: List[str] = ["validation"], shuffle_seed: int = 42, random_rotation: float = 0.2, random_zoom=0.1, random_flip: str = "horizontal_and_vertical"):
        """
            Loads and prepares a tfds validation dataset.
            The dataset is automatically downloaded by the tfds module and only requires a dataset name.
            Preparation can include data shuffling, augmentation and resnet50-preprocessing.
            The dataset is then set as a training dataset for this class' model.

            Parameter
            -----------
            dataset_name (str) - Name of the tfds dataset to be used
            resnet50_preprocessing (bool) - If true, applies resnet50 specific preprocessing to the dataset
            shuffle (bool) - If true, shuffles the data
            augment (bool) - If true, applies the following augmentation steps to the data (rotation, zoom, flip)
            split (List[str]) - the split to be used for training dataset
            shuffle_seed (int) - Seed to be applied for the random shuffling
            random_rotation (float) - amount of random rotation applied to the data during augmentation
            random_zoom (float) - amount of random zooming applied to the data during augmentation
            random_flip (str) - specifies in which direction the data should be flipped during augmentation
        """
        print("Loading tfds validation dataset")
        self.ds_val = load_tfds(ds_name=dataset_name, split=split, batch_size=self.batch_size)

        print("Preparing tfds validation dataset")
        self.ds_val = prepare_dataset(self.ds_train, img_size=self.image_size, batch_size=None, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed, random_rotation=random_rotation, random_zoom=random_zoom, random_flip=random_flip)

    def load_prepare_external_train_dataset(self, ds_train: tf.data.Dataset, resnet50_preprocessing: bool, shuffle: bool, augment: bool, shuffle_seed: int = 42, random_rotation: float = 0.2, random_zoom=0.1, random_flip: str = "horizontal_and_vertical"):
        """
            Loads and prepares an external train dataset.
            Preparation can include data shuffling, augmentation and resnet50-preprocessing.
            The dataset is then set as a training dataset for this class' model.

            Parameter
            -----------
            ds_train (tf.data.Dataset) - external dataset to be set for this class as train dataset
            resnet50_preprocessing (bool) - If true, applies resnet50 specific preprocessing to the dataset
            shuffle (bool) - If true, shuffles the data
            augment (bool) - If true, applies the following augmentation steps to the data (rotation, zoom, flip)
            split (List[str]) - the split to be used for training dataset
            shuffle_seed (int) - Seed to be applied for the random shuffling
            random_rotation (float) - amount of random rotation applied to the data during augmentation
            random_zoom (float) - amount of random zooming applied to the data during augmentation
            random_flip (str) - specifies in which direction the data should be flipped during augmentation
        """
        print("Preparing external train dataset")
        self.ds_train = prepare_dataset(ds_train, img_size=self.image_size, batch_size=None, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed, random_rotation=random_rotation, random_zoom=random_zoom, random_flip=random_flip)

    def load_prepare_external_validation_dataset(self, ds_val: tf.data.Dataset, resnet50_preprocessing: bool, shuffle: bool, augment: bool, shuffle_seed: int = 42, random_rotation: float = 0.2, random_zoom=0.1, random_flip: str = "horizontal_and_vertical"):
        """
            Loads and prepares an external validation dataset.
            Preparation can include data shuffling, augmentation and resnet50-preprocessing.
            The dataset is then set as a training dataset for this class' model.

            Parameter
            -----------
            ds_val (tf.data.Dataset) - external dataset to be set for this class as validation dataset
            resnet50_preprocessing (bool) - If true, applies resnet50 specific preprocessing to the dataset
            shuffle (bool) - If true, shuffles the data
            augment (bool) - If true, applies the following augmentation steps to the data (rotation, zoom, flip)
            split (List[str]) - the split to be used for training dataset
            shuffle_seed (int) - Seed to be applied for the random shuffling
            random_rotation (float) - amount of random rotation applied to the data during augmentation
            random_zoom (float) - amount of random zooming applied to the data during augmentation
            random_flip (str) - specifies in which direction the data should be flipped during augmentation
        """
        print("Preparing external validation dataset")
        self.ds_val = prepare_dataset(ds_val, img_size=self.image_size, batch_size=None, resnet50_preprocessing=resnet50_preprocessing, shuffle=shuffle, augment=augment, shuffle_seed=shuffle_seed, random_rotation=random_rotation, random_zoom=random_zoom, random_flip=random_flip)

    def _setup_folders(self):
        check_create_folder(self.dir_saved_model)
        check_create_folder(self.dir_saved_weights)
        check_create_folder(self.dir_checkpoints)
        check_create_folder(self.dir_plt_path)

    def _check_resnet_model_initialized(self):
        if self.resnet_model is None:
            raise NotInitializedError("ResNet model was not initialized")

    def _check_datasets_initialized(self):
        if self.ds_val is None or self.ds_train is None:
            raise NotInitializedError("Validation/ Training dataset was not initialized")


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
        print(f"Saving metrics figure in: {file_save_name}")
        plt.savefig(file_save_name)
    else:
        plt.show()


def check_create_folder(dir: str):
    """
        Checks if a folder exists on the current file, if not, this function creates that folder.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    check_dir = os.path.join(base_dir, dir)
    if not os.path.exists(check_dir):
        print(f"Directory {check_dir} does not exist, creating it")
        os.mkdir(check_dir)


class NotInitializedError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
