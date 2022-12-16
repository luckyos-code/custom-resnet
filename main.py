from utils import *
from keras import activations
import tensorflow as tf
from keras.applications import ResNet50

import numpy as np

# to make effectrs of the change we need to apply the modifcations of the model
# like i.e. here: https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py
# for this we need to save and reload the model
# TODO: check if this is still currently needed


RUN_NAME = "test-1"


def train_resnet():
    AUTOTUNE = tf.data.AUTOTUNE

    BATCH_SIZE = 32
    CLASSES = 10
    train_ds = load_train_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)  # this does only returns cropped images with unused image space
    val_ds = load_val_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)

    # resize datasets
    IMG_SIZE = (160, 160)
    train_ds = train_ds.map(lambda x, y: (resize_image(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (resize_image(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)

    # preprocess dataset
    train_ds = train_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    IMG_SHAPE = IMG_SIZE + (3,)

    print("Loading Model")
    model: Model = ResNet50(
        include_top=True,
        weights=None,
        input_shape=IMG_SHAPE,
        classes=CLASSES,
    )

    #change_activation_function(model, activations.tanh)

    print("Compiling Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    # checkpoint setup
    checkpoint_path = f"training/{RUN_NAME}/{RUN_NAME}.ckpt"
    # this saves the weights for us, which we can reload later on
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    epochs = 20
    print(f"Training Model with {epochs} epochs")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[cp_callback]
    )

    print("Saving model")
    model.save(f"saved_model/{RUN_NAME}.h5")

    visualize_training(history, epochs, f"Metrics-{RUN_NAME}.png")


def main():
    train_resnet()


if __name__ == "__main__":
    main()
