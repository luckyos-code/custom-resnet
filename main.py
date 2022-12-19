from utils import *
from keras import activations
import tensorflow as tf
from keras.applications import ResNet50

import numpy as np

RUN_NAME = "test-1"


def train_resnet():
    BATCH_SIZE = 16
    CLASSES = 10
    train_ds = load_train_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)  # this does only returns cropped images with unused image space
    val_ds = load_val_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)

    # resize datasets
    IMG_SIZE = 224

    print("Preparing dataset")
    train_ds = prepare_dataset(train_ds, img_size=IMG_SIZE, apply_resnet50_preprocessing=True, shuffle=True, augment=False)
    val_ds = prepare_dataset(val_ds, img_size=IMG_SIZE, apply_resnet50_preprocessing=True)

    print("Loading Model")
    model: Model = ResNet50(
        include_top=True,
        weights=None,
        # input_shape=IMG_SHAPE, # -> if include_top is specified, then dont use input_shape, but use default shape of (224,224,3)
        classes=CLASSES,
        classifier_activation=None,  # activation function for "top" layer, if None -> returns logits
    )

    #change_activation_function(model, activations.tanh)

    print("Compiling Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    # checkpoint setup
    checkpoint_path = f"training/{RUN_NAME}/{RUN_NAME}.ckpt"
    # this saves the weights for us, which we can reload later on
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    epochs = 1
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


def load_resnet():
    IMG_SHAPE = (224, 224, 3)  # channel_last data format
    IMG_SIZE = 224
    BATCH_SIZE = 16
    val_ds = load_val_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)
    val_ds = prepare_dataset(val_ds, img_size=IMG_SIZE, apply_resnet50_preprocessing=True)

    model: tf.keras.Model = load_model_weights(IMG_SHAPE)
    model.load_weights("training/test-1/test-1.ckpt")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    _, acc = model.evaluate(val_ds, verbose="2")
    print("Loaded model, accuracy: {:5.2f}%".format(100 * acc))


def main():
    # train_resnet()
    load_resnet()


if __name__ == "__main__":
    main()
