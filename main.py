from utils import *
from keras import activations
import tensorflow as tf


# to make effectrs of the change we need to apply the modifcations of the model
# like i.e. here: https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py
# for this we need to save and reload the model
# TODO: check if this is still currently needed


RUN_NAME = "resnet-imagenette-tanh-3"


def main():
    AUTOTUNE = tf.data.AUTOTUNE

    BATCH_SIZE = 24
    train_ds = load_train_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)
    val_ds = load_val_ds("imagenette/160px-v2", batch_size=BATCH_SIZE)

    # resize datasets
    SIZE = (224, 224)
    train_ds = train_ds.map(lambda x, y: (resize_image(x, SIZE), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (resize_image(x, SIZE), y), num_parallel_calls=AUTOTUNE)

    # preprocess dataset
    train_ds = train_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model: Model = ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=10,
    )

    change_activation_function(model, activations.tanh)

    model = apply_modifications(model)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    # checkpoint setup
    checkpoint_path = f"training/{RUN_NAME}/cp.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback]
    )

    model.save(f"saved_model/{RUN_NAME}.h5")

    visualize_training(history, epochs, f"Metrics-{RUN_NAME}.png")


if __name__ == "__main__":
    main()
