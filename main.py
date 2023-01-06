from custom_resnet import PreTrainResNet
import tensorflow as tf


def main():
    print(f"Tensorflow Version: {tf.version.VERSION}")

    resnet = PreTrainResNet(batch_size=32, image_size=224, run_name="custom-resnet-1")

    resnet.load_prepare_tfds_train_dataset("imagenette", resnet50_preprocessing=True, shuffle=True, augment=True)
    resnet.load_prepare_tfds_validation_dataset("imagenette", resnet50_preprocessing=True, shuffle=True, augment=False)

    # load an empty resnet50 model with random weights
    resnet.load_resnet_model(pooling="avg")

    output_layer = tf.keras.layers.Dense(10, activation="softmax")
    resnet.append_layer_to_resnet_model(output_layer)

    # compile model
    resnet.compile_resnet_model()

    # train model
    resnet.train_resnet_model(epochs=20, use_checkpoints=True, save_model=True, save_training_metrics_figure=True)


if __name__ == "__main__":
    main()
