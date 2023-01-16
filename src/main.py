from custom_resnet import PreTrainResNet
import tensorflow as tf

def test_resnet_training():
    resnet = PreTrainResNet(batch_size=32, image_size=224, run_name="custom-resnet-1")

    resnet.load_prepare_tfds_train_dataset("imagenette", resnet50_preprocessing=True, shuffle=True, augment=True)
    resnet.load_prepare_tfds_validation_dataset("imagenette", resnet50_preprocessing=True, shuffle=True, augment=False)

    # load an empty resnet50 model with random weights
    resnet.load_resnet_model(pooling="avg")

    output_layer = tf.keras.layers.Dense(10, activation="softmax", name="dense_output")
    resnet.append_layers_to_resnet_model(output_layer)
    resnet.model.summary()

    # compile model
    resnet.compile_resnet_model()

    # train model
    resnet.train_resnet_model(epochs=1, use_checkpoints=True, save_weights=True, save_model=True, save_training_metrics_figure=True)


def test_resnet_weight_loading():
    loaded_resnet = PreTrainResNet(batch_size=32, image_size=224, run_name="custom-resnet-1")
    loaded_resnet.load_resnet_model(pre_trained_weights_location="saved_weights/custom-resnet-1", freeze_layers=True, idx_unfreezed_layer=0)


def test_restoring():
    loaded_resnet = PreTrainResNet(batch_size=32, image_size=224, run_name="custom-resnet-1")

    loaded_resnet.load_prepare_tfds_train_dataset("imagenette", resnet50_preprocessing=True, shuffle=True, augment=True)
    loaded_resnet.load_prepare_tfds_validation_dataset("imagenette", resnet50_preprocessing=True, shuffle=True, augment=False)

    loaded_resnet.load_model_from_h5_file()

    # compile model
    loaded_resnet.compile_resnet_model()

    # train model
    loaded_resnet.train_resnet_model(epochs=10, use_checkpoints=True, save_weights=True, save_model=True, save_training_metrics_figure=True)


def main():
    print(f"Tensorflow Version: {tf.version.VERSION}")


if __name__ == "__main__":
    main()
