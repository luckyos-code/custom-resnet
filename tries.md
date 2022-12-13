resnet-imagenette-relu-1:
    - resnet with imagenette
    - unmodified activation functions

resnet-imagenette-tanh-1:
    - resnet with imagenette
    - model.activation = tanh, all layers that have a settable activation function is set to tanh

resnet-imagenette-tanh-2:
    - resnet with imagenette
    - model.activation = tanh


resnet-imagenette-tanh-3:
    - resnet with imagenette
    - model.activation = tanh
    - reloading model after changing activation

resnet-imagenette-tanh-4:
    - resnet with imagenette
    - model.activation = tanh, all layers that have a settable activation function is set to tanh
    - reloading model after changing activation

resnet-imagenette-tanh-5:
    - resnet with imagenette
    - change activation function of activation layer to tanh
    - reloading model after changing activation
