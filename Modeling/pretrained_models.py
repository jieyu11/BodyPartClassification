from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    InceptionV3, InceptionResNetV2, VGG16, VGG19, ResNet50, ResNet50V2,
    ResNet101, ResNet101V2, ResNet152, ResNet152V2, DenseNet121, DenseNet169,
    DenseNet201
    )
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#
# pretrained models in tensorflow:
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
#
_pretrained_models_dict = {
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "ResNet50V2": ResNet50V2,
    "ResNet101": ResNet101,
    "ResNet101V2": ResNet101V2,
    "ResNet152": ResNet152,
    "ResNet152V2": ResNet152V2,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
}


def load_pretrained_model(pretrained, image_size=224):
    """
    loading pretrained model given name.
    """
    Model = _pretrained_models_dict.get(pretrained, None)
    if not Model:
        logger.error("Pretrained model name %s is not found in %s." % (
                     pretrained,
                     str([k for k in _pretrained_models_dict.keys()]))
                     )

    m = Model(weights='imagenet', input_shape=(image_size, image_size, 3),
              include_top=False, pooling='max')
    logger.info("Pretrained model %s is loaded." % pretrained)
    return m
