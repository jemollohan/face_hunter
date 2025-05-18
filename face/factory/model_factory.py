
import tensorflow_hub as hub

from logger_factory import get_logger

logger = get_logger("model_factory")

MODEL_URL_KAGGLE_RESNET = "https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/TensorFlow2/640x640/1"

def load_model(model_url:str) -> None:
    logger.info("Loading Model from [{}]...".format(model_url))
    try:
        model = hub.load(model_url)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Please ensure you have an active internet connection if loading for the first time.")
        exit()
    return model


def get_kaggle_model():
    return load_model(MODEL_URL_KAGGLE_RESNET)