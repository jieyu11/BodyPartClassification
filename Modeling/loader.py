from tensorflow.keras.models import load_model
import os
import logging

logger = logging.getLogger(__name__)


class Loader:
    """
    Class to load an existing XGBoost model.
    """

    def __init__(self, filename):
        """
        Filename of the model.
        """
        self.filename = filename

    def load_model(self):
        """
        Load an existing model.
        """
        model = None
        if not os.path.exists(self.filename):
            logger.error("Loading model: %s doesn't exist!" % self.filename)
        else:
            logger.info("Model: %s exists! Load it." % self.filename)
            model = load_model(self.filename)
        return model
