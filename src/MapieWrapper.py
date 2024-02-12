from keras.layers import Dense, Flatten, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping


class TensorflowToMapie():
    """
    Class that aimes to make compatible a tensorflow model
    with MAPIE. To do so, this class create fit, predict,
    predict_proba and _sklearn_is_fitted_ attributes to the model.
    """
    def __init__(self, model: Sequential) -> None:
        self.model = model
        self.pred_proba = None
        # TODO - fix
        self.trained_ = True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs = 1, batch_size = None,
            verbose = 'auto', validation_data = None, callbacks = None ) -> None:
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=validation_data, callbacks=callbacks)
        self.trained_ = True
        self.classes_ = np.arange(self.model.layers[-1].units)

    # def predict_proba(self, X: np.ndarray) -> np.ndarray:
    #     preds = self.model.predict(X, verbose=0)
    #     print("in predict proba")
    #     print(preds[:, 0])
    #     return preds[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # pred_proba = self.predict_proba(X)
        # pred = (pred_proba == pred_proba.max(axis=1)[:, None]).astype(int)
        print("in predict")
        preds = self.model.predict(X, verbose=0)
        print(preds[:, 0])
        return preds[:, 0]

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False
