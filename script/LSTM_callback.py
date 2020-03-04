import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
import numpy as np


class LSTM_callback(Callback):
    def __init__(self, logdir="./logs", validation_data=(), interval=1, scaler=None):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.logdir = logdir
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = y_pred.reshape(-1, 1)
            y_true = self.y_val.reshape(-1, 1)
            preds_unscaled = self.scaler.inverse_transform(y_pred)
            y_test_unscaled = self.scaler.inverse_transform(y_true)

            MSE = ((y_test_unscaled - preds_unscaled) ** 2).mean()

            file_writer = tf.summary.create_file_writer(os.path.join(self.logdir, "MSE"))
            file_writer.set_as_default()
            tf.summary.scalar('MSE unscaled', data=MSE, step=epoch)

            print("MSE unscaled - epoch: {:d} - score: {:.6f}".format(epoch, MSE))
