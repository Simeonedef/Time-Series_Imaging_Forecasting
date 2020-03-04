import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
import numpy as np


class MSE_HW_callback(Callback):
    def __init__(self, HW, test_size, seq_len=0, logdir="./logs", lstm=True, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.HW = HW
        self.test_size = test_size
        self.seq_length = seq_len
        self.logdir = logdir
        self.LSTM = lstm

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            HW = self.HW
            seq_len = self.seq_length
            test_size = self.test_size

            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = y_pred.reshape(-1)
            y_true = self.y_val


            if self.LSTM:
                forecast_multiplier = HW.forecast_multiplier[seq_len - 1:seq_len - 1 + test_size]

            else:
                forecast_multiplier = HW.forecast_multiplier[0:test_size]

            test_predictions = np.multiply(y_pred, forecast_multiplier)
            test_real = y_true[0:test_size]
            MSE_test = ((test_real - test_predictions) ** 2).mean()

            file_writer = tf.summary.create_file_writer(os.path.join(self.logdir, "MSE_HW"))
            file_writer.set_as_default()
            tf.summary.scalar('MSE with HW multiplier', data=MSE_test, step=epoch)

            print("MSE with HW - epoch: {:d} - score: {:.6f}".format(epoch, MSE_test))
