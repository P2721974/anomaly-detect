# tests/test_utils/test_progress.py

import unittest
from io import StringIO
from contextlib import redirect_stdout
from utils.progress import tqdm_bar, single_bar, TqdmKerasCallback
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class TestProgressUtils(unittest.TestCase):

    def test_tqdm_bar_outputs_correct_length(self):
        data = list(range(5))
        output = []

        for i in tqdm_bar(data, desc="Testing tqdm_bar", unit="it", leave=False):
            output.append(i)

        self.assertEqual(output, data)

    def test_tqdm_keras_callback_updates_epochs(self):
        # Create a dummy Keras model
        model = Sequential([
            Dense(4, input_shape=(3,), activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy')

        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, size=(10,))

        callback = TqdmKerasCallback(total_epochs=3, desc="Train AE Test", unit="epoch")
        history = model.fit(X, y, epochs=3, batch_size=2, verbose=0, callbacks=[callback])

        self.assertEqual(len(history.history["loss"]), 3)


if __name__ == "__main__":
    unittest.main()
