# tests/test_utils/test_file_saver.py

import unittest
import tempfile
import os
import json
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from utils.file_saver import (
    safe_save_path,
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    save_keras_model
)


class TestFileSaverUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = os.path.join(self.temp_dir.name, "output")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_safe_save_path_appends_timestamp(self):
        result = safe_save_path(self.base_path, extension=".pkl")
        self.assertTrue(result.endswith(".pkl"))
        self.assertTrue("output_" in os.path.basename(result))

    def test_pickle_save_and_load(self):
        data = {"key": "value", "arr": [1, 2, 3]}
        path = os.path.join(self.temp_dir.name, "test.pkl")
        save_pickle(data, path)
        loaded = load_pickle(path)

        self.assertEqual(data["key"], loaded["key"])
        self.assertEqual(data["arr"], loaded["arr"])

    def test_json_save_and_load(self):
        metadata = {"model_type": "test", "accuracy": 0.95}
        path = os.path.join(self.temp_dir.name, "meta.json")
        save_json(metadata, path)
        with open(path) as f:
            raw = json.load(f)

        self.assertEqual(raw["model_type"], "test")
        self.assertAlmostEqual(raw["accuracy"], 0.95)

        loaded = load_json(path)
        self.assertEqual(loaded, metadata)

    def test_save_keras_model_creates_file(self):
        model = Sequential([Dense(4, input_shape=(3,), activation='relu'), Dense(2)])
        model_path = os.path.join(self.temp_dir.name, "keras_model.keras")

        save_keras_model(model, model_path)
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.getsize(model_path) > 0)


if __name__ == "__main__":
    unittest.main()
