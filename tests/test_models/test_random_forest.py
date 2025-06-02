# tests/models/test_random_forest.py

import unittest
import tempfile
import os
import numpy as np

from models.random_forest import RandomForestModel


class TestRandomForestModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "rf_model")

        # 4 samples, 2 features, binary labels
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 1, 0, 1])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_and_evaluate_outputs_metrics(self):
        model = RandomForestModel(input_dim=2)
        model.train(self.X, y=self.y)
        metrics = model.evaluate(self.X, self.y)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_predict_outputs_labels(self):
        model = RandomForestModel(input_dim=2)
        model.train(self.X, y=self.y)
        preds = model.predict(self.X)

        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_model_save_and_load(self):
        model = RandomForestModel(input_dim=2)
        model.train(self.X, y=self.y)
        model.save(self.model_path, metrics={"accuracy": 1.0})

        new_model = RandomForestModel()
        new_model.load(self.model_path)
        preds = new_model.predict(self.X)

        self.assertEqual(len(preds), len(self.X))
        metadata = new_model.get_metadata(self.model_path)
        self.assertIn("model_type", metadata)


if __name__ == "__main__":
    unittest.main()