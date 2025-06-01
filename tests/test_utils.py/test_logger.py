# tests/test_utils/test_logger.py

import unittest
import logging
from io import StringIO
from utils.logger import get_logger


class TestLoggerUtils(unittest.TestCase):

    def setUp(self):
        # Capture log output for inspection
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setLevel(logging.DEBUG)

    def test_get_logger_returns_logger_instance(self):
        logger = get_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_emits_messages_at_expected_level(self):
        logger = get_logger("test_logger")
        logger.setLevel(logging.INFO)
        logger.addHandler(self.handler)

        logger.debug("debug")   # should not appear
        logger.info("info")
        logger.warning("warn")

        self.handler.flush()
        output = self.log_stream.getvalue()

        self.assertIn("info", output)
        self.assertIn("warn", output)
        self.assertNotIn("debug", output)

    def test_logger_does_not_duplicate_handlers(self):
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")

        self.assertEqual(logger1, logger2)
        num_handlers = len(logger1.handlers)
        logger3 = get_logger("test_logger")
        self.assertEqual(len(logger3.handlers), num_handlers)


if __name__ == "__main__":
    unittest.main()