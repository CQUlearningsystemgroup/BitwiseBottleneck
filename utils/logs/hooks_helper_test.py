

"""Tests for hooks_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf  # pylint: disable=g-bad-import-order

from utils.logs import hooks_helper
from utils.misc import keras_utils


class BaseTest(unittest.TestCase):

  def setUp(self):
    super(BaseTest, self).setUp()
    if keras_utils.is_v2_0:
      tf.compat.v1.disable_eager_execution()

  def test_raise_in_non_list_names(self):
    with self.assertRaises(ValueError):
      hooks_helper.get_train_hooks(
          'LoggingTensorHook, ProfilerHook', model_dir="", batch_size=256)

  def test_raise_in_invalid_names(self):
    invalid_names = ['StepCounterHook', 'StopAtStepHook']
    with self.assertRaises(ValueError):
      hooks_helper.get_train_hooks(invalid_names, model_dir="", batch_size=256)

  def validate_train_hook_name(self,
                               test_hook_name,
                               expected_hook_name,
                               **kwargs):
    returned_hook = hooks_helper.get_train_hooks(
        [test_hook_name], model_dir="", **kwargs)
    self.assertEqual(len(returned_hook), 1)
    self.assertIsInstance(returned_hook[0], tf.estimator.SessionRunHook)
    self.assertEqual(returned_hook[0].__class__.__name__.lower(),
                     expected_hook_name)

  def test_get_train_hooks_logging_tensor_hook(self):
    self.validate_train_hook_name('LoggingTensorHook', 'loggingtensorhook')

  def test_get_train_hooks_profiler_hook(self):
    self.validate_train_hook_name('ProfilerHook', 'profilerhook')

  def test_get_train_hooks_examples_per_second_hook(self):
    self.validate_train_hook_name('ExamplesPerSecondHook',
                                  'examplespersecondhook')

  def test_get_logging_metric_hook(self):
    test_hook_name = 'LoggingMetricHook'
    self.validate_train_hook_name(test_hook_name, 'loggingmetrichook')

if __name__ == '__main__':
  tf.test.main()
