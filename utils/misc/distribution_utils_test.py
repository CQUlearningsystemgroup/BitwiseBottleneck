
""" Tests for distribution util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from utils.misc import distribution_utils


class GetDistributionStrategyTest(tf.test.TestCase):
  """Tests for get_distribution_strategy."""
  def test_one_device_strategy_cpu(self):
    ds = distribution_utils.get_distribution_strategy(num_gpus=0)
    self.assertEquals(ds.num_replicas_in_sync, 1)
    self.assertEquals(len(ds.extended.worker_devices), 1)
    self.assertIn('CPU', ds.extended.worker_devices[0])

  def test_one_device_strategy_gpu(self):
    ds = distribution_utils.get_distribution_strategy(num_gpus=1)
    self.assertEquals(ds.num_replicas_in_sync, 1)
    self.assertEquals(len(ds.extended.worker_devices), 1)
    self.assertIn('GPU', ds.extended.worker_devices[0])

  def test_mirrored_strategy(self):
    ds = distribution_utils.get_distribution_strategy(num_gpus=5)
    self.assertEquals(ds.num_replicas_in_sync, 5)
    self.assertEquals(len(ds.extended.worker_devices), 5)
    for device in ds.extended.worker_devices:
      self.assertIn('GPU', device)


class PerReplicaBatchSizeTest(tf.test.TestCase):
  """Tests for per_replica_batch_size."""

  def test_batch_size(self):
    self.assertEquals(
        distribution_utils.per_replica_batch_size(147, num_gpus=0), 147)
    self.assertEquals(
        distribution_utils.per_replica_batch_size(147, num_gpus=1), 147)
    self.assertEquals(
        distribution_utils.per_replica_batch_size(147, num_gpus=7), 21)

  def test_batch_size_with_remainder(self):
    with self.assertRaises(ValueError):
        distribution_utils.per_replica_batch_size(147, num_gpus=5)


if __name__ == "__main__":
  tf.test.main()
