

import unittest

from absl import flags
import tensorflow as tf

from utils.flags import core as flags_core  # pylint: disable=g-bad-import-order


def define_flags():
  flags_core.define_base(num_gpu=False)
  flags_core.define_performance(dynamic_loss_scale=True, loss_scale=True)
  flags_core.define_image()
  flags_core.define_benchmark()


class BaseTester(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(BaseTester, cls).setUpClass()
    define_flags()

  def test_default_setting(self):
    """Test to ensure fields exist and defaults can be set.
    """

    defaults = dict(
        data_dir="dfgasf",
        model_dir="dfsdkjgbs",
        train_epochs=534,
        epochs_between_evals=15,
        batch_size=256,
        hooks=["LoggingTensorHook"],
        num_parallel_calls=18,
        inter_op_parallelism_threads=5,
        intra_op_parallelism_threads=10,
        data_format="channels_first"
    )

    flags_core.set_defaults(**defaults)
    flags_core.parse_flags()

    for key, value in defaults.items():
      assert flags.FLAGS.get_flag_value(name=key, default=None) == value

  def test_benchmark_setting(self):
    defaults = dict(
        hooks=["LoggingMetricHook"],
        benchmark_log_dir="/tmp/12345",
        gcp_project="project_abc",
    )

    flags_core.set_defaults(**defaults)
    flags_core.parse_flags()

    for key, value in defaults.items():
      assert flags.FLAGS.get_flag_value(name=key, default=None) == value

  def test_booleans(self):
    """Test to ensure boolean flags trigger as expected.
    """

    flags_core.parse_flags([__file__, "--use_synthetic_data"])

    assert flags.FLAGS.use_synthetic_data

  def test_parse_dtype_info(self):
    flags_core.parse_flags([__file__, "--dtype", "fp16"])
    self.assertEqual(flags_core.get_tf_dtype(flags.FLAGS), tf.float16)
    self.assertEqual(flags_core.get_loss_scale(flags.FLAGS,
                                               default_for_fp16=2), 2)

    flags_core.parse_flags(
        [__file__, "--dtype", "fp16", "--loss_scale", "5"])
    self.assertEqual(flags_core.get_loss_scale(flags.FLAGS,
                                               default_for_fp16=2), 5)

    flags_core.parse_flags(
        [__file__, "--dtype", "fp16", "--loss_scale", "dynamic"])
    self.assertEqual(flags_core.get_loss_scale(flags.FLAGS,
                                               default_for_fp16=2), "dynamic")

    flags_core.parse_flags([__file__, "--dtype", "fp32"])
    self.assertEqual(flags_core.get_tf_dtype(flags.FLAGS), tf.float32)
    self.assertEqual(flags_core.get_loss_scale(flags.FLAGS,
                                               default_for_fp16=2), 1)

    flags_core.parse_flags([__file__, "--dtype", "fp32", "--loss_scale", "5"])
    self.assertEqual(flags_core.get_loss_scale(flags.FLAGS,
                                               default_for_fp16=2), 5)


    with self.assertRaises(SystemExit):
      flags_core.parse_flags([__file__, "--dtype", "int8"])

    with self.assertRaises(SystemExit):
      flags_core.parse_flags([__file__, "--dtype", "fp16",
                              "--loss_scale", "abc"])

  def test_get_nondefault_flags_as_str(self):
    defaults = dict(
        clean=True,
        data_dir="abc",
        hooks=["LoggingTensorHook"],
        stop_threshold=1.5,
        use_synthetic_data=False
    )
    flags_core.set_defaults(**defaults)
    flags_core.parse_flags()

    expected_flags = ""
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)

    flags.FLAGS.clean = False
    expected_flags += "--noclean"
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)

    flags.FLAGS.data_dir = "xyz"
    expected_flags += " --data_dir=xyz"
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)

    flags.FLAGS.hooks = ["aaa", "bbb", "ccc"]
    expected_flags += " --hooks=aaa,bbb,ccc"
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)

    flags.FLAGS.stop_threshold = 3.
    expected_flags += " --stop_threshold=3.0"
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)

    flags.FLAGS.use_synthetic_data = True
    expected_flags += " --use_synthetic_data"
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)

    # Assert that explicit setting a flag to its default value does not cause it
    # to appear in the string
    flags.FLAGS.use_synthetic_data = False
    expected_flags = expected_flags[:-len(" --use_synthetic_data")]
    self.assertEqual(flags_core.get_nondefault_flags_as_str(), expected_flags)


if __name__ == "__main__":
  unittest.main()
