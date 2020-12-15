

"""Mock objects and related functions for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MockBenchmarkLogger(object):
  """This is a mock logger that can be used in dependent tests."""

  def __init__(self):
    self.logged_metric = []

  def log_metric(self, name, value, unit=None, global_step=None,
                 extras=None):
    self.logged_metric.append({
        "name": name,
        "value": float(value),
        "unit": unit,
        "global_step": global_step,
        "extras": extras})
