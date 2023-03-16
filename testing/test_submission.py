"""Tests for submitted functions"""

import unittest
from submission.submission import *

import gym.spaces
import numpy as np


class MockPlayerHome:
    def __init__(self):
        self.action_space = gym.spaces.Box(-1 * np.ones(3), np.ones(3))
        self.obs_dict = {
            "t_in": True,
            "t_out": True,
            "t_out_6hr": True,
            "t_out_12hr": True,
            "ghi": True,
            "ghi_6hr": True,
            "ghi_12hr": True,
            "time_of_day": True,
            "day_of_week": True,
            "is_holiday": True,
            "occupancy_status": True,
            "my_demand": True
        }


class SubmissionLib(unittest.TestCase):
    def test_predict(self):
        """
        General test for predict function
        """
        home = MockPlayerHome()
        self.assertIsInstance(list(predict(home)), list, msg="Ensure")


if __name__ == '__main__':
    unittest.main()
