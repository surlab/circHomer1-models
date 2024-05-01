import src.computation as comp
import numpy as np

import pytest


def test_normalize_to_t0():
	timepoints = np.array([-1,0,1])
	values = np.array([1,2,3])
	assert (comp.normalize_to_t0(timepoints, values) == values/2).all()