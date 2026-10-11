"""Reliability as an ordinary test, run with the rest of your suite.

    pytest examples/test_reliability.py
"""

import halflife
from my_agent import my_agent


def test_claims_agent_holds_up():
    halflife.assert_reliable(my_agent, min_half_life=10)
