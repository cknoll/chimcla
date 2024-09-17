"""
This modules enables interactive debugging on failing tests.
It is tailored at the workflow of the main developer.
Delete it if it causes problems.

It is only active if a special environment variable is "True":

export PYTEST_IPS=True
"""

import os
import pytest

if os.getenv("PYTEST_IPS") == "True":
    import ipydex

    def pytest_runtest_setup(item):
        print("This invocation of pytest is customized")

    def pytest_exception_interact(node, call, report):
        # use frame_upcount=1 to prevent landing somewhere inside the unittest module
        # Note: this works for self.assertTrue but self.assertEqual would require frame_upcount=2
        # TODO: implement `frame_upcount_leave_ut=True` in ipydex
        ipydex.ips_excepthook(call.excinfo.type, call.excinfo.value, call.excinfo.tb, frame_upcount=1)


# source: https://stackoverflow.com/a/55301318 (and llm)
def pytest_addoption(parser):
    parser.addoption("--keep-data", action="store_true")
    parser.addoption("--no-parallel", action="store_true")

@pytest.fixture
def keep_data(request):
    return request.config.getoption("--keep-data")

@pytest.fixture
def no_parallel(request):
    return request.config.getoption("--no-parallel")