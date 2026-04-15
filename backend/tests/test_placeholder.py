"""
Placeholder tests — expand this file as you build out the application.

These smoke tests ensure pytest can always be invoked successfully in CI,
even while the test suite is still being developed.
"""


def test_placeholder_always_passes():
    """Baseline test so pytest collects at least one test."""
    assert True


def test_environment_import():
    """Ensure Python imports are working correctly in the test environment."""
    import os
    import sys

    # Python version must be 3.9+
    assert sys.version_info >= (3, 9), "Python 3.9+ required"

    # Basic sanity: os module works
    assert os.path.sep in ("/", "\\")
