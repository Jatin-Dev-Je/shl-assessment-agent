# conftest.py
# Configures pytest-asyncio for async test support.

import pytest

# Set asyncio mode to auto so @pytest.mark.asyncio works without decorating every test
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
