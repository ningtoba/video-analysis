"""pytest configuration."""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


# Enable asyncio for all tests marked with @pytest.mark.asyncio
pytest_plugins = ("pytest_asyncio",)
