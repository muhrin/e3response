from jax import random
import pytest
import reax


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


@pytest.fixture
def test_engine() -> reax.Engine:
    return reax.Engine()
