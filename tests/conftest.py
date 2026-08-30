import os

from jax import random
import pytest
import reax


@pytest.fixture(autouse=True)
def no_jax_preallocate():
    # Make sure we don't pre allocate memory, this is just antisocial
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


@pytest.fixture
def test_engine() -> reax.Engine:
    return reax.Engine()
