import pathlib

import hydra
import pytest
import reax
import tensorial
from tensorial import reaxkit as rkit

import e3response
import e3response.data

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "configs"
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"


@pytest.mark.parametrize("model", ["nequip_electric", "mace"])
def test_load_train(model):
    trainer = reax.Trainer()

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        cfg = hydra.compose(config_name="train", overrides=[f"model={model}", "data=bto"])
        datamodule = e3response.data.BtoDataModule(r_max=5.0, data_dir=DATA_PATH / "bto")
        from_data = rkit.FromData(
            cfg["from_data"],
            engine=trainer.engine,
            rngs=trainer.rngs,
            datamodule=datamodule,
        )
        trainer.run(from_data)

        module = tensorial.config.instantiate(cfg["model"])
        assert isinstance(module, tensorial.ReaxModule)
