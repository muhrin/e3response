import logging
import pathlib
import pickle
from typing import Any, Callable, Final, Optional, Sequence, Union

from ase import Atoms
import jraph
import reax
from tensorial import gcnn
from typing_extensions import override

from e3response import keys

_LOGGER = logging.getLogger(__name__)

__all__ = ("CshNmrDataModule",)


class CshNmrDataModule(reax.DataModule):
    """Calcium Silicate Hydrate dataset from pre-processed pickle file containing ASE atoms objects with NMR tensor data."""

    _max_padding: gcnn.data.GraphPadding = None

    def __init__(
        self,
        r_max: float,
        data_file: Union[str, pathlib.Path] = "data/csh_nmr/csh_dataset.pkl",
        train_val_test_split: Sequence[Union[int, float]] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        limit: Optional[Union[int, str]] = None,
    ) -> None:
        super().__init__()

        # Params
        self._rmax: Final[float] = r_max
        self._data_file: Final[str] = str(data_file)
        self._train_val_test_split: Final[Sequence[Union[int, float]]] = train_val_test_split
        self._batch_size: Final[int] = batch_size
        self._limit = limit

        # State
        self.batch_size_per_device = batch_size
        self.data_train: Optional[reax.data.Dataset] = None
        self.data_val: Optional[reax.data.Dataset] = None
        self.data_test: Optional[reax.data.Dataset] = None

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        if self.data_train is not None:
            return

        structures = self._load_structures()

        train, val, test = reax.data.random_split(
            stage.rng, dataset=structures, lengths=self._train_val_test_split
        )

        def to_graph(atoms):
            return gcnn.atomic.graph_from_ase(
                atoms,
                r_max=self._rmax,
                atom_include_keys=("numbers", "NMR_tensors", "mask"),
                global_include_keys=[],
            )

        train_graphs = list(map(to_graph, train))
        val_graphs = list(map(to_graph, val))
        test_graphs = list(map(to_graph, test))

        def calc_padding(graphs):
            return gcnn.data.GraphBatcher.calculate_padding(
                graphs, batch_size=self._batch_size, with_shuffle=True
            )

        self._max_padding = gcnn.data.max_padding(
            *map(calc_padding, (train_graphs, val_graphs, test_graphs))
        )

        self.data_train = train_graphs
        self.data_val = val_graphs
        self.data_test = test_graphs

    def _load_structures(self) -> list[Atoms]:
        path = pathlib.Path(self._data_file)
        _LOGGER.info("Loading dataset from %s", path.absolute())

        with open(path, "rb") as file:
            structures = pickle.load(file)

        if not isinstance(structures, list) or not all(isinstance(s, Atoms) for s in structures):
            raise ValueError("Pickle file does not contain a list of `ase.Atoms` objects.")

        actual_limit = None
        if isinstance(self._limit, str):
            if self._limit.lower() == "only bulk":
                actual_limit = 277
            else:
                raise ValueError(f"Unknown limit option: {self._limit}")
        else:
            actual_limit = self._limit

        if actual_limit is not None:
            structures = structures[:actual_limit]

        _LOGGER.info(f"Number of loaded structures: {len(structures)}")

        return structures

    @override
    def train_dataloader(self) -> reax.DataLoader[Any]:
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException("Call setup() before dataloader.")
        return gcnn.data.GraphLoader(
            self.data_train,
            batch_size=self._batch_size,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader[Any]:
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException("Call setup() before dataloader.")
        return gcnn.data.GraphLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader[Any]:
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException("Call setup() before dataloader.")
        return gcnn.data.GraphLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )
