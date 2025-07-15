from collections.abc import Callable
from typing import Union

from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph
from tensorial import gcnn
from tensorial.gcnn import atomic
from tensorial.gcnn.keys import predicted
import tensorial.typing as tt

from . import keys

__all__ = "MagneticShieldingTensor"


class MagneticShieldingTensor(linen.Module):
    """
    flax.linen.Module for computing magnetic shielding tensors σ_{k, ij}
    for each atom k, based on the linear response of the induced magnetic field
    B^{k}_ind to an applied external magnetic field B_ext.

    The Jacobian is computed as:
        σ_{k, ij} = - ∂B_ind_{k, i} / ∂B_ext_j

    Returns:
        A graph where each node has a (3, 3) tensor stored in `out_field`.
    """

    B_ind_fn: gcnn.GraphFunction
    B_ind_field: str = "B_ind_predicted"
    external_magnetic_field: str = "external_magnetic_field"
    out_field: str = "NMR_tensors_predicted"

    def setup(self) -> None:

        self._jacobian_fn = gcnn.jacobian(
            of=f"nodes.{self.B_ind_field}",
            wrt=f"globals.{self.external_magnetic_field}",
            has_aux=True,
            sum_axis=False,
        )(self.B_ind_fn)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        B_ext_zeros = jnp.zeros_like(graph.globals[self.external_magnetic_field])
        shielding, graph = self._jacobian_fn(graph, B_ext_zeros)

        shielding = -shielding.sum(2)

        updates = gcnn.utils.UpdateGraphDicts(graph)
        updates.nodes[self.out_field] = shielding

        graph = updates.get()
        # print("graph nodes keys:", graph.nodes.keys())
        # print("B_ind_predicted shape:", graph.nodes["B_ind_predicted"].shape)
        # print("NMR_tensors shape:", graph.nodes["NMR_tensors"].shape)
        # print("NMR_tensors_predicted shape:", graph.nodes["NMR_tensors_predicted"].shape)
        # print("graph globals keys:", graph.globals.keys())
        # print("external magnetic field shape:", graph.globals["external_magnetic_field"].shape)

        return graph
