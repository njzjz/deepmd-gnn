# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test models."""

import unittest
from copy import deepcopy
from typing import Any, Callable, ClassVar, Optional

import deepmd.pt.model  # noqa: F401
import numpy as np
import torch
from deepmd.dpmodel.output_def import (
    check_deriv,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    extend_input_and_build_neighbor_list,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from deepmd_gnn.mace import MaceModel
from deepmd_gnn.nequip import NequipModel

GLOBAL_SEED = 20240822

torch.set_default_dtype(torch.float64)


class PTTestCase:
    """Common test case."""

    module: "torch.nn.Module"
    """PT module to test."""

    skipTest: Callable[[str], None]  # noqa: N815
    """Skip test method."""

    @property
    def script_module(self) -> torch.jit.ScriptModule:
        """Script module."""
        with torch.jit.optimized_execution(should_optimize=False):
            return torch.jit.script(self.module)

    @property
    def deserialized_module(self) -> "torch.nn.Module":
        """Deserialized module."""
        return self.module.deserialize(self.module.serialize())

    @property
    def modules_to_test(self) -> list["torch.nn.Module"]:
        """Modules to test."""
        return [
            self.module,
            self.deserialized_module,
        ]

    def test_jit(self) -> None:
        """Test jit."""
        if getattr(self, "skip_test_jit", False):
            self.skipTest("Skip test jit.")
        self.script_module  # noqa: B018

    @classmethod
    def convert_to_numpy(cls, xx: torch.Tensor) -> np.ndarray:
        """Convert to numpy array."""
        return to_numpy_array(xx)

    @classmethod
    def convert_from_numpy(cls, xx: np.ndarray) -> torch.Tensor:
        """Convert from numpy array."""
        return to_torch_tensor(xx)

    def forward_wrapper_cpu_ref(self, module):
        module.to("cpu")
        return self.forward_wrapper(module, on_cpu=True)

    def forward_wrapper(self, module, on_cpu=False):
        def create_wrapper_method(method):
            def wrapper_method(self, *args, **kwargs):  # noqa: ARG001
                # convert to torch tensor
                args = [to_torch_tensor(arg) for arg in args]
                kwargs = {k: to_torch_tensor(v) for k, v in kwargs.items()}
                if on_cpu:
                    args = [
                        arg.detach().cpu() if arg is not None else None for arg in args
                    ]
                    kwargs = {
                        k: v.detach().cpu() if v is not None else None
                        for k, v in kwargs.items()
                    }
                # forward
                output = method(*args, **kwargs)
                # convert to numpy array
                if isinstance(output, tuple):
                    output = tuple(to_numpy_array(o) for o in output)
                elif isinstance(output, dict):
                    output = {k: to_numpy_array(v) for k, v in output.items()}
                else:
                    output = to_numpy_array(output)
                return output

            return wrapper_method

        class WrapperModule:
            __call__ = create_wrapper_method(module.__call__)
            if hasattr(module, "forward_lower"):
                forward_lower = create_wrapper_method(module.forward_lower)

        return WrapperModule()


class ModelTestCase:
    """Common test case for model."""

    module: torch.nn.Module
    """Module to test."""
    modules_to_test: list[torch.nn.Module]
    """Modules to test."""
    expected_type_map: list[str]
    """Expected type map."""
    expected_rcut: float
    """Expected cut-off radius."""
    expected_dim_fparam: int
    """Expected number (dimension) of frame parameters."""
    expected_dim_aparam: int
    """Expected number (dimension) of atomic parameters."""
    expected_sel_type: list[int]
    """Expected selected atom types."""
    expected_aparam_nall: bool
    """Expected shape of atomic parameters."""
    expected_model_output_type: list[str]
    """Expected output type for the model."""
    model_output_equivariant: list[str]
    """Outputs that are equivariant to the input rotation."""
    expected_sel: list[int]
    """Expected number of neighbors."""
    expected_has_message_passing: bool
    """Expected whether having message passing."""
    expected_nmpnn: int
    """Expected number of MPNN."""
    forward_wrapper: ClassVar[Callable[[Any, bool], Any]]
    """Class wrapper for forward method."""
    forward_wrapper_cpu_ref: Callable[[Any], Any]
    """Convert model to CPU method."""
    aprec_dict: dict[str, Optional[float]]
    """Dictionary of absolute precision in each test."""
    rprec_dict: dict[str, Optional[float]]
    """Dictionary of relative precision in each test."""
    epsilon_dict: dict[str, Optional[float]]
    """Dictionary of epsilons in each test."""

    skipTest: Callable[[str], None]  # noqa: N815
    """Skip test method."""
    output_def: dict[str, Any]
    """Output definition."""

    def test_get_type_map(self) -> None:
        """Test get_type_map."""
        for module in self.modules_to_test:
            assert module.get_type_map() == self.expected_type_map

    def test_get_rcut(self) -> None:
        """Test get_rcut."""
        for module in self.modules_to_test:
            assert module.get_rcut() == self.expected_rcut * self.expected_nmpnn

    def test_get_dim_fparam(self) -> None:
        """Test get_dim_fparam."""
        for module in self.modules_to_test:
            assert module.get_dim_fparam() == self.expected_dim_fparam

    def test_get_dim_aparam(self) -> None:
        """Test get_dim_aparam."""
        for module in self.modules_to_test:
            assert module.get_dim_aparam() == self.expected_dim_aparam

    def test_get_sel_type(self) -> None:
        """Test get_sel_type."""
        for module in self.modules_to_test:
            assert module.get_sel_type() == self.expected_sel_type

    def test_is_aparam_nall(self) -> None:
        """Test is_aparam_nall."""
        for module in self.modules_to_test:
            assert module.is_aparam_nall() == self.expected_aparam_nall

    def test_model_output_type(self) -> None:
        """Test model_output_type."""
        for module in self.modules_to_test:
            assert module.model_output_type() == self.expected_model_output_type

    def test_get_nnei(self) -> None:
        """Test get_nnei."""
        expected_nnei = sum(self.expected_sel)
        for module in self.modules_to_test:
            assert module.get_nnei() == expected_nnei

    def test_get_ntypes(self) -> None:
        """Test get_ntypes."""
        for module in self.modules_to_test:
            assert module.get_ntypes() == len(self.expected_type_map)

    def test_has_message_passing(self) -> None:
        """Test has_message_passing."""
        for module in self.modules_to_test:
            assert module.has_message_passing() == self.expected_has_message_passing

    def test_forward(self) -> None:
        """Test forward and forward_lower."""
        test_spin = getattr(self, "test_spin", False)
        nf = 2
        natoms = 5
        aprec = (
            0
            if self.aprec_dict.get("test_forward", None) is None
            else self.aprec_dict["test_forward"]
        )
        rng = np.random.default_rng(GLOBAL_SEED)
        coord = 4.0 * rng.random([1, natoms, 3]).repeat(nf, 0).reshape([nf, -1])
        atype = np.array([[0, 0, 0, 1, 1] * nf], dtype=int).reshape([nf, -1])
        spin = 0.5 * rng.random([1, natoms, 3]).repeat(nf, 0).reshape([nf, -1])
        cell = 6.0 * np.repeat(np.eye(3)[None, ...], nf, axis=0).reshape([nf, 9])
        coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.expected_rcut + 1.0 if test_spin else self.expected_rcut,
            self.expected_sel,
            mixed_types=self.module.mixed_types(),
            box=cell,
        )
        coord_normalized = normalize_coord(
            coord.reshape(nf, natoms, 3),
            cell.reshape(nf, 3, 3),
        )
        coord_ext_large, atype_ext_large, mapping_large = extend_coord_with_ghosts(
            coord_normalized,
            atype,
            cell,
            self.module.get_rcut(),
        )
        nlist_large = build_neighbor_list(
            coord_ext_large,
            atype_ext_large,
            natoms,
            self.expected_rcut,
            self.expected_sel,
            distinguish_types=(not self.module.mixed_types()),
        )
        spin_ext = np.take_along_axis(
            spin.reshape(nf, -1, 3),
            np.repeat(np.expand_dims(mapping, axis=-1), 3, axis=-1),
            axis=1,
        )
        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])
        ret = []
        ret_lower = []
        for _module in self.modules_to_test:
            module = self.forward_wrapper(_module)
            input_dict = {
                "coord": coord,
                "atype": atype,
                "box": cell,
                "aparam": aparam,
                "fparam": fparam,
            }
            if test_spin:
                input_dict["spin"] = spin
            ret.append(module(**input_dict))

            input_dict_lower = {
                "extended_coord": coord_ext_large,
                "extended_atype": atype_ext_large,
                "nlist": nlist_large,
                "aparam": aparam,
                "fparam": fparam,
                "mapping": mapping_large,
            }
            if test_spin:
                input_dict_lower["extended_spin"] = spin_ext

            # use shuffled nlist, simulating the lammps interface
            rng.shuffle(input_dict_lower["nlist"], axis=-1)
            ret_lower.append(module.forward_lower(**input_dict_lower))

            input_dict_lower = {
                "extended_coord": coord_ext_large,
                "extended_atype": atype_ext_large,
                "nlist": nlist_large,
                "aparam": aparam,
                "fparam": fparam,
            }
            if test_spin:
                input_dict_lower["extended_spin"] = spin_ext

            # use shuffled nlist, simulating the lammps interface
            rng.shuffle(input_dict_lower["nlist"], axis=-1)
            ret_lower.append(module.forward_lower(**input_dict_lower))

        for kk in ret[0]:
            # ensure the first frame and the second frame are the same
            np.testing.assert_allclose(
                ret[0][kk][0],
                ret[0][kk][1],
                err_msg=f"compare {kk} between frame 0 and 1",
            )

            subret = [rr[kk] for rr in ret if rr is not None]
            if len(subret):
                for ii, rr in enumerate(subret[1:]):
                    if subret[0] is None:
                        assert rr is None
                    else:
                        np.testing.assert_allclose(
                            subret[0],
                            rr,
                            err_msg=f"compare {kk} between 0 and {ii}",
                        )
        for kk in ret_lower[0]:
            subret = []
            for rr in ret_lower:
                if rr is not None:
                    subret.append(rr[kk])
            if len(subret):
                for ii, rr in enumerate(subret[1:]):
                    if kk == "expanded_force":
                        # use mapping to scatter sum the forces
                        rr = np.take_along_axis(  # noqa: PLW2901
                            rr,
                            np.repeat(
                                np.expand_dims(mapping_large, axis=-1),
                                3,
                                axis=-1,
                            ),
                            axis=1,
                        )
                    if subret[0] is None:
                        assert rr is None
                    else:
                        np.testing.assert_allclose(
                            subret[0],
                            rr,
                            atol=1e-5,
                            err_msg=f"compare {kk} between 0 and {ii}",
                        )
        same_keys = set(ret[0].keys()) & set(ret_lower[0].keys())
        assert same_keys
        for key in same_keys:
            for rr in ret:
                if rr[key] is not None:
                    rr1 = rr[key]
                    break
            else:
                continue
            for rr in ret_lower:
                if rr[key] is not None:
                    rr2 = rr[key]
                    break
            else:
                continue
            np.testing.assert_allclose(rr1, rr2, atol=aprec)

    def test_permutation(self) -> None:
        """Test permutation."""
        if getattr(self, "skip_test_permutation", False):
            self.skipTest("Skip test permutation.")
        test_spin = getattr(self, "test_spin", False)
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1
        aprec = (
            0
            if self.aprec_dict.get("test_permutation", None) is None
            else self.aprec_dict["test_permutation"]
        )
        idx = [0, 1, 2, 3, 4]
        idx_perm = [1, 0, 4, 3, 2]
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        spin = 0.1 * rng.random([natoms, 3])
        atype = np.array([0, 0, 0, 1, 1])
        coord_perm = coord[idx_perm]
        spin_perm = spin[idx_perm]
        atype_perm = atype[idx_perm]

        # reshape for input
        coord = coord.reshape([nf, -1])
        coord_perm = coord_perm.reshape([nf, -1])
        spin_perm = spin_perm.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        atype_perm = atype_perm.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        aparam = None
        fparam = None
        aparam_perm = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
            aparam_perm = aparam[:, idx_perm, :]
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])

        ret = []
        module = self.forward_wrapper(self.module)
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": cell,
            "aparam": aparam,
            "fparam": fparam,
        }
        if test_spin:
            input_dict["spin"] = spin
        ret.append(module(**input_dict))
        # permutation
        input_dict["coord"] = coord_perm
        input_dict["atype"] = atype_perm
        input_dict["aparam"] = aparam_perm
        if test_spin:
            input_dict["spin"] = spin_perm
        ret.append(module(**input_dict))

        for kk in ret[0]:
            if kk in self.output_def:
                if ret[0][kk] is None:
                    assert ret[1][kk] is None
                    continue
                atomic = self.output_def[kk].atomic
                if atomic:
                    np.testing.assert_allclose(
                        ret[0][kk][:, idx_perm],
                        ret[1][kk][:, idx],  # for extended output
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                    )
                else:
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                    )
            else:
                msg = f"Unknown output key: {kk}"
                raise RuntimeError(msg)

    def test_trans(self) -> None:
        """Test translation."""
        if getattr(self, "skip_test_trans", False):
            self.skipTest("Skip test translation.")
        test_spin = getattr(self, "test_spin", False)
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1
        aprec = (
            1e-14
            if self.aprec_dict.get("test_rot", None) is None
            else self.aprec_dict["test_rot"]
        )
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        spin = 0.1 * rng.random([natoms, 3])
        atype = np.array([0, 0, 0, 1, 1])
        shift = (rng.random([3]) - 0.5) * 2.0
        coord_s = np.matmul(
            np.remainder(np.matmul(coord + shift, np.linalg.inv(cell)), 1.0),
            cell,
        )

        # reshape for input
        coord = coord.reshape([nf, -1])
        spin = spin.reshape([nf, -1])
        coord_s = coord_s.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])

        ret = []
        module = self.forward_wrapper(self.module)
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": cell,
            "aparam": aparam,
            "fparam": fparam,
        }
        if test_spin:
            input_dict["spin"] = spin
        ret.append(module(**input_dict))
        # translation
        input_dict["coord"] = coord_s
        ret.append(module(**input_dict))

        for kk in ret[0]:
            if kk in self.output_def:
                if ret[0][kk] is None:
                    assert ret[1][kk] is None
                    continue
                np.testing.assert_allclose(
                    ret[0][kk],
                    ret[1][kk],
                    err_msg=f"compare {kk} before and after transform",
                    atol=aprec,
                )
            else:
                msg = f"Unknown output key: {kk}"
                raise RuntimeError(msg)

    def test_rot(self) -> None:
        """Test rotation."""
        if getattr(self, "skip_test_rot", False):
            self.skipTest("Skip test rotation.")
        test_spin = getattr(self, "test_spin", False)
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1
        aprec = (
            0
            if self.aprec_dict.get("test_rot", None) is None
            else self.aprec_dict["test_rot"]
        )
        # rotate only coord and shift to the center of cell
        cell = 10.0 * np.eye(3)
        coord = 2.0 * rng.random([natoms, 3])
        spin = 0.1 * rng.random([natoms, 3])
        atype = np.array([0, 0, 0, 1, 1])
        shift = np.array([4.0, 4.0, 4.0])
        from scipy.stats import (
            special_ortho_group,
        )

        rmat = special_ortho_group.rvs(3)
        coord_rot = np.matmul(coord, rmat)
        spin_rot = np.matmul(spin, rmat)

        # reshape for input
        coord = (coord + shift).reshape([nf, -1])
        spin = spin.reshape([nf, -1])
        coord_rot = (coord_rot + shift).reshape([nf, -1])
        spin_rot = spin_rot.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])

        ret = []
        module = self.forward_wrapper(self.module)
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": cell,
            "aparam": aparam,
            "fparam": fparam,
        }
        if test_spin:
            input_dict["spin"] = spin
        ret.append(module(**input_dict))
        # rotation
        input_dict["coord"] = coord_rot
        if test_spin:
            input_dict["spin"] = spin_rot
        ret.append(module(**input_dict))

        for kk in ret[0]:
            if kk in self.output_def:
                if ret[0][kk] is None:
                    assert ret[1][kk] is None
                    continue
                rot_equivariant = (
                    check_deriv(self.output_def[kk])
                    or kk in self.model_output_equivariant
                )
                if not rot_equivariant:
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                    )
                else:
                    v_size = self.output_def[kk].size
                    if v_size == 3:
                        rotated_ret_0 = np.matmul(ret[0][kk], rmat)
                        ret_1 = ret[1][kk]
                    elif v_size == 9:
                        ret_0 = ret[0][kk].reshape(-1, 3, 3)
                        batch_rmat_t = np.repeat(
                            rmat.T.reshape(1, 3, 3),
                            ret_0.shape[0],
                            axis=0,
                        )
                        batch_rmat = np.repeat(
                            rmat.reshape(1, 3, 3),
                            ret_0.shape[0],
                            axis=0,
                        )
                        rotated_ret_0 = np.matmul(
                            batch_rmat_t,
                            np.matmul(ret_0, batch_rmat),
                        )
                        ret_1 = ret[1][kk].reshape(-1, 3, 3)
                    else:
                        # unsupported dim
                        continue
                    np.testing.assert_allclose(
                        rotated_ret_0,
                        ret_1,
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                    )
            else:
                msg = f"Unknown output key: {kk}"
                raise RuntimeError(msg)

        # rotate coord and cell
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        spin = 0.1 * rng.random([natoms, 3])
        atype = np.array([0, 0, 0, 1, 1])
        coord_rot = np.matmul(coord, rmat)
        cell_rot = np.matmul(cell, rmat)
        spin_rot = np.matmul(spin, rmat)

        # reshape for input
        coord = coord.reshape([nf, -1])
        spin = spin.reshape([nf, -1])
        coord_rot = coord_rot.reshape([nf, -1])
        spin_rot = spin_rot.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])
        cell_rot = cell_rot.reshape([nf, 9])

        ret = []
        module = self.forward_wrapper(self.module)
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": cell,
            "aparam": aparam,
            "fparam": fparam,
        }
        if test_spin:
            input_dict["spin"] = spin
        ret.append(module(**input_dict))
        # rotation
        input_dict["coord"] = coord_rot
        input_dict["box"] = cell_rot
        if test_spin:
            input_dict["spin"] = spin_rot
        ret.append(module(**input_dict))

        for kk in ret[0]:
            if kk in self.output_def:
                if ret[0][kk] is None:
                    assert ret[1][kk] is None
                    continue
                rot_equivariant = (
                    check_deriv(self.output_def[kk])
                    or kk in self.model_output_equivariant
                )
                if not rot_equivariant:
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[1][kk],
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                    )
                else:
                    v_size = self.output_def[kk].size
                    if v_size == 3:
                        rotated_ret_0 = np.matmul(ret[0][kk], rmat)
                        ret_1 = ret[1][kk]
                    elif v_size == 9:
                        ret_0 = ret[0][kk].reshape(-1, 3, 3)
                        batch_rmat_t = np.repeat(
                            rmat.T.reshape(1, 3, 3),
                            ret_0.shape[0],
                            axis=0,
                        )
                        batch_rmat = np.repeat(
                            rmat.reshape(1, 3, 3),
                            ret_0.shape[0],
                            axis=0,
                        )
                        rotated_ret_0 = np.matmul(
                            batch_rmat_t,
                            np.matmul(ret_0, batch_rmat),
                        )
                        ret_1 = ret[1][kk].reshape(-1, 3, 3)
                    else:
                        # unsupported dim
                        continue
                    np.testing.assert_allclose(
                        rotated_ret_0,
                        ret_1,
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                    )
            else:
                msg = f"Unknown output key: {kk}"
                raise RuntimeError(msg)

    def test_smooth(self) -> None:
        """Test smooth."""
        if getattr(self, "skip_test_smooth", False):
            self.skipTest("Skip test smooth.")
        test_spin = getattr(self, "test_spin", False)
        rng = np.random.default_rng(GLOBAL_SEED)
        epsilon = (
            1e-5
            if self.epsilon_dict.get("test_smooth", None) is None
            else self.epsilon_dict["test_smooth"]
        )
        assert epsilon is not None
        # required prec.
        rprec = (
            1e-5
            if self.rprec_dict.get("test_smooth", None) is None
            else self.rprec_dict["test_smooth"]
        )
        aprec = (
            1e-5
            if self.aprec_dict.get("test_smooth", None) is None
            else self.aprec_dict["test_smooth"]
        )
        natoms = 10
        nf = 1
        cell = 10.0 * np.eye(3)
        atype0 = np.arange(2)
        atype1 = rng.integers(0, 2, size=natoms - 2)
        atype = np.concatenate([atype0, atype1]).reshape(natoms)
        spin = 0.1 * rng.random([natoms, 3])
        coord0 = np.array(
            [
                0.0,
                0.0,
                0.0,
                self.expected_rcut - 0.5 * epsilon,
                0.0,
                0.0,
                0.0,
                self.expected_rcut - 0.5 * epsilon,
                0.0,
            ],
        ).reshape(-1, 3)
        coord1 = rng.random([natoms - coord0.shape[0], 3])
        coord1 = np.matmul(coord1, cell)
        coord = np.concatenate([coord0, coord1], axis=0)

        coord0 = deepcopy(coord)
        coord1 = deepcopy(coord)
        coord1[1][0] += epsilon
        coord2 = deepcopy(coord)
        coord2[2][1] += epsilon
        coord3 = deepcopy(coord)
        coord3[1][0] += epsilon
        coord3[2][1] += epsilon

        # reshape for input
        coord0 = coord0.reshape([nf, -1])
        coord1 = coord1.reshape([nf, -1])
        coord2 = coord2.reshape([nf, -1])
        coord3 = coord3.reshape([nf, -1])
        spin = spin.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])

        ret = []
        module = self.forward_wrapper(self.module)
        input_dict = {"atype": atype, "box": cell, "aparam": aparam, "fparam": fparam}
        if test_spin:
            input_dict["spin"] = spin
        # coord0
        input_dict["coord"] = coord0
        ret.append(module(**input_dict))
        # coord1
        input_dict["coord"] = coord1
        ret.append(module(**input_dict))
        # coord2
        input_dict["coord"] = coord2
        ret.append(module(**input_dict))
        # coord3
        input_dict["coord"] = coord3
        ret.append(module(**input_dict))

        for kk in ret[0]:
            if kk in self.output_def:
                if ret[0][kk] is None:
                    for ii in range(len(ret) - 1):
                        assert ret[ii + 1][kk] is None
                    continue
                for ii in range(len(ret) - 1):
                    np.testing.assert_allclose(
                        ret[0][kk],
                        ret[ii + 1][kk],
                        err_msg=f"compare {kk} before and after transform",
                        atol=aprec,
                        rtol=rprec,
                    )
            else:
                msg = f"Unknown output key: {kk}"
                raise RuntimeError(msg)

    def test_autodiff(self) -> None:
        """Test autodiff."""
        if getattr(self, "skip_test_autodiff", False):
            self.skipTest("Skip test autodiff.")
        test_spin = getattr(self, "test_spin", False)

        places = 4
        delta = 1e-5

        def finite_difference(f, x, delta=1e-6):
            in_shape = x.shape
            y0 = f(x)
            out_shape = y0.shape
            res = np.empty(out_shape + in_shape)
            for idx in np.ndindex(*in_shape):
                diff = np.zeros(in_shape)
                diff[idx] += delta
                y1p = f(x + diff)
                y1n = f(x - diff)
                res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
            return res

        def stretch_box(old_coord, old_box, new_box):
            ocoord = old_coord.reshape(-1, 3)
            obox = old_box.reshape(3, 3)
            nbox = new_box.reshape(3, 3)
            ncoord = ocoord @ np.linalg.inv(obox) @ nbox
            return ncoord.reshape(old_coord.shape)

        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nf = 1
        cell = rng.random([3, 3])
        cell = (cell + cell.T) + 5.0 * np.eye(3)
        coord = rng.random([natoms, 3])
        coord = np.matmul(coord, cell)
        spin = 0.1 * rng.random([natoms, 3])
        atype = np.array([0, 0, 0, 1, 1])

        # reshape for input
        coord = coord.reshape([nf, -1])
        spin = spin.reshape([nf, -1])
        atype = atype.reshape([nf, -1])
        cell = cell.reshape([nf, 9])

        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])

        module = self.forward_wrapper(self.module)

        # only test force and virial for energy model
        def ff_coord(_coord):
            input_dict = {
                "coord": _coord,
                "atype": atype,
                "box": cell,
                "aparam": aparam,
                "fparam": fparam,
            }
            if test_spin:
                input_dict["spin"] = spin
            return module(**input_dict)["energy"]

        def ff_spin(_spin):
            input_dict = {
                "coord": coord,
                "atype": atype,
                "box": cell,
                "aparam": aparam,
                "fparam": fparam,
            }
            if test_spin:
                input_dict["spin"] = _spin
            return module(**input_dict)["energy"]

        fdf = -finite_difference(ff_coord, coord, delta=delta).squeeze()
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": cell,
            "aparam": aparam,
            "fparam": fparam,
        }
        if test_spin:
            input_dict["spin"] = spin
        rff = module(**input_dict)["force"]
        np.testing.assert_almost_equal(
            fdf.reshape(-1, 3),
            rff.reshape(-1, 3),
            decimal=places,
        )

        if test_spin:
            # magnetic force
            fdf = -finite_difference(ff_spin, spin, delta=delta).squeeze()
            rff = module(**input_dict)["force_mag"]
            np.testing.assert_almost_equal(
                fdf.reshape(-1, 3),
                rff.reshape(-1, 3),
                decimal=places,
            )

        if not test_spin:

            def ff_cell(bb):
                input_dict = {
                    "coord": stretch_box(coord, cell, bb),
                    "atype": atype,
                    "box": bb,
                    "aparam": aparam,
                    "fparam": fparam,
                }
                return module(**input_dict)["energy"]

            fdv = (
                -(
                    finite_difference(ff_cell, cell, delta=delta)
                    .reshape(-1, 3, 3)
                    .transpose(0, 2, 1)
                    @ cell.reshape(-1, 3, 3)
                )
                .squeeze()
                .reshape(9)
            )
            input_dict = {
                "coord": stretch_box(coord, cell, cell),
                "atype": atype,
                "box": cell,
                "aparam": aparam,
                "fparam": fparam,
            }
            rfv = module(**input_dict)["virial"]
            np.testing.assert_almost_equal(
                fdv.reshape(-1, 9),
                rfv.reshape(-1, 9),
                decimal=places,
            )
        else:
            # not support virial by far
            pass

    def test_device_consistence(self) -> None:
        """Test forward consistency between devices."""
        test_spin = getattr(self, "test_spin", False)
        nf = 1
        natoms = 5
        rng = np.random.default_rng(GLOBAL_SEED)
        coord = 4.0 * rng.random([natoms, 3]).reshape([nf, -1])
        atype = np.array([0, 0, 0, 1, 1], dtype=int).reshape([nf, -1])
        spin = 0.5 * rng.random([natoms, 3]).reshape([nf, -1])
        cell = 6.0 * np.eye(3).reshape([nf, 9])
        aparam = None
        fparam = None
        if self.module.get_dim_aparam() > 0:
            aparam = rng.random([nf, natoms, self.module.get_dim_aparam()])
        if self.module.get_dim_fparam() > 0:
            fparam = rng.random([nf, self.module.get_dim_fparam()])
        ret = []
        device_module = self.forward_wrapper(self.module)
        ref_module = self.forward_wrapper_cpu_ref(deepcopy(self.module))

        for module in [device_module, ref_module]:
            input_dict = {
                "coord": coord,
                "atype": atype,
                "box": cell,
                "aparam": aparam,
                "fparam": fparam,
            }
            if test_spin:
                input_dict["spin"] = spin
            ret.append(module(**input_dict))
        for kk in ret[0]:
            subret = [rr[kk] for rr in ret if rr is not None]
            if len(subret):
                for ii, rr in enumerate(subret[1:]):
                    if subret[0] is None:
                        assert rr is None
                    else:
                        np.testing.assert_allclose(
                            subret[0],
                            rr,
                            err_msg=f"compare {kk} between 0 and {ii}",
                            atol=1e-10,
                        )


class EnerModelTest(ModelTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_rcut = 5.0
        cls.expected_type_map = ["O", "H"]
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_sel_type = [0, 1]
        cls.expected_aparam_nall = False
        cls.expected_model_output_type = ["energy"]
        cls.model_output_equivariant = []
        cls.expected_sel = [46, 92]
        cls.expected_sel_mix = sum(cls.expected_sel)  # type: ignore[attr-defined]
        cls.expected_has_message_passing = False
        cls.aprec_dict = {}
        cls.rprec_dict = {}
        cls.epsilon_dict = {}


class TestMaceModel(unittest.TestCase, EnerModelTest, PTTestCase):  # type: ignore[misc]
    """Test MACE model."""

    @property
    def modules_to_test(self) -> list[torch.nn.Module]:  # type: ignore[override]
        """Modules to test."""
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)  # type: ignore[attr-defined]
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module,
            ]
        return modules

    _script_module: torch.jit.ScriptModule

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        EnerModelTest.setUpClass()

        torch.manual_seed(GLOBAL_SEED + 1)
        cls.module = MaceModel(
            type_map=cls.expected_type_map,
            sel=138,
            precision="float64",
        )
        with torch.jit.optimized_execution(should_optimize=False):
            cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = False
        cls.expected_sel_type = []
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_nmpnn = 2


class TestNequipModel(unittest.TestCase, EnerModelTest, PTTestCase):  # type: ignore[misc]
    """Test Nequip model."""

    @property
    def modules_to_test(self) -> list[torch.nn.Module]:  # type: ignore[override]
        """Modules to test."""
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)  # type: ignore[attr-defined]
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module,
            ]
        return modules

    _script_module: torch.jit.ScriptModule

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class."""
        EnerModelTest.setUpClass()

        torch.manual_seed(GLOBAL_SEED + 1)
        cls.module = NequipModel(
            type_map=cls.expected_type_map,
            sel=138,
            r_max=cls.expected_rcut,
            num_layers=2,
            precision="float64",
        )
        with torch.jit.optimized_execution(should_optimize=False):
            cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = False
        cls.expected_sel_type = []
        cls.expected_dim_fparam = 0
        cls.expected_dim_aparam = 0
        cls.expected_nmpnn = 2
