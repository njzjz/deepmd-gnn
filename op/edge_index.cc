// SPDX-License-Identifier: LGPL-3.0-or-later
// Target the oldest stable ABI version used by this file's helper APIs.
#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#endif

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#ifdef DEEPMD_GNN_WITH_CUDA
#include "edge_index_cuda.h"
#endif

namespace {
namespace th = torch::headeronly;
namespace ts = torch::stable;

struct EdgeIndexShape {
  int64_t nf = 0;
  int64_t nloc = 0;
  int64_t nnei = 0;
  int64_t nall = 0;
};

EdgeIndexShape validate_shape(const ts::Tensor& nlist_tensor,
                              const ts::Tensor& atype_tensor) {
  EdgeIndexShape shape;
  if (nlist_tensor.dim() == 2) {
    if (atype_tensor.dim() != 1) {
      throw std::invalid_argument("atype_tensor must be 1D");
    }
    shape.nf = 1;
    shape.nloc = nlist_tensor.size(0);
    shape.nnei = nlist_tensor.size(1);
    shape.nall = atype_tensor.size(0);
  } else if (nlist_tensor.dim() == 3) {
    if (atype_tensor.dim() != 2) {
      throw std::invalid_argument("atype_tensor must be 2D");
    }
    shape.nf = nlist_tensor.size(0);
    shape.nloc = nlist_tensor.size(1);
    shape.nnei = nlist_tensor.size(2);
    if (atype_tensor.size(0) != shape.nf) {
      throw std::invalid_argument(
          "atype_tensor must have the same size as nlist_tensor");
    }
    shape.nall = atype_tensor.size(1);
  } else {
    throw std::invalid_argument("nlist_tensor must be 2D or 3D");
  }
  if (shape.nloc > shape.nall) {
    throw std::invalid_argument("nloc must not exceed nall");
  }
  return shape;
}

void validate_mm_tensor(const ts::Tensor& mm_tensor) {
  if (mm_tensor.dim() != 1) {
    throw std::invalid_argument("mm_tensor must be 1D");
  }
}

ts::Tensor to_device_contiguous(const ts::Tensor& tensor,
                                const ts::Device& device) {
  return ts::contiguous(ts::to(tensor, device));
}

ts::Tensor to_cpu_contiguous(const ts::Tensor& tensor) {
  return to_device_contiguous(tensor, ts::Device(th::DeviceType::CPU));
}

bool type_in_mm(const int64_t atom_type, const int64_t* mm, const int64_t nmm) {
  for (int64_t mm_idx = 0; mm_idx < nmm; mm_idx++) {
    if (atom_type == mm[mm_idx]) {
      return true;
    }
  }
  return false;
}

template <typename Fn>
void visit_edges(const EdgeIndexShape& shape,
                 const int64_t* nlist,
                 const int64_t* atype,
                 const int64_t* mm,
                 const int64_t nmm,
                 Fn&& fn) {
  for (int64_t ff = 0; ff < shape.nf; ff++) {
    for (int64_t ii = 0; ii < shape.nloc; ii++) {
      for (int64_t jj = 0; jj < shape.nnei; jj++) {
        const int64_t idx = ff * shape.nloc * shape.nnei + ii * shape.nnei + jj;
        const int64_t kk = nlist[idx];
        const bool valid_neighbor = kk >= 0 && kk < shape.nall;
        // Dense output still needs a safe placeholder for masked-out slots.
        const int64_t safe_kk = valid_neighbor ? kk : 0;
        const int64_t global_kk = ff * shape.nall + safe_kk;
        const int64_t global_ii = ff * shape.nall + ii;
        bool valid = valid_neighbor;
        if (valid && nmm > 0) {
          const bool in_mm1 = type_in_mm(atype[global_ii], mm, nmm);
          const bool in_mm2 = type_in_mm(atype[global_kk], mm, nmm);
          valid = !(in_mm1 && in_mm2);
        }
        fn(idx, global_kk, global_ii, valid);
      }
    }
  }
}

ts::Tensor edge_index_cpu_kernel(const ts::Tensor& nlist_tensor,
                                 const ts::Tensor& atype_tensor,
                                 const ts::Tensor& mm_tensor) {
  ts::Tensor nlist_tensor_ = to_cpu_contiguous(nlist_tensor);
  ts::Tensor atype_tensor_ = to_cpu_contiguous(atype_tensor);
  ts::Tensor mm_tensor_ = to_cpu_contiguous(mm_tensor);

  const EdgeIndexShape shape = validate_shape(nlist_tensor_, atype_tensor_);
  validate_mm_tensor(mm_tensor_);
  const int64_t nmm = mm_tensor_.size(0);
  const int64_t* nlist = nlist_tensor_.const_data_ptr<int64_t>();
  const int64_t* atype = atype_tensor_.const_data_ptr<int64_t>();
  const int64_t* mm = mm_tensor_.const_data_ptr<int64_t>();

  std::vector<int64_t> edge_index;
  edge_index.reserve(shape.nf * shape.nloc * shape.nnei * 2);
  visit_edges(shape, nlist, atype, mm, nmm,
              [&edge_index](const int64_t /*idx*/, const int64_t global_kk,
                            const int64_t global_ii, const bool valid) {
                if (!valid) {
                  return;
                }
                edge_index.push_back(global_kk);
                edge_index.push_back(global_ii);
              });

  const int64_t edge_size = static_cast<int64_t>(edge_index.size() / 2);
  ts::Tensor edge_index_tensor =
      ts::empty({edge_size, 2}, th::ScalarType::Long, th::Layout::Strided,
                ts::Device(th::DeviceType::CPU));
  int64_t* edge_index_data = edge_index_tensor.mutable_data_ptr<int64_t>();
  std::copy(edge_index.begin(), edge_index.end(), edge_index_data);
  return ts::to(edge_index_tensor, nlist_tensor.device());
}

#ifdef DEEPMD_GNN_WITH_CUDA
ts::Tensor edge_index_cuda_kernel(const ts::Tensor& nlist_tensor,
                                  const ts::Tensor& atype_tensor,
                                  const ts::Tensor& mm_tensor) {
  const ts::Device device = nlist_tensor.device();
  ts::Tensor nlist_tensor_ = to_device_contiguous(nlist_tensor, device);
  ts::Tensor atype_tensor_ = to_device_contiguous(atype_tensor, device);
  ts::Tensor mm_tensor_ = to_device_contiguous(mm_tensor, device);

  const EdgeIndexShape shape = validate_shape(nlist_tensor_, atype_tensor_);
  validate_mm_tensor(mm_tensor_);
  const int64_t nmm = mm_tensor_.size(0);
  const int64_t max_edge_size = shape.nf * shape.nloc * shape.nnei;
  ts::Tensor edge_index_tensor = ts::empty(
      {max_edge_size, 2}, th::ScalarType::Long, th::Layout::Strided, device);

  const int64_t edge_size =
      edge_index_cuda(nlist_tensor_.const_data_ptr<int64_t>(),
                      atype_tensor_.const_data_ptr<int64_t>(),
                      mm_tensor_.const_data_ptr<int64_t>(),
                      edge_index_tensor.mutable_data_ptr<int64_t>(), shape.nf,
                      shape.nloc, shape.nnei, shape.nall, nmm, device.index());
  return ts::narrow(edge_index_tensor, 0, 0, edge_size);
}
#endif

ts::Tensor edge_index_kernel(const ts::Tensor& nlist_tensor,
                             const ts::Tensor& atype_tensor,
                             const ts::Tensor& mm_tensor) {
#ifdef DEEPMD_GNN_WITH_CUDA
  if (nlist_tensor.device().is_cuda()) {
    return edge_index_cuda_kernel(nlist_tensor, atype_tensor, mm_tensor);
  }
#endif
  return edge_index_cpu_kernel(nlist_tensor, atype_tensor, mm_tensor);
}

std::tuple<ts::Tensor, ts::Tensor> dense_edge_index_kernel(
    const ts::Tensor& nlist_tensor,
    const ts::Tensor& atype_tensor,
    const ts::Tensor& mm_tensor) {
  ts::Tensor nlist_tensor_ = to_cpu_contiguous(nlist_tensor);
  ts::Tensor atype_tensor_ = to_cpu_contiguous(atype_tensor);
  ts::Tensor mm_tensor_ = to_cpu_contiguous(mm_tensor);

  const EdgeIndexShape shape = validate_shape(nlist_tensor_, atype_tensor_);
  validate_mm_tensor(mm_tensor_);
  const int64_t nmm = mm_tensor_.size(0);
  const int64_t* nlist = nlist_tensor_.const_data_ptr<int64_t>();
  const int64_t* atype = atype_tensor_.const_data_ptr<int64_t>();
  const int64_t* mm = mm_tensor_.const_data_ptr<int64_t>();
  const int64_t dense_edge_size = shape.nf * shape.nloc * shape.nnei;

  ts::Tensor edge_index_tensor =
      ts::empty({dense_edge_size, 2}, th::ScalarType::Long, th::Layout::Strided,
                ts::Device(th::DeviceType::CPU));
  ts::Tensor edge_mask_tensor =
      ts::empty({dense_edge_size}, th::ScalarType::Bool, th::Layout::Strided,
                ts::Device(th::DeviceType::CPU));
  int64_t* edge_index = edge_index_tensor.mutable_data_ptr<int64_t>();
  bool* edge_mask = edge_mask_tensor.mutable_data_ptr<bool>();

  visit_edges(
      shape, nlist, atype, mm, nmm,
      [edge_index, edge_mask](const int64_t idx, const int64_t global_kk,
                              const int64_t global_ii, const bool valid) {
        edge_index[2 * idx] = global_kk;
        edge_index[2 * idx + 1] = global_ii;
        edge_mask[idx] = valid;
      });

  return std::make_tuple(ts::to(edge_index_tensor, nlist_tensor.device()),
                         ts::to(edge_mask_tensor, nlist_tensor.device()));
}
}  // namespace

STABLE_TORCH_LIBRARY(deepmd_gnn, m) {
  m.def("edge_index(Tensor nlist, Tensor atype, Tensor mm) -> Tensor");
  m.def(
      "dense_edge_index(Tensor nlist, Tensor atype, Tensor mm) -> (Tensor, "
      "Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(deepmd_gnn, CompositeExplicitAutograd, m) {
  m.impl("edge_index", TORCH_BOX(edge_index_kernel));
  m.impl("dense_edge_index", TORCH_BOX(dense_edge_index_kernel));
}

// Compatibility with old models frozen by deepmd_mace package.
STABLE_TORCH_LIBRARY(deepmd_mace, m) {
  m.def("mace_edge_index(Tensor nlist, Tensor atype, Tensor mm) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(deepmd_mace, CompositeExplicitAutograd, m) {
  m.impl("mace_edge_index", TORCH_BOX(edge_index_kernel));
}
