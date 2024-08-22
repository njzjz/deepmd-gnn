// SPDX-License-Identifier: LGPL-3.0-or-later
#include <torch/torch.h>

#include <iostream>

torch::Tensor mace_edge_index(const torch::Tensor &nlist_tensor,
                              const torch::Tensor &atype_tensor,
                              const torch::Tensor &mm_tensor)
{
  torch::Tensor nlist_tensor_ = nlist_tensor.cpu().contiguous();
  torch::Tensor atype_tensor_ = atype_tensor.cpu().contiguous();
  torch::Tensor mm_tensor_ = mm_tensor.cpu().contiguous();
  const int64_t nloc = nlist_tensor_.size(0);
  const int64_t nnei = nlist_tensor_.size(1);
  const int64_t nall = atype_tensor_.size(0);
  const int64_t nmm = mm_tensor_.size(0);
  int64_t *nlist = nlist_tensor_.view({-1}).data_ptr<int64_t>();
  int64_t *atype = atype_tensor_.view({-1}).data_ptr<int64_t>();
  int64_t *mm = mm_tensor_.view({-1}).data_ptr<int64_t>();

  std::vector<int64_t> edge_index;
  edge_index.reserve(nloc * nnei * 2);

  for (int64_t ii = 0; ii < nloc; ii++)
  {
    for (int64_t jj = 0; jj < nnei; jj++)
    {
      int64_t idx = ii * nnei + jj;
      int64_t kk = nlist[idx];
      if (kk < 0)
      {
        continue;
      }
      // check if both atype[ii] and atype[kk] are in mm
      bool in_mm1 = false;
      for (int64_t mm_idx = 0; mm_idx < nmm; mm_idx++)
      {
        if (atype[ii] == mm[mm_idx])
        {
          in_mm1 = true;
          break;
        }
      }
      bool in_mm2 = false;
      for (int64_t mm_idx = 0; mm_idx < nmm; mm_idx++)
      {
        if (atype[kk] == mm[mm_idx])
        {
          in_mm2 = true;
          break;
        }
      }
      if (in_mm1 && in_mm2)
      {
        continue;
      }
      // add edge
      edge_index.push_back(kk);
      edge_index.push_back(ii);
    }
  }
  // convert to tensor
  int64_t edge_size = edge_index.size() / 2;
  torch::Tensor edge_index_tensor =
      torch::tensor(edge_index, torch::kInt64).view({edge_size, 2});
  // to nlist_tensor.device
  return edge_index_tensor.to(nlist_tensor.device());
}

TORCH_LIBRARY(deepmd_mace, m) { m.def("mace_edge_index", mace_edge_index); }