// SPDX-License-Identifier: LGPL-3.0-or-later
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "edge_index_cuda.h"

namespace {

void check_cuda(cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(error));
  }
}

struct DeviceBuffer {
  int64_t* ptr = nullptr;

  explicit DeviceBuffer(int64_t size) {
    if (size > 0) {
      check_cuda(cudaMalloc(&ptr, sizeof(int64_t) * size), "cudaMalloc");
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

__device__ bool keep_edge(const int64_t* __restrict__ nlist,
                          const int64_t* __restrict__ atype,
                          const int64_t* __restrict__ mm,
                          int64_t tid,
                          int64_t nloc,
                          int64_t nnei,
                          int64_t nall,
                          int64_t nmm) {
  const int64_t ff = tid / (nloc * nnei);
  const int64_t local = tid % (nloc * nnei);
  const int64_t ii = local / nnei;
  const int64_t kk = nlist[tid];
  if (kk < 0 || kk >= nall) {
    return false;
  }

  const int64_t global_ii = ff * nall + ii;
  const int64_t global_kk = ff * nall + kk;

  bool in_mm1 = false;
  for (int64_t mm_idx = 0; mm_idx < nmm; ++mm_idx) {
    if (atype[global_ii] == mm[mm_idx]) {
      in_mm1 = true;
      break;
    }
  }

  bool in_mm2 = false;
  for (int64_t mm_idx = 0; mm_idx < nmm; ++mm_idx) {
    if (atype[global_kk] == mm[mm_idx]) {
      in_mm2 = true;
      break;
    }
  }

  return !(in_mm1 && in_mm2);
}

__global__ void mark_edges_kernel(const int64_t* __restrict__ nlist,
                                  const int64_t* __restrict__ atype,
                                  const int64_t* __restrict__ mm,
                                  int64_t* __restrict__ flags,
                                  int64_t total_slots,
                                  int64_t nloc,
                                  int64_t nnei,
                                  int64_t nall,
                                  int64_t nmm) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      static_cast<int64_t>(threadIdx.x);
  if (tid >= total_slots) {
    return;
  }
  flags[tid] = keep_edge(nlist, atype, mm, tid, nloc, nnei, nall, nmm) ? 1 : 0;
}

__global__ void scatter_edges_kernel(const int64_t* __restrict__ nlist,
                                     const int64_t* __restrict__ flags,
                                     const int64_t* __restrict__ prefix,
                                     int64_t* __restrict__ edge_index,
                                     int64_t total_slots,
                                     int64_t nloc,
                                     int64_t nnei,
                                     int64_t nall) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      static_cast<int64_t>(threadIdx.x);
  if (tid >= total_slots || flags[tid] == 0) {
    return;
  }

  const int64_t ff = tid / (nloc * nnei);
  const int64_t local = tid % (nloc * nnei);
  const int64_t ii = local / nnei;
  const int64_t kk = nlist[tid];
  const int64_t out_idx = prefix[tid] - 1;

  edge_index[2 * out_idx] = ff * nall + kk;
  edge_index[2 * out_idx + 1] = ff * nall + ii;
}

}  // namespace

int64_t edge_index_cuda(const int64_t* nlist,
                        const int64_t* atype,
                        const int64_t* mm,
                        int64_t* edge_index,
                        int64_t nf,
                        int64_t nloc,
                        int64_t nnei,
                        int64_t nall,
                        int64_t nmm,
                        int device_index) {
  if (device_index >= 0) {
    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
  }

  const int64_t total_slots = nf * nloc * nnei;
  if (total_slots == 0) {
    return 0;
  }

  DeviceBuffer flags(total_slots);
  DeviceBuffer prefix(total_slots);

  constexpr int threads = 256;
  const auto blocks =
      static_cast<unsigned int>((total_slots + threads - 1) / threads);
  mark_edges_kernel<<<blocks, threads>>>(nlist, atype, mm, flags.ptr,
                                         total_slots, nloc, nnei, nall, nmm);
  check_cuda(cudaGetLastError(), "mark_edges_kernel launch");

  thrust::inclusive_scan(thrust::device, thrust::device_pointer_cast(flags.ptr),
                         thrust::device_pointer_cast(flags.ptr + total_slots),
                         thrust::device_pointer_cast(prefix.ptr));
  check_cuda(cudaGetLastError(), "thrust::inclusive_scan");

  int64_t edge_count = 0;
  check_cuda(cudaMemcpy(&edge_count, prefix.ptr + total_slots - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost),
             "cudaMemcpy edge_count");

  if (edge_count > 0) {
    scatter_edges_kernel<<<blocks, threads>>>(nlist, flags.ptr, prefix.ptr,
                                              edge_index, total_slots, nloc,
                                              nnei, nall);
    check_cuda(cudaGetLastError(), "scatter_edges_kernel launch");
  }
  return edge_count;
}
