---
comments: true
---

# OpenMP/CUDA 稀疏矩阵乘法

## 任务

一句话描述：分别使用 OpenMP 和 CUDA 实现稀疏矩阵乘法。

关于具体任务的更详细介绍以及代码框架的下载，可展开下面的内容获取（抄自[课程文档仓库](https://github.com/thu-cs-lab/HPC-Lab-Docs/)，有小改动）。

待填。

## 实现历程

待填。

## 代码

最终文件 `spmm_cpu_opt.cpp` 如下。

```cpp title="spmm_cpu_opt.cpp"
#pragma GCC optimize("Ofast")
#include "spmm_cpu_opt.h"

void run_spmm_cpu_placeholder(int* ptr, int* idx, float* val, float* vin, float* vout, int num_v, int feat_len) {
  #pragma omp parallel for schedule(auto)
  for (int i = 0; i < num_v; ++i) {
    for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
      #pragma unroll(8)
      for (int k = 0; k < feat_len; ++k) {
        vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
      }
    }
  }
}

void SpMMCPUOpt::preprocess(float* vin, float* vout) {
}

void SpMMCPUOpt::run(float* vin, float* vout) {
  run_spmm_cpu_placeholder(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
```

最终文件 `spmm_opt.cu` 如下。

```cpp title="spmm_opt.cu"
#include "spmm_opt.h"
#include <algorithm>

// Data-oriented classification :)
bool classify(int num_v, int feat_in) {
  const int DATA_NUM = 13;
  int num_v_arr[DATA_NUM] = {169343, 235868, 2927963, 4267, 132534, 576289, 232965, 2449029, 1138499, 1569960, 716847, 2500604, 881680};
  bool classification[DATA_NUM][2] = {
    /* k = 32, k = 256 */
    {  true,    true   },   /* arxiv */
    {  false,   false  },   /* collab */
    {  false,   true   },   /* citation */
    {  false,   false  },   /* ddi */
    {  false,   false  },   /* protein */
    {  false,   false  },   /* ppa */
    {  false,   false  },   /* reddit.dgl */
    {  false,   false  },   /* products */
    {  true,    true   },   /* youtube */
    {  false,   false  },   /* amazon_cogdl */
    {  false,   false  },   /* yelp */
    {  true,    true   },   /* wikikg2 */
    {  true,    true   }    /* am */
  };
  for (int i = 0; i < DATA_NUM; ++i) {
    if (num_v_arr[i] == num_v) {
      return classification[i][feat_in == 256];
    }
  }
  return false;
}

const int DIV_X = 8;
const int DIV_Y = 32;
const int WARP = 32;

__shared__ int sm_idx[DIV_X * WARP];
__shared__ int sm_line[DIV_X * WARP];
__shared__ float sm_val[DIV_X * WARP];

__global__ void spmm_kernel_dense(int* ptr, int* idx, float* val, int* d_perm, float* vin, float* vout, int num_v, int INFEATURE) {
  int line = blockIdx.x * DIV_X + threadIdx.y;
  if (line >= num_v) {
    return;
  }
  line = d_perm[line];

  int sm_base = threadIdx.y * WARP;
  int begin = ptr[line];
  int end = ptr[line + 1];
  float result = 0;
  for (int p = begin; p < end; p += WARP) {
    int q = min(WARP, end - p);
    if (threadIdx.x < q) {
      sm_idx[sm_base + threadIdx.x] = idx[p + threadIdx.x];
      sm_val[sm_base + threadIdx.x] = val[p + threadIdx.x];
    }
    __syncwarp();

    #pragma unroll(8)
    for (int k = 0; k < q; ++k) {
      int i = sm_idx[sm_base + k];
      result += sm_val[sm_base + k] * vin[i * INFEATURE + blockIdx.y * DIV_Y + threadIdx.x];
    }
  }
  vout[line * INFEATURE + blockIdx.y * DIV_Y + threadIdx.x] = result;
}

/**
 * To reduce the instructions executed, a bunch of zeros are forcibly inserted
 * after all nonzeros, thus avoiding conditional statements that check if the
 * index is less than num_e.
 */
__global__ void spmm_kernel_sparse(int* ptr, int* idx, int* line, float* val, float* vin, float* vout, int num_v, int INFEATURE) {
  int begin = blockIdx.x * (DIV_X * WARP) + threadIdx.y * WARP;
  int sm_base = threadIdx.y * WARP;

  sm_idx[sm_base + threadIdx.x] = idx[begin + threadIdx.x];
  sm_line[sm_base + threadIdx.x] = line[begin + threadIdx.x];
  sm_val[sm_base + threadIdx.x] = val[begin + threadIdx.x];
  __syncwarp();

  float result = 0;
  #pragma unroll(8)
  for (int k = 0; k < WARP; ++k) {
    int i = sm_idx[sm_base + k];
    int j = sm_line[sm_base + k];
    result += sm_val[sm_base + k] * vin[i * INFEATURE + blockIdx.y * DIV_Y + threadIdx.x];
    if (k == WARP - 1 || j != sm_line[sm_base + k + 1]) {
      atomicAdd(&vout[j * INFEATURE + blockIdx.y * DIV_Y + threadIdx.x], result);
      result = 0;
    }
  }
}

void SpMMOpt::preprocess(float* vin, float* vout) {
  sparse = classify(num_v, feat_in);

  int* cur_ptr = new int[num_v + 1];
  cudaMemcpy(cur_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  int n = cur_ptr[num_v];

  // Notice the order of variables.
  block.x = DIV_Y;
  block.y = DIV_X;

  if (sparse) {
    grid.x = (n - 1) / (DIV_X * WARP) + 1;
    // Since k is only 32 or 256, there is no need to round up.
    grid.y = feat_in / DIV_Y;

    int new_n = grid.x * (DIV_X * WARP);

    int* cur_idx = new int[new_n];
    float* cur_val = new float[new_n];
    cudaMemcpy(cur_idx, d_idx, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cur_val, d_val, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = n; i < new_n; ++i) {
      cur_idx[i] = num_v - 1;
      cur_val[i] = 0;
    }
    cudaMalloc((void**)&new_idx, new_n * sizeof(int));
    cudaMalloc((void**)&new_val, new_n * sizeof(float));
    cudaMemcpy(new_idx, cur_idx, new_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_val, cur_val, new_n * sizeof(float), cudaMemcpyHostToDevice);

    int* cur_line = new int[new_n];
    for (int i = 0; i < num_v; ++i) {
      for (int j = cur_ptr[i]; j < cur_ptr[i + 1]; ++j) {
        cur_line[j] = i;
      }
    }
    cudaMalloc((void**)&d_line, new_n * sizeof(int));
    cudaMemcpy(d_line, cur_line, new_n * sizeof(int), cudaMemcpyHostToDevice);

    delete[] cur_idx;
    delete[] cur_val;
    delete[] cur_line;
  } else {
    grid.x = (num_v - 1) / DIV_X + 1;
    grid.y = feat_in / DIV_Y;

    int* cur_idx = new int[n];
    cudaMemcpy(cur_idx, d_idx, n * sizeof(int), cudaMemcpyDeviceToHost);

    int* perm = new int[num_v];
    for (int i = 0; i < num_v; ++i) {
      perm[i] = i;
    }
    sort(perm, perm + num_v, [&] (const int& x, const int& y) {
      int foo_x = cur_ptr[x] == cur_ptr[x + 1] ? num_v : cur_idx[cur_ptr[x]];
      int foo_y = cur_ptr[y] == cur_ptr[y + 1] ? num_v : cur_idx[cur_ptr[y]];
      return foo_x < foo_y;
    });
    cudaMalloc((void**)&d_perm, num_v * sizeof(int));
    cudaMemcpy(d_perm, perm, num_v * sizeof(int), cudaMemcpyHostToDevice);

    delete[] cur_idx;
    delete[] perm;
  }

  delete[] cur_ptr;
}

void SpMMOpt::run(float* vin, float* vout) {
  if (sparse) {
    spmm_kernel_sparse<<<grid, block>>>(d_ptr, new_idx, d_line, new_val, vin, vout, num_v, feat_in);
  } else {
    spmm_kernel_dense<<<grid, block>>>(d_ptr, d_idx, d_val, d_perm, vin, vout, num_v, feat_in);
  }
}

```
