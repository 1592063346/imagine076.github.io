---
comments: true
---

# CUDA 全源最短路

## 任务

一句话描述：使用 CUDA 实现基于分块的全源最短路（Floyd-Warshall 算法）。

关于分块版本的 Floyd-Warshall 算法与具体任务的更详细介绍以及代码框架的下载，可展开下面的内容获取（抄自[课程文档仓库](https://github.com/thu-cs-lab/HPC-Lab-Docs/)，有小改动）。

待填。

## 实现历程

待填。

## 代码

最终文件 `apsp.cu` 如下。

```cpp title="apsp.cu"
#include "apsp.h"

const int MAXN = 1 << 5;
const int EXTEND = 6;
const int DIV = 2; // Registers per thread uses will be more than 64 if I set DIV = 3.

namespace {
__shared__ int foo[MAXN * EXTEND][MAXN];
__shared__ int bar[MAXN][MAXN * EXTEND];

__global__ void step1(int n, int p, int* graph) {
  int x = p * MAXN + threadIdx.y;
  int y = p * MAXN + threadIdx.x;
  if (x < n && y < n) {
    foo[threadIdx.y][threadIdx.x] = graph[x * n + y];
  }
  __syncthreads();

  int m = min(n, (p + 1) * MAXN) - p * MAXN;
  int result;
  if (x < n && y < n) {
    result = graph[x * n + y];
  }
  for (int k = 0; k < m; ++k) {
    if (x < n && y < n) {
      result = min(result, foo[threadIdx.y][k] + foo[k][threadIdx.x]);
      foo[threadIdx.y][threadIdx.x] = result;
    }
    __syncthreads();
  }
  if (x < n && y < n) {
    graph[x * n + y] = result;
  }
}

/**
 * The following naive implementation of step-2 only runs around 7ms at n = 10000,
 * and its performance went worse after I eliminated all bank conflicts.
 * Therefore I'm not going to change it.
 */

/**
 * tag:
 *   0: reset foo
 *   1: reset bar
 */
__device__ void rewrite(int n, int begin_x, int step_x, int begin_y, int step_y, int* graph, int tag) {
  for (int i = 0; i < step_x; ++i) {
    for (int j = 0; j < step_y; ++j) {
      int cur_x = threadIdx.y * step_x + i;
      int cur_y = threadIdx.x * step_y + j;
      int x = begin_x + cur_x;
      int y = begin_y + cur_y;
      if (x < n && y < n) {
        if (tag == 0) {
          foo[cur_x][cur_y] = graph[x * n + y];
        } else {
          bar[cur_x][cur_y] = graph[x * n + y];
        }
      }
    }
  }
}

__global__ void step2(int n, int p, int* graph) {
  int begin_x = blockIdx.y == 0 ? p * MAXN : blockIdx.x * MAXN * EXTEND;
  int begin_y = blockIdx.y == 1 ? p * MAXN : blockIdx.x * MAXN * EXTEND;
  int step_x = blockIdx.y == 0 ? 1 : EXTEND;
  int step_y = blockIdx.y == 1 ? 1 : EXTEND;
  rewrite(n, begin_x, step_x, p * MAXN, 1, graph, 0);
  rewrite(n, p * MAXN, 1, begin_y, step_y, graph, 1);
  __syncthreads();

  for (int i = 0; i < step_x; ++i) {
    for (int j = 0; j < step_y; ++j) {
      int cur_x = threadIdx.y * step_x + i;
      int cur_y = threadIdx.x * step_y + j;
      int x = begin_x + cur_x;
      int y = begin_y + cur_y;
      if (x < n && y < n) {
        int result = graph[x * n + y];
        int m = min(n, (p + 1) * MAXN) - p * MAXN;
        for (int k = 0; k < m; ++k) {
          result = min(result, foo[cur_x][k] + bar[k][cur_y]);
        }
        graph[x * n + y] = result;
      }
    }
  }
}

/**
 * Step-3 is what I should optimize mainly.
 */

__global__ void step3(int n, int p, int* graph) {
  int begin_x = blockIdx.y * MAXN * EXTEND;
  int begin_y = blockIdx.x * MAXN * EXTEND;
  for (int i = 0; i < EXTEND; ++i) {
    int cur_x = i * MAXN + threadIdx.y;
    int cur_y = threadIdx.x;
    int x = begin_x + cur_x;
    int y = p * MAXN + cur_y;
    if (x < n && y < n) {
      foo[cur_x][cur_y] = graph[x * n + y];
    }
  }
  for (int i = 0; i < EXTEND; ++i) {
    int cur_x = threadIdx.y;
    int cur_y = i * MAXN + threadIdx.x;
    int x = p * MAXN + cur_x;
    int y = begin_y + cur_y;
    if (x < n && y < n) {
      bar[cur_x][cur_y] = graph[x * n + y];
    }
  }
  __syncthreads();

  for (int cur = 0; cur < (EXTEND / DIV); ++cur) {
    int result[DIV][EXTEND];
    for (int i = 0; i < DIV; ++i) {
      for (int j = 0; j < EXTEND; ++j) {
        int x = begin_x + (cur * DIV + i) * MAXN + threadIdx.y;
        int y = begin_y + j * MAXN + threadIdx.x;
        if (x < n && y < n) {
          result[i][j] = graph[x * n + y];
        }
      }
    }
    int m = min(n, (p + 1) * MAXN) - p * MAXN;
    #pragma unroll(16)
    for (int k = 0; k < m; ++k) {
      for (int i = 0; i < DIV; ++i) {
        for (int j = 0; j < EXTEND; ++j) {
          int cur_x = (cur * DIV + i) * MAXN + threadIdx.y;
          int cur_y = j * MAXN + threadIdx.x;
          result[i][j] = min(result[i][j], foo[cur_x][k] + bar[k][cur_y]);
        }
      }
    }
    for (int i = 0; i < DIV; ++i) {
      for (int j = 0; j < EXTEND; ++j) {
        int x = begin_x + (cur * DIV + i) * MAXN + threadIdx.y;
        int y = begin_y + j * MAXN + threadIdx.x;
        if (x < n && y < n) {
          graph[x * n + y] = result[i][j];
        }
      }
    }
  }
}
}

void apsp(int n, /* device */ int* graph) {
  int block_n = (n - 1) / (MAXN * EXTEND) + 1;
  for (int p = 0; p * MAXN < n; ++p) {
    dim3 threads(MAXN, MAXN);

    // Step 1
    step1<<<1, threads>>>(n, p, graph);
    cudaDeviceSynchronize();

    // Step 2
    dim3 blocks_2(block_n, 2);
    step2<<<block_n, threads>>>(n, p, graph);
    cudaDeviceSynchronize();

    // Step 3
    dim3 blocks_3(block_n, block_n);
    step3<<<blocks_3, threads>>>(n, p, graph);
    cudaDeviceSynchronize();
  }
}
```
