
// Metal Shader Kernels for Corepy
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Element-wise Operations
// ============================================================================

kernel void add_arrays(device const float* inA [[ buffer(0) ]],
                       device const float* inB [[ buffer(1) ]],
                       device float* result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = inA[index] + inB[index];
}

kernel void sub_arrays(device const float* inA [[ buffer(0) ]],
                       device const float* inB [[ buffer(1) ]],
                       device float* result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = inA[index] - inB[index];
}

kernel void mul_arrays(device const float* inA [[ buffer(0) ]],
                       device const float* inB [[ buffer(1) ]],
                       device float* result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = inA[index] * inB[index];
}

kernel void div_arrays(device const float* inA [[ buffer(0) ]],
                       device const float* inB [[ buffer(1) ]],
                       device float* result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = inA[index] / inB[index];
}

// Broadcasting Helper
inline uint get_broadcast_index(uint linear_index, constant int* shape, constant int* strides, int rank) {
    uint offset = 0;
    for (int i = rank - 1; i >= 0; --i) {
        int dim_size = shape[i];
        int coord = linear_index % dim_size;
        linear_index /= dim_size;
        offset += coord * strides[i];
    }
    return offset;
}

#define BROADCAST_KERNEL(NAME, OP) \
kernel void NAME(device const float* inA [[ buffer(0) ]], \
                 device const float* inB [[ buffer(1) ]], \
                 device float* result [[ buffer(2) ]], \
                 constant int* shape [[ buffer(3) ]], \
                 constant int* stridesA [[ buffer(4) ]], \
                 constant int* stridesB [[ buffer(5) ]], \
                 constant int& rank [[ buffer(6) ]], \
                 uint index [[ thread_position_in_grid ]]) \
{ \
    uint idxA = get_broadcast_index(index, shape, stridesA, rank); \
    uint idxB = get_broadcast_index(index, shape, stridesB, rank); \
    result[index] = inA[idxA] OP inB[idxB]; \
}

BROADCAST_KERNEL(add_broadcast, +)
BROADCAST_KERNEL(sub_broadcast, -)
BROADCAST_KERNEL(mul_broadcast, *)
BROADCAST_KERNEL(div_broadcast, /)

kernel void fill_array(device float* result [[ buffer(0) ]],
                       constant float& value [[ buffer(1) ]],
                       uint index [[ thread_position_in_grid ]])
{
    result[index] = value;
}

// ============================================================================
// Reduction Operations  
// ============================================================================

// Two-pass reduction: each threadgroup reduces to a partial sum, then CPU sums partials
// For simplicity, using atomic operations for small arrays

kernel void sum_reduce_partial(device const float* input [[ buffer(0) ]],
                               device float* partials [[ buffer(1) ]],
                               constant uint& count [[ buffer(2) ]],
                               uint tid [[ thread_position_in_threadgroup ]],
                               uint gid [[ threadgroup_position_in_grid ]],
                               uint blockDim [[ threads_per_threadgroup ]],
                               threadgroup float* shared [[ threadgroup(0) ]])
{
    // Each thread loads one element
    uint idx = gid * blockDim + tid;
    shared[tid] = (idx < count) ? input[idx] : 0.0f;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 writes partial sum
    if (tid == 0) {
        partials[gid] = shared[0];
    }
}

// Simple sum for small arrays (single threadgroup)
kernel void sum_reduce_simple(device const float* input [[ buffer(0) ]],
                              device float* output [[ buffer(1) ]],
                              constant uint& count [[ buffer(2) ]],
                              uint tid [[ thread_index_in_threadgroup ]],
                              uint blockDim [[ threads_per_threadgroup ]],
                              threadgroup float* shared [[ threadgroup(0) ]])
{
    // Each thread sums multiple elements if needed
    float sum = 0.0f;
    for (uint i = tid; i < count; i += blockDim) {
        sum += input[i];
    }
    shared[tid] = sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[0] = shared[0];
    }
}

kernel void max_reduce_simple(device const float* input [[ buffer(0) ]],
                              device float* output [[ buffer(1) ]],
                              constant uint& count [[ buffer(2) ]],
                              uint tid [[ thread_index_in_threadgroup ]],
                              uint blockDim [[ threads_per_threadgroup ]],
                              threadgroup float* shared [[ threadgroup(0) ]])
{
    float maxVal = -INFINITY;
    for (uint i = tid; i < count; i += blockDim) {
        maxVal = max(maxVal, input[i]);
    }
    shared[tid] = maxVal;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[0] = shared[0];
    }
}

kernel void min_reduce_simple(device const float* input [[ buffer(0) ]],
                              device float* output [[ buffer(1) ]],
                              constant uint& count [[ buffer(2) ]],
                              uint tid [[ thread_index_in_threadgroup ]],
                              uint blockDim [[ threads_per_threadgroup ]],
                              threadgroup float* shared [[ threadgroup(0) ]])
{
    float minVal = INFINITY;
    for (uint i = tid; i < count; i += blockDim) {
        minVal = min(minVal, input[i]);
    }
    shared[tid] = minVal;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[0] = shared[0];
    }
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

// Naive matmul: C[i,j] = sum_k(A[i,k] * B[k,j])
// For production, should use tiled approach with shared memory
kernel void matmul_naive(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant int& M [[ buffer(3) ]],  // Rows of A/C
                         constant int& K [[ buffer(4) ]],  // Cols of A, Rows of B
                         constant int& N [[ buffer(5) ]],  // Cols of B/C
                         uint2 gid [[ thread_position_in_grid ]])
{
    int row = gid.y;
    int col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Tiled matmul for better cache utilization (TILE_SIZE x TILE_SIZE tiles)
#define TILE_SIZE 16

kernel void matmul_tiled(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant int& M [[ buffer(3) ]],
                         constant int& K [[ buffer(4) ]],
                         constant int& N [[ buffer(5) ]],
                         uint2 gid [[ thread_position_in_grid ]],
                         uint2 tid [[ thread_position_in_threadgroup ]],
                         uint2 tgid [[ threadgroup_position_in_grid ]],
                         threadgroup float* tileA [[ threadgroup(0) ]],
                         threadgroup float* tileB [[ threadgroup(1) ]])
{
    int row = tgid.y * TILE_SIZE + tid.y;
    int col = tgid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A
        int aCol = t * TILE_SIZE + tid.x;
        if (row < M && aCol < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[row * K + aCol];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }
        
        // Load tile of B
        int bRow = t * TILE_SIZE + tid.y;
        if (bRow < K && col < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[bRow * N + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Transpose
// ============================================================================

// Vectorized Element-wise Operations (float4)
kernel void add_arrays_float4(device const float4* inA [[ buffer(0) ]],
                              device const float4* inB [[ buffer(1) ]],
                              device float4* result [[ buffer(2) ]],
                              uint index [[ thread_position_in_grid ]])
{
    result[index] = inA[index] + inB[index];
}

kernel void transpose_naive(device const float* in [[ buffer(0) ]],
                            device float* out [[ buffer(1) ]],
                            constant uint& M [[ buffer(2) ]], // Rows of Input
                            constant uint& N [[ buffer(3) ]], // Cols of Input
                            uint2 gid [[ thread_position_in_grid ]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    out[col * M + row] = in[row * N + col];
}

// Tiled Transpose
#define TRANSPOSE_TILE_SIZE 32
#define TRANSPOSE_BLOCK_ROWS 8

kernel void transpose_tiled(device const float* in [[ buffer(0) ]],
                            device float* out [[ buffer(1) ]],
                            constant uint& M [[ buffer(2) ]],
                            constant uint& N [[ buffer(3) ]],
                            uint2 gid [[ thread_position_in_grid ]],
                            uint2 tid [[ thread_position_in_threadgroup ]],
                            uint2 tgid [[ threadgroup_position_in_grid ]],
                            threadgroup float* tile [[ threadgroup(0) ]])
{
    // Handle tile loading
    uint x = tgid.x * TRANSPOSE_TILE_SIZE + tid.x;
    uint y = tgid.y * TRANSPOSE_TILE_SIZE + tid.y;
    uint index_in = y * N + x;
    
    // Load data into shared memory
    // Handle bounds
    if (x < N && y < M) {
        tile[tid.y * TRANSPOSE_TILE_SIZE + tid.x] = in[index_in];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write out transposed
    // Transpose tile logical coordinates: x becomes y, y becomes x
    x = tgid.y * TRANSPOSE_TILE_SIZE + tid.x;
    y = tgid.x * TRANSPOSE_TILE_SIZE + tid.y;
    
    uint index_out = y * M + x;
    
    if (x < M && y < N) {
        out[index_out] = tile[tid.x * TRANSPOSE_TILE_SIZE + tid.y];
    }
}
