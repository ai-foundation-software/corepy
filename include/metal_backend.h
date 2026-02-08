// Metal Backend C Interface
// This header provides C-compatible function declarations for the Metal backend.
// Only compiled and linked on macOS.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Initialization
// ============================================================================

/// Check if Metal is available on this system
bool metal_is_available();

/// Initialize the Metal device, command queue, and compile shaders
void metal_init();

/// Cleanup Metal resources
void metal_cleanup();

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise addition: result[i] = a[i] + b[i]
void metal_add(const float* a, const float* b, float* result, int size);

/// Element-wise subtraction: result[i] = a[i] - b[i]
void metal_sub(const float* a, const float* b, float* result, int size);

/// Element-wise multiplication: result[i] = a[i] * b[i]
void metal_mul(const float* a, const float* b, float* result, int size);

/// Element-wise division: result[i] = a[i] / b[i]
void metal_div(const float* a, const float* b, float* result, int size);

// ============================================================================
// Broadcasting Operations
// ============================================================================

typedef enum {
    METAL_OP_ADD = 0,
    METAL_OP_SUB = 1,
    METAL_OP_MUL = 2,
    METAL_OP_DIV = 3
} MetalOpType;

/// Broadcasted binary operation
/// shape: Output shape (rank elements)
/// stridesA: Input A strides (rank elements)
/// stridesB: Input B strides (rank elements)
/// rank: Number of dimensions
/// size: Total elements in output
/// sizeA: Total elements in input A buffer (for bounds safety)
/// sizeB: Total elements in input B buffer (for bounds safety)
void metal_broadcast_op(MetalOpType op, const float* a, const float* b, float* result,
                        const int* shape, const int* stridesA, const int* stridesB, int rank, int size, int sizeA, int sizeB);

// ============================================================================
// Reduction Operations
// ============================================================================

/// Compute sum of all elements
float metal_sum_f32(const float* data, int size);

/// Compute mean of all elements
float metal_mean_f32(const float* data, int size);

/// Compute maximum of all elements
float metal_max_f32(const float* data, int size);

/// Compute minimum of all elements
float metal_min_f32(const float* data, int size);

// ============================================================================
// Matrix Multiplication
// ============================================================================

/// Matrix multiplication: C = A @ B
/// A is MxK, B is KxN, C is MxN (row-major layout)
void metal_matmul_f32(const float* a, const float* b, float* c, int m, int k, int n);

// ============================================================================
// Transpose Utils
// ============================================================================

/// Transpose matrix: Output = Input^T
/// Input is MxN, Output is NxM
void metal_transpose_f32(const float* in, float* out, int m, int n);

#ifdef __cplusplus
}
#endif
