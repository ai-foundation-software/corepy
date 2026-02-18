
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Existing kernels
int add_one_kernel(int x);

// New Metal Kernels (conditionally include if needed, or just declarations)
#ifdef __APPLE__
void metal_add(const float* a, const float* b, float* result, int size);
#endif

#ifdef __cplusplus
}
#endif
