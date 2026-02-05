
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void metal_init();
void metal_add(const float* a, const float* b, float* result, int size);

#ifdef __cplusplus
}
#endif
