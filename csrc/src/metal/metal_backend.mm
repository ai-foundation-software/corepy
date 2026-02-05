
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <iostream>
#include "metal_backend.h"

// Managed Metal State
id<MTLDevice> device = nil;
id<MTLCommandQueue> commandQueue = nil;
id<MTLComputePipelineState> addPipelineState = nil;

extern "C" {

void metal_init() {
    if (device != nil) return; // Already initialized

    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        std::cerr << "Failed to find Metal device!" << std::endl;
        return;
    }
    
    commandQueue = [device newCommandQueue];
    if (commandQueue == nil) {
        std::cerr << "Failed to create Metal command queue!" << std::endl;
        return;
    }

    // Load Default Library
    // In a real Python extension, we might need to load from a string or file path
    // For now, assuming embedded default library or runtime compilation
    NSError* error = nil;
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    
    if (defaultLibrary == nil) {
        // Fallback: This often happens in extensions. We might need to compile source at runtime.
        // For this basic implementation, we just print error.
        std::cerr << "Failed to load default Metal library. Make sure .metallib is built." << std::endl;
        return;
    }

    id<MTLFunction> addFunction = [defaultLibrary newFunctionWithName:@"add_arrays"];
    if (addFunction == nil) {
        std::cerr << "Failed to find 'add_arrays' function!" << std::endl;
        return;
    }

    addPipelineState = [device newComputePipelineStateWithFunction:addFunction error:&error];
    if (addPipelineState == nil) {
        std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }
}

void metal_add(const float* a, const float* b, float* result, int size) {
    if (device == nil) metal_init();
    if (addPipelineState == nil) {
        std::cerr << "Metal not initialized properly, skipping dispatch." << std::endl;
        return;
    }

    // Create buffers
    // In production, these should be cached/managed tensors, not new allocations every time!
    NSUInteger bufferSize = size * sizeof(float);
    id<MTLBuffer> bufferA = [device newBufferWithBytes:a length:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b length:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferRes = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:addPipelineState];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferRes offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    NSUInteger threadGroupSize = addPipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > size) threadGroupSize = size;
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy back result
    memcpy(result, [bufferRes contents], bufferSize);
}

}
