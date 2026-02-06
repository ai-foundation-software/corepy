
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <iostream>
#include "metal_backend.h"

// ============================================================================
// Metal State Management
// ============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLLibrary> g_library = nil;

// Pipeline states for each kernel
static id<MTLComputePipelineState> g_addPipeline = nil;
static id<MTLComputePipelineState> g_subPipeline = nil;
static id<MTLComputePipelineState> g_mulPipeline = nil;
static id<MTLComputePipelineState> g_divPipeline = nil;
static id<MTLComputePipelineState> g_sumPipeline = nil;
static id<MTLComputePipelineState> g_maxPipeline = nil;
static id<MTLComputePipelineState> g_minPipeline = nil;
static id<MTLComputePipelineState> g_matmulNaivePipeline = nil;
static id<MTLComputePipelineState> g_matmulTiledPipeline = nil;

// Thread group size for reductions
static const NSUInteger REDUCTION_THREADGROUP_SIZE = 256;
static const NSUInteger MATMUL_TILE_SIZE = 16;

extern "C" {

// ============================================================================
// Initialization
// ============================================================================

bool metal_is_available() {
    if (g_device != nil) return true;
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil) {
        // Don't keep ref, just check availability
        return true;
    }
    return false;
}

static id<MTLComputePipelineState> createPipeline(NSString* functionName) {
    if (g_library == nil) return nil;
    
    NSError* error = nil;
    id<MTLFunction> func = [g_library newFunctionWithName:functionName];
    if (func == nil) {
        std::cerr << "Metal: Failed to find function '" << [functionName UTF8String] << "'" << std::endl;
        return nil;
    }
    
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
    if (pipeline == nil) {
        std::cerr << "Metal: Failed to create pipeline for '" << [functionName UTF8String] 
                  << "': " << [[error localizedDescription] UTF8String] << std::endl;
    }
    return pipeline;
}

void metal_init() {
    if (g_device != nil) return; // Already initialized
    
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device == nil) {
            std::cerr << "Metal: Failed to find Metal device!" << std::endl;
            return;
        }
        
        g_commandQueue = [g_device newCommandQueue];
        if (g_commandQueue == nil) {
            std::cerr << "Metal: Failed to create command queue!" << std::endl;
            return;
        }
        
        NSError* error = nil;
        
        // 1. Try Environment Variable Override (Best for CI/Dev)
        if (g_library == nil) {
            const char* envPath = std::getenv("COREPY_METAL_LIB_PATH");
            if (envPath != nullptr) {
                // std::cout << "Metal: Loading library from env: " << envPath << std::endl;
                NSString* path = [NSString stringWithUTF8String:envPath];
                NSURL* url = [NSURL fileURLWithPath:path];
                g_library = [g_device newLibraryWithFile:[path stringByExpandingTildeInPath] error:&error];
                
                if (g_library == nil) {
                     std::cerr << "Metal: Failed to load library from COREPY_METAL_LIB_PATH='" 
                               << envPath << "': " 
                               << [[error localizedDescription] UTF8String] << std::endl;
                }
            }
        }

        // 2. Try default library (application bundle)
        if (g_library == nil) {
            g_library = [g_device newDefaultLibrary];
        }
        
        // 3. Fallback: Source compilation (Development)
        if (g_library == nil) {
            // ... (existing bundle fallback, likely won't work in Python ext but kept)
            NSString* metalPath = [[NSBundle mainBundle] pathForResource:@"kernels" ofType:@"metal"];
            if (metalPath != nil) {
                NSString* source = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
                if (source != nil) {
                    g_library = [g_device newLibraryWithSource:source options:nil error:&error];
                }
            }
            
            if (g_library == nil) {
                std::cerr << "Metal: Failed to load library. Ensure .metallib is built and COREPY_METAL_LIB_PATH is set." << std::endl;
                return;
            }
        }
        
        // Create all pipeline states
        g_addPipeline = createPipeline(@"add_arrays");
        g_subPipeline = createPipeline(@"sub_arrays");
        g_mulPipeline = createPipeline(@"mul_arrays");
        g_divPipeline = createPipeline(@"div_arrays");
        g_sumPipeline = createPipeline(@"sum_reduce_simple");
        g_maxPipeline = createPipeline(@"max_reduce_simple");
        g_minPipeline = createPipeline(@"min_reduce_simple");
        g_matmulNaivePipeline = createPipeline(@"matmul_naive");
        g_matmulTiledPipeline = createPipeline(@"matmul_tiled");
    }
}

void metal_cleanup() {
    g_addPipeline = nil;
    g_subPipeline = nil;
    g_mulPipeline = nil;
    g_divPipeline = nil;
    g_sumPipeline = nil;
    g_maxPipeline = nil;
    g_minPipeline = nil;
    g_matmulNaivePipeline = nil;
    g_matmulTiledPipeline = nil;
    g_library = nil;
    g_commandQueue = nil;
    g_device = nil;
}

// ============================================================================
// Element-wise Operations
// ============================================================================

void metal_add(const float* a, const float* b, float* result, int size) {
    if (g_device == nil) metal_init();
    if (g_addPipeline == nil) return;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_addPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(g_addPipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)size);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        memcpy(result, [bufferC contents], bufferSize);
    }
}

void metal_sub(const float* a, const float* b, float* result, int size) {
    if (g_device == nil) metal_init();
    if (g_subPipeline == nil) return;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_subPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(g_subPipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)size);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        memcpy(result, [bufferC contents], bufferSize);
    }
}

void metal_mul(const float* a, const float* b, float* result, int size) {
    if (g_device == nil) metal_init();
    if (g_mulPipeline == nil) return;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_mulPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(g_mulPipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)size);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        memcpy(result, [bufferC contents], bufferSize);
    }
}

void metal_div(const float* a, const float* b, float* result, int size) {
    if (g_device == nil) metal_init();
    if (g_divPipeline == nil) return;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_divPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(g_divPipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)size);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        memcpy(result, [bufferC contents], bufferSize);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

float metal_sum_f32(const float* data, int size) {
    if (g_device == nil) metal_init();
    if (g_sumPipeline == nil) return 0.0f;
    if (size <= 0) return 0.0f;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        NSUInteger count = (NSUInteger)size;
        
        id<MTLBuffer> inputBuffer = [g_device newBufferWithBytes:data length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [g_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> countBuffer = [g_device newBufferWithBytes:&count length:sizeof(uint) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_sumPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        
        // Allocate threadgroup memory for reduction
        NSUInteger threadGroupSize = MIN(REDUCTION_THREADGROUP_SIZE, g_sumPipeline.maxTotalThreadsPerThreadgroup);
        [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float) atIndex:0];
        
        // Single threadgroup for simple reduction
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        float result = ((float*)[outputBuffer contents])[0];
        return result;
    }
}

float metal_mean_f32(const float* data, int size) {
    if (size <= 0) return 0.0f;
    float sum = metal_sum_f32(data, size);
    return sum / (float)size;
}

float metal_max_f32(const float* data, int size) {
    if (g_device == nil) metal_init();
    if (g_maxPipeline == nil) return 0.0f;
    if (size <= 0) return 0.0f;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        NSUInteger count = (NSUInteger)size;
        
        id<MTLBuffer> inputBuffer = [g_device newBufferWithBytes:data length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [g_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> countBuffer = [g_device newBufferWithBytes:&count length:sizeof(uint) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_maxPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        
        NSUInteger threadGroupSize = MIN(REDUCTION_THREADGROUP_SIZE, g_maxPipeline.maxTotalThreadsPerThreadgroup);
        [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float) atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        return ((float*)[outputBuffer contents])[0];
    }
}

float metal_min_f32(const float* data, int size) {
    if (g_device == nil) metal_init();
    if (g_minPipeline == nil) return 0.0f;
    if (size <= 0) return 0.0f;
    
    @autoreleasepool {
        NSUInteger bufferSize = size * sizeof(float);
        NSUInteger count = (NSUInteger)size;
        
        id<MTLBuffer> inputBuffer = [g_device newBufferWithBytes:data length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [g_device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> countBuffer = [g_device newBufferWithBytes:&count length:sizeof(uint) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:g_minPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        
        NSUInteger threadGroupSize = MIN(REDUCTION_THREADGROUP_SIZE, g_minPipeline.maxTotalThreadsPerThreadgroup);
        [encoder setThreadgroupMemoryLength:threadGroupSize * sizeof(float) atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        return ((float*)[outputBuffer contents])[0];
    }
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

void metal_matmul_f32(const float* a, const float* b, float* c, int m, int k, int n) {
    if (g_device == nil) metal_init();
    
    // Use tiled for larger matrices, naive for small ones
    id<MTLComputePipelineState> pipeline = (m >= 32 && n >= 32) ? g_matmulTiledPipeline : g_matmulNaivePipeline;
    if (pipeline == nil) pipeline = g_matmulNaivePipeline;
    if (pipeline == nil) return;
    
    @autoreleasepool {
        NSUInteger sizeA = m * k * sizeof(float);
        NSUInteger sizeB = k * n * sizeof(float);
        NSUInteger sizeC = m * n * sizeof(float);
        
        id<MTLBuffer> bufferA = [g_device newBufferWithBytes:a length:sizeA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [g_device newBufferWithBytes:b length:sizeB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [g_device newBufferWithLength:sizeC options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuf = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBytes:&m length:sizeof(int) atIndex:3];
        [encoder setBytes:&k length:sizeof(int) atIndex:4];
        [encoder setBytes:&n length:sizeof(int) atIndex:5];
        
        if (pipeline == g_matmulTiledPipeline) {
            // Tiled kernel needs threadgroup memory
            NSUInteger tileMemSize = MATMUL_TILE_SIZE * MATMUL_TILE_SIZE * sizeof(float);
            [encoder setThreadgroupMemoryLength:tileMemSize atIndex:0];
            [encoder setThreadgroupMemoryLength:tileMemSize atIndex:1];
            
            MTLSize gridSize = MTLSizeMake((n + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
                                           (m + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE, 1);
            MTLSize threadgroupSize = MTLSizeMake(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
            
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        } else {
            // Naive kernel
            MTLSize gridSize = MTLSizeMake(n, m, 1);
            NSUInteger tgSize = MIN(16, pipeline.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroupSize = MTLSizeMake(tgSize, tgSize, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }
        
        [encoder endEncoding];
        
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        memcpy(c, [bufferC contents], sizeC);
    }
}

} // extern "C"
