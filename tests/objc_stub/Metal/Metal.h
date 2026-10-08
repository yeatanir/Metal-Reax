// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// STUB of the part of Metal used by src/metal/metal_backend.mm, written from memory of Apple's headers (see Foundation.h).
#pragma once
#import <Foundation/Foundation.h>

typedef struct { NSUInteger width, height, depth; } MTLSize;
static inline MTLSize MTLSizeMake(NSUInteger w, NSUInteger h, NSUInteger d) { MTLSize s = {w, h, d}; return s; }
typedef NSUInteger MTLResourceOptions;
enum { MTLResourceStorageModeShared = 0 };
enum MTLCommandBufferStatus : NSUInteger {
  MTLCommandBufferStatusNotEnqueued = 0, MTLCommandBufferStatusEnqueued = 1, MTLCommandBufferStatusCommitted = 2,
  MTLCommandBufferStatusScheduled = 3, MTLCommandBufferStatusCompleted = 4, MTLCommandBufferStatusError = 5 };
enum MTLGPUFamily : NSInteger { MTLGPUFamilyApple7 = 1007, MTLGPUFamilyApple8 = 1008, MTLGPUFamilyApple9 = 1009, MTLGPUFamilyMac2 = 2002, MTLGPUFamilyMetal3 = 5001 };
enum MTLMathMode : NSInteger { MTLMathModeSafe = 0, MTLMathModeRelaxed = 1, MTLMathModeFast = 2 };

@interface MTLCompileOptions : NSObject
@property(nonatomic) BOOL fastMathEnabled;
@property(nonatomic) MTLMathMode mathMode;
@end

@protocol MTLBuffer <NSObject>
- (void *)contents;
@property(readonly) NSUInteger length;
@end
@protocol MTLFunction <NSObject>
@property(readonly) NSString *name;
@end
@protocol MTLComputePipelineState <NSObject>
@property(readonly) NSUInteger maxTotalThreadsPerThreadgroup;
@property(readonly) NSUInteger threadExecutionWidth;
@property(readonly) NSUInteger staticThreadgroupMemoryLength;
@end
@protocol MTLLibrary <NSObject>
- (id<MTLFunction>)newFunctionWithName:(NSString *)functionName;
@end
@protocol MTLComputeCommandEncoder <NSObject>
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)threadsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup;
- (void)endEncoding;
@end
@protocol MTLCommandBuffer <NSObject>
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (void)commit;
- (void)waitUntilCompleted;
@property(readonly) MTLCommandBufferStatus status;
@property(readonly) NSError *error;
@property(readonly) CFTimeInterval GPUStartTime;
@property(readonly) CFTimeInterval GPUEndTime;
@end
@protocol MTLCommandQueue <NSObject>
- (id<MTLCommandBuffer>)commandBuffer;
@end
@protocol MTLDevice <NSObject>
@property(readonly) NSString *name;
@property(readonly) MTLSize maxThreadsPerThreadgroup;
@property(readonly, getter=hasUnifiedMemory) BOOL hasUnifiedMemory;
@property(readonly) uint64_t recommendedMaxWorkingSetSize;
@property(readonly) NSUInteger maxBufferLength;
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source options:(MTLCompileOptions *)options error:(NSError **)error;
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction error:(NSError **)error;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
- (BOOL)supportsFamily:(MTLGPUFamily)gpuFamily;
@end
id<MTLDevice> MTLCreateSystemDefaultDevice(void);
