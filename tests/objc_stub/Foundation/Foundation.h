// SPDX-License-Identifier: GPL-2.0-only
// SPDX-FileCopyrightText: 2026 Anirban Phukan
// STUB of the tiny part of Foundation used by src/metal/metal_backend.mm, written from memory of Apple's headers. It exists only
// so that `clang -fsyntax-only` on Linux can catch typos and type errors in the Objective-C++ (tests/check_objcxx_syntax.sh).
// It is NOT the real API: passing this check proves nothing about the real SDK.
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef double NSTimeInterval;
typedef double CFTimeInterval;
typedef signed char BOOL;
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#ifndef nil
#define nil ((id)0)
#endif
#define NS_ASSUME_NONNULL_BEGIN
#define NS_ASSUME_NONNULL_END
#define API_AVAILABLE(...)

@protocol NSObject
- (NSUInteger)hash;
@end
@interface NSObject <NSObject>
+ (instancetype)new;
- (instancetype)init;
@end
@interface NSString : NSObject
+ (instancetype)stringWithUTF8String:(const char *)nullTerminatedCString;
- (const char *)UTF8String;
- (NSUInteger)length;
@end
@interface NSError : NSObject
@property(readonly, copy) NSString *localizedDescription;
@end
@interface NSProcessInfo : NSObject
+ (NSProcessInfo *)processInfo;
@property(readonly, copy) NSString *operatingSystemVersionString;
@end
