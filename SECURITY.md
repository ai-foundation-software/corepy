# Security Policy

## Supported Versions

Currently, the following versions of CorePy are supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| v0.3.0  | :white_check_mark: |
| < v3.0  | :x:                |

## Reporting a Vulnerability

If you discover any security-related issues (such as buffer overflows, out-of-bounds reads, segfaults triggered from Python space, or arbitrary execution from unsafe pointers), please do **NOT** open a public issue. 

Instead, please email **ai.foundation.software@gmail.com**.

We will confirm the issue within 48 hours and release a patch as soon as possible. We prioritize any vulnerability that escapes the rust-native safety bounds to execute malicious code or read arbitrary memory from the host system.

## Memory Safety Guarantees

CorePy bridges Python (`sys`) and Rust. Despite the use of `unsafe` FFI bindings and raw memory pointers:
- The allocator strictly manages memory limits.
- The lazy compiler performs dimension bounds-checking prior to compute.
- Buffer pools strictly track references using ARC (Atomic Reference Counting) to prevent use-after-free and double-free errors.
