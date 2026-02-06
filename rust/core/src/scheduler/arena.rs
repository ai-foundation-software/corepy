// ============================================================================
// Thread-Local Arena Allocator
// ============================================================================
//
// RESPONSIBILITIES:
// - Provide fast, thread-local memory allocation for temporary tensors
// - Reduce allocator contention in multi-threaded workloads
// - Automatic cleanup when thread exits
//
// DESIGN:
// - Bump allocator: O(1) allocation, batch deallocation
// - Thread-local storage: No synchronization overhead
// - Configurable arena size via COREPY_ARENA_SIZE env var
// - Integration with rayon thread pool
//
// USAGE PATTERN:
//   with_arena(|arena| {
//       let buf = arena.alloc::<f32>(1024);
//       // ... use buffer ...
//   }); // Arena automatically resets

use std::cell::RefCell;
use std::env;

/// Default arena size per thread: 1 MB
const DEFAULT_ARENA_SIZE: usize = 1024 * 1024;

/// Thread-local arena for temporary allocations
///
/// Uses bump allocation: allocations are O(1), all freed at once when arena resets.
/// Perfect for temporary buffers needed during tensor operations.
#[allow(dead_code)]
pub struct ThreadArena {
    #[allow(dead_code)]
    buffer: Vec<u8>,
    offset: usize,
}

impl ThreadArena {
    /// Create a new arena with the specified size
    pub fn new(size: usize) -> Self {
        ThreadArena {
            buffer: vec![0u8; size],
            offset: 0,
        }
    }

    /// Create arena with size from environment variable or default
    pub fn with_default_size() -> Self {
        let size = env::var("COREPY_ARENA_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_ARENA_SIZE);

        Self::new(size)
    }

    /// Allocate bytes from the arena
    ///
    /// Returns raw pointer to allocated memory.
    /// Memory is NOT initialized (for performance).
    ///
    /// # Safety
    /// - Caller must not use pointer after arena reset
    /// - Caller must ensure proper alignment for type T
    #[allow(dead_code)]
    pub unsafe fn alloc_bytes(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        // Align the current offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        let end = aligned_offset + size;
        if end > self.buffer.len() {
            // Arena exhausted
            return None;
        }

        let ptr = self.buffer.as_mut_ptr().add(aligned_offset);
        self.offset = end;
        Some(ptr)
    }

    /// Allocate typed slice from arena
    ///
    /// Returns None if arena doesn't have enough space.
    ///
    /// # Safety
    /// - Returned slice is valid until arena reset
    /// - Memory is uninitialized
    #[allow(dead_code)]
    pub unsafe fn alloc<T>(&mut self, count: usize) -> Option<*mut T> {
        let size = count * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        self.alloc_bytes(size, align).map(|ptr| ptr as *mut T)
    }

    /// Reset the arena, invalidating all previous allocations
    ///
    /// This is O(1) - just resets the offset pointer.
    /// Memory is not cleared for performance.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get current memory usage
    #[allow(dead_code)]
    pub fn used_bytes(&self) -> usize {
        self.offset
    }

    /// Get total arena capacity
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get remaining space
    #[allow(dead_code)]
    pub fn available_bytes(&self) -> usize {
        self.buffer.len() - self.offset
    }
}

// Thread-local storage for arena
thread_local! {
    static ARENA: RefCell<ThreadArena> = RefCell::new(ThreadArena::with_default_size());
}

/// Execute function with access to thread-local arena
///
/// Arena is automatically reset after the function completes.
///
/// # Example
/// ```
/// with_arena(|arena| {
///     let temp_buffer = unsafe { arena.alloc::<f32>(1000) };
///     // ... use buffer ...
/// });
/// // temp_buffer is now invalid
/// ```
pub fn with_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut ThreadArena) -> R,
{
    ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        let result = f(&mut arena);
        arena.reset(); // Auto-cleanup
        result
    })
}

/// Get arena statistics for debugging
#[allow(dead_code)]
pub fn arena_stats() -> (usize, usize, usize) {
    ARENA.with(|arena| {
        let arena = arena.borrow();
        (
            arena.used_bytes(),
            arena.capacity(),
            arena.available_bytes(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic_allocation() {
        let mut arena = ThreadArena::new(1024);

        unsafe {
            let ptr1 = arena.alloc::<f32>(10).expect("allocation failed");
            assert!(!ptr1.is_null());

            let ptr2 = arena.alloc::<f32>(10).expect("allocation failed");
            assert!(!ptr2.is_null());
            assert_ne!(ptr1, ptr2);
        }
    }

    #[test]
    fn test_arena_exhaustion() {
        let mut arena = ThreadArena::new(100);

        unsafe {
            // Allocate almost all space
            let _ptr1 = arena.alloc::<u8>(90).expect("allocation failed");

            // This should fail
            let ptr2 = arena.alloc::<u8>(20);
            assert!(ptr2.is_none());
        }
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = ThreadArena::new(1024);

        unsafe {
            let _ptr1 = arena.alloc::<f32>(100).expect("allocation failed");
            assert!(arena.used_bytes() > 0);

            arena.reset();
            assert_eq!(arena.used_bytes(), 0);

            // Should be able to allocate again
            let _ptr2 = arena.alloc::<f32>(100).expect("allocation failed");
        }
    }

    #[test]
    fn test_with_arena() {
        let result = with_arena(|arena| {
            unsafe {
                let ptr = arena.alloc::<i32>(10).expect("allocation failed");
                // Write some data
                for i in 0..10 {
                    *ptr.add(i) = i as i32;
                }
                42
            }
        });

        assert_eq!(result, 42);

        // Arena should be reset
        let (used, _, _) = arena_stats();
        assert_eq!(used, 0);
    }
}
