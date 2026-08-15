use std::alloc::{self, Layout};
use std::ptr;

const ALIGNMENT: usize = 64; // Cache-line + AVX-512 friendly

/// A raw buffer with guaranteed 64-byte alignment.
/// This is the memory backbone of CoreArray.
#[derive(Debug)]
pub struct AlignedBuffer {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBuffer {
    /// Allocate `size` bytes with 64-byte alignment, zero-initialized.
    pub fn new(size: usize) -> Self {
        if size == 0 {
            return AlignedBuffer {
                ptr: ptr::null_mut(),
                layout: Layout::from_size_align(0, ALIGNMENT).unwrap(),
            };
        }
        let layout = Layout::from_size_align(size, ALIGNMENT)
            .expect("Invalid layout for aligned allocation");
        // SAFETY: layout has non-zero size and valid alignment
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        AlignedBuffer { ptr, layout }
    }

    pub fn clone(&self) -> Self {
        let new_buf = AlignedBuffer::new(self.len());
        if self.len() > 0 {
            unsafe {
                ptr::copy_nonoverlapping(self.ptr, new_buf.ptr, self.len());
            }
        }
        new_buf
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            // SAFETY: ptr was allocated with this layout
            unsafe { alloc::dealloc(self.ptr, self.layout) };
        }
    }
}

// SAFETY: The buffer is an owned heap allocation, safe to send across threads.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}
