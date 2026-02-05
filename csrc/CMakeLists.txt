cmake_minimum_required(VERSION 3.15)
project(corepy_kernels VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Configuring for macOS...")
    
    # Enable Objective-C++ for Metal
    enable_language(OBJCXX)
    
    find_library(METAL_LIB Metal)
    find_library(FOUNDATION_LIB Foundation)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Apple Silicon detected. Building with NEON optimizations (via compiler auto-vectorization).")
        if(METAL_LIB)
            message(STATUS "Metal framework found at ${METAL_LIB}. Enabling COREPY_USE_METAL definition.")
            target_link_libraries(corepy_kernels ${METAL_LIB} ${FOUNDATION_LIB})
            target_compile_definitions(corepy_kernels PRIVATE COREPY_USE_METAL)
        else()
             message(WARNING "Metal framework not found. GPU acceleration will be unavailable.")
        endif()
    else()
        message(STATUS "Intel Mac detected.")
    endif()
endif()

# OpenBLAS detection - cross-platform
if(WIN32)
    # On Windows, check for OPENBLAS_DIR or CMAKE_PREFIX_PATH
    if(DEFINED ENV{OPENBLAS_DIR})
        set(OpenBLAS_INCLUDE_DIRS "$ENV{OPENBLAS_DIR}/include")
        set(OpenBLAS_LIBRARIES "$ENV{OPENBLAS_DIR}/lib/libopenblas.lib")
        if(NOT EXISTS "${OpenBLAS_LIBRARIES}")
            # Try .dll.a for MinGW
            set(OpenBLAS_LIBRARIES "$ENV{OPENBLAS_DIR}/lib/libopenblas.dll.a")
        endif()
        set(OpenBLAS_FOUND TRUE)
    else()
        find_package(OpenBLAS)
    endif()
else()
    find_package(OpenBLAS REQUIRED)
endif()

if(NOT OpenBLAS_FOUND AND NOT WIN32)
    message(FATAL_ERROR "OpenBLAS not found! Install with: apt-get install libopenblas-dev (Ubuntu) or brew install openblas (macOS)")
endif()

# Include directories
if(OpenBLAS_INCLUDE_DIRS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/cpu/elementwise.cpp
    src/cpu/dummy.cpp
    src/cpu/matmul.cpp
    src/cpu/reduce.cpp

    src/cpu/blas_kernels.cpp
)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    list(APPEND SOURCES
        src/metal/metal_backend.mm
    )
endif()

# Create static library
add_library(corepy_kernels STATIC ${SOURCES})

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(corepy_kernels "-framework Metal" "-framework Foundation")
endif()

if(OpenBLAS_FOUND)
    target_compile_definitions(corepy_kernels PRIVATE COREPY_USE_OPENBLAS)
    target_link_libraries(corepy_kernels ${OpenBLAS_LIBRARIES})
endif()

target_include_directories(corepy_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Enable AVX2 SIMD optimizations if supported
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    if(MSVC)
        target_compile_options(corepy_kernels PRIVATE /arch:AVX2)
    else()
        target_compile_options(corepy_kernels PRIVATE -mavx2 -mfma)
    endif()
    target_compile_definitions(corepy_kernels PRIVATE COREPY_USE_AVX2)
endif()

# Installation (optional)
install(TARGETS corepy_kernels DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
