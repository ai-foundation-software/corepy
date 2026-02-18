cmake_minimum_required(VERSION 3.15)
project(corepy_kernels VERSION 0.2.4 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# =============================================================================
# Metal (macOS) Support
# =============================================================================
# =============================================================================
# Library Definition
# =============================================================================

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

# Metal source files must be added if detected
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    find_library(METAL_LIB Metal)
    find_library(FOUNDATION_LIB Foundation)
    
    if(METAL_LIB AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
         set(COREPY_USE_METAL ON)
         list(APPEND SOURCES src/metal/metal_backend.mm)
         
         # Compile Metal kernels (custom command) - Pre-target
         set(METAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal")
         set(METALLIB_OUT "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")
         find_program(XCRUN_EXE xcrun)
         if(XCRUN_EXE)
            # Test if metal compiler is actually available
            execute_process(
                COMMAND ${XCRUN_EXE} -find metal
                RESULT_VARIABLE METAL_FIND_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(METAL_FIND_RESULT EQUAL 0)
                add_custom_command(
                    OUTPUT ${METALLIB_OUT}
                    COMMAND ${XCRUN_EXE} -sdk macosx metal -c ${METAL_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
                    COMMAND ${XCRUN_EXE} -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${METALLIB_OUT}
                    DEPENDS ${METAL_SRC}
                    COMMENT "Compiling Metal kernels..."
                )
                add_custom_target(compile_metal_kernels ALL DEPENDS ${METALLIB_OUT})
                install(FILES ${METALLIB_OUT} DESTINATION ".")
            else()
                message(WARNING "Metal compiler not found, building without Metal shader compilation. Metal support in Rust will handle kernels at runtime.")
            endif()
         endif()
    endif()
endif()

# Create static library
add_library(corepy_kernels STATIC ${SOURCES})
target_include_directories(corepy_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# =============================================================================
# Target Configuration & Linking
# =============================================================================

# Apple/Metal Configuration
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    enable_language(OBJCXX)
    
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_compile_definitions(corepy_kernels PRIVATE COREPY_PLATFORM_APPLE_SILICON)
        if(COREPY_USE_METAL)
             target_link_libraries(corepy_kernels ${METAL_LIB} ${FOUNDATION_LIB})
             target_compile_definitions(corepy_kernels PRIVATE COREPY_USE_METAL)
             if(TARGET compile_metal_kernels)
                 add_dependencies(corepy_kernels compile_metal_kernels)
             endif()
        endif()
    endif()
endif()

# OpenBLAS Configuration
if(OpenBLAS_FOUND)
    target_compile_definitions(corepy_kernels PRIVATE COREPY_USE_OPENBLAS)
    target_link_libraries(corepy_kernels ${OpenBLAS_LIBRARIES})
endif()

# AVX2 Optimizations (x86)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    if(MSVC)
        target_compile_options(corepy_kernels PRIVATE /arch:AVX2)
    else()
        target_compile_options(corepy_kernels PRIVATE -mavx2 -mfma)
    endif()
    target_compile_definitions(corepy_kernels PRIVATE COREPY_USE_AVX2)
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS corepy_kernels DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
