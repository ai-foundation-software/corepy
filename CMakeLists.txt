cmake_minimum_required(VERSION 3.20)
project(corepy LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find pybind11 with fallback
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add C++ source directory
add_subdirectory(csrc)

# Add Bindings directory (where the extension is defined)
add_subdirectory(bindings/python)

# Future: Add Rust build integration here (via Corrosion or custom command)
