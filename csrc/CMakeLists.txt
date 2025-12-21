add_library(corepy_kernels OBJECT
    kernels/dummy.cpp
)

target_include_directories(corepy_kernels PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(corepy_kernels PUBLIC cxx_std_20)
