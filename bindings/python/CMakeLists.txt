python_add_library(_corepy_cpp MODULE main.cpp)

# Link against the object library from csrc and pybind11 headers
target_link_libraries(_corepy_cpp PRIVATE corepy_kernels pybind11::headers)

# Install the extension into the corepy package directory
install(TARGETS _corepy_cpp DESTINATION corepy)
