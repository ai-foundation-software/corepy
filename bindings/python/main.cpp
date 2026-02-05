#include <pybind11/pybind11.h>
#include "corepy_kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(_corepy_cpp, m) {
    m.doc() = "Corepy C++ Backend"; 
    
    m.def("add_one", &add_one_kernel, "A function that adds one");

#ifdef __APPLE__
    m.def("metal_add", [](py::buffer a, py::buffer b, py::buffer result) {
        py::buffer_info info_a = a.request();
        py::buffer_info info_b = b.request();
        py::buffer_info info_res = result.request();

        if (info_a.size != info_b.size || info_a.size != info_res.size) {
            throw std::runtime_error("Input shapes must match");
        }

        metal_add(static_cast<float*>(info_a.ptr),
                  static_cast<float*>(info_b.ptr),
                  static_cast<float*>(info_res.ptr),
                  info_a.size);
    }, "Add two arrays using Metal");
#endif
}
