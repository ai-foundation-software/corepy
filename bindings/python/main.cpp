#include <pybind11/pybind11.h>
#include "corepy_kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(_corepy_cpp, m) {
    m.doc() = "Corepy C++ Backend"; 
    
    m.def("add_one", &corepy::add_one_kernel, "A function that adds one");
}
