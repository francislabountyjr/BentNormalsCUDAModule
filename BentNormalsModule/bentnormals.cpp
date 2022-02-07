// Copyright (C) 2022, Francis LaBounty, All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <math.h>

using namespace std;
namespace py = pybind11;

// @bentnormals_kernel.cu
void run_kernel(float* height_array, float* h_mask_out, int width, int height, int raylength, int raycount, bool tiled);

py::array_t<float> calculateMask(py::array_t<float>& height_array, int width, int height, int raylength = 60, int raycount = 30, bool tiled = true)
{
	// get buffer info and pointer from numpy array
	py::buffer_info buf = height_array.request();

	float* height_ptr = (float*)buf.ptr;

	// ensure height_array has width * height elements
	if (buf.size != width * height)
	{
		throw runtime_error("Exception: height_array does not have (width * height) elements");
	}

	// set up return array
	py::array_t<float> mask(width * height * 4);
	py::buffer_info mask_buf = mask.request();

	float* mask_ptr = (float*)mask_buf.ptr;

	run_kernel(height_ptr, mask_ptr, width, height, raylength, raycount, tiled);

	return mask;
}

PYBIND11_MODULE(bentnormals, m)
{
	m.doc() = "Calculate bent normals mask from a (height, width) float32 NumPy array in the range 0-255 and store in a (height, width, 4) RGBA float32 NumPy array.";

	m.def("calculate_mask", &calculateMask, "Uses CUDA GPU to calculate bent normals mask", py::arg("height_array"),
		py::arg("width"), py::arg("height"), py::arg("raylength"), py::arg("raycount"), py::arg("tiled"));
}