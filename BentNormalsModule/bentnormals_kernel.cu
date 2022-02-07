// Copyright (C) 2022, Francis LaBounty, All rights reserved.

#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "Cuda Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
		{
			exit(code);
		}
	}
}

__inline__ __device__ float clamp(float val, float min, float max)
{
	return fmin(fmax(val, min), max);
}

__inline__ __device__ void swap(float& a, float& b)
{
	float t = a;
	a = b;
	b = t;
}

__device__ float raycastCuda(cudaTextureObject_t height_tex, float y0, float x0, float angle, float length, float width, float height)
{
	float baserow = y0;
	float basecol = x0;

	float x1 = truncf(x0 + cosf(angle) * length);
	float y1 = truncf(y0 + sinf(angle) * length);

	bool steep = fabsf(y1 - y0) > fabsf(x1 - x0);

	if (steep)
	{
		swap(x0, y0);
		swap(x1, y1);
	}

	if (x0 > x1)
	{
		swap(x0, x1);
		swap(y0, y1);
	}

	float deltax = x1 - x0;
	float deltay = fabsf(y1 - y0);
	float error = deltax / 2.f;
	float y = y0;
	float ystep = (y0 < y1) ? 1.f : -1.f;
	float maxelevation = 0.f;

	for (float x = x0; x <= x1; x++)
	{
		float row, col;
		float distance;
		if (steep)
		{
			row = x;
			col = y;
		}
		else
		{
			row = y;
			col = x;
		}

		distance = sqrtf((row - baserow) * (row - baserow) + (col - basecol) * (col - basecol));
		maxelevation = fmax(maxelevation, (tex2D<float>(height_tex, col / width, row / height) - tex2D<float>(height_tex, basecol / width, baserow / height)) / fmax(1.f, distance));

		error = error - deltay;
		if (error < 0.f)
		{
			y = y + ystep;
			error = error + deltax;
		}
	}

	return maxelevation;
}

__global__ void raycastCudaKernel(cudaTextureObject_t height_tex, float* mask_output, float anglestep, int raylength, int raycount, int height, int width, int step_size, bool tiled)
{
	for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < height; row += blockDim.y * gridDim.y)
	{
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += blockDim.x * gridDim.x)
		{
			float xsum = 0.f;
			float ysum = 0.f;

			float averagex = 0.f; // R
			float averagey = 0.f; // G

			for (float a = 0.f; a < 2.f * (float)M_PI; a += anglestep)
			{
				float xdir = cosf(a);
				float ydir = sinf(a);

				xsum += fabsf(xdir);
				ysum += fabsf(ydir);

				float ray = raycastCuda(height_tex, row, col, a, raylength, width, height);

				averagex += xdir * ray;
				averagey += ydir * ray;
			}

			averagex /= xsum;
			averagey /= ysum;

			mask_output[(row * step_size) + (4 * col)] = averagex; // R
			mask_output[(row * step_size) + (4 * col) + 1] = averagey; // G
			mask_output[(row * step_size) + (4 * col) + 2] = averagey; // B store copy of averagey so future min-max scaling isn't affected
			mask_output[(row * step_size) + (4 * col) + 3] = averagex; // A store copy of averagex so future min-max scaling isn't affected
		}
	}
}

void run_kernel(float* height_array, float* h_mask_out, int width, int height, int raylength, int raycount, bool tiled)
{
	// texture object set up
	cudaDeviceProp prop;
	gpuErrchk(cudaGetDeviceProperties(&prop, 0));

	float* d_height;
	cudaTextureObject_t height_tex;
	size_t pitch;

	// create CUDA stream
	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	// allocate pitched device memory
	gpuErrchk(cudaMallocPitch(&d_height, &pitch, width * sizeof(float), height));

	// copy memory from host to device
	gpuErrchk(cudaMemcpy2DAsync(d_height, pitch, height_array, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream));

	// CUDA resource description
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = d_height;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
	resDesc.res.pitch2D.pitchInBytes = pitch;

	// CUDA texture description
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = true; // required for 'Wrap' addressMode
	if (tiled)
	{
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
	}
	else
	{
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
	}

	// create texture object
	gpuErrchk(cudaCreateTextureObject(&height_tex, &resDesc, &texDesc, NULL));

	// initialize device pointers
	float* d_mask_out;

	float total_bytes = width * height * 4 * sizeof(float);

	// allocate device memory for return array
	gpuErrchk(cudaMalloc<float>(&d_mask_out, total_bytes));

	// specify block size and grid size
	int n_threads = 256;
	int num_sms;
	int num_blocks_per_sm;

	// get optimal block settings based on device properties
	gpuErrchk(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
	gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, raycastCudaKernel, n_threads, 0));

	int n_blocks = min(num_blocks_per_sm * num_sms, ((height * width) + n_threads - 1) / n_threads);

	int blocks = sqrt(n_blocks); // for easier image indexing
	int threads = sqrt(n_threads); // for easier image indexing

	// calculate anglestep and step_size
	float anglestep = 2.f * (float)M_PI / raycount;
	int step_size = width * 4;

	// launch raycast kernel
	raycastCudaKernel<<<dim3(blocks, blocks), dim3(threads, threads), 0, stream>>>(height_tex, d_mask_out, anglestep, raylength, raycount, height, width, step_size, tiled);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaStreamSynchronize(stream));

	// copy data from device to host memory
	gpuErrchk(cudaMemcpy(h_mask_out, d_mask_out, total_bytes, cudaMemcpyDeviceToHost));

	// cleanup
	gpuErrchk(cudaStreamDestroy(stream));
	gpuErrchk(cudaDestroyTextureObject(height_tex));
	gpuErrchk(cudaFree(d_mask_out));
	gpuErrchk(cudaFree(d_height));
}