#include <iostream>
#include <cstdio>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "common.h"
#include <cuda_runtime.h>

using namespace std;

__global__ void get_histogram(unsigned char* input, int width, int height, unsigned int *histogram){
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height)){
		const int tid = yIndex * width + xIndex;
		atomicAdd(&histogram[(int)input[tid]], 1);
	}
}

__global__ void normalize_histogram(unsigned int *histogram, unsigned int *h_s, float n_constant){
	const int idx = threadIdx.x;
	for (size_t i = 0; i < idx; i++) {
		h_s[idx] += histogram[i];
	}
	h_s[idx] *= n_constant;
}

__global__ void copy_to_output(unsigned char* input, unsigned char* output, int width, int height, unsigned int *histogram){
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height)){
		const int tid = yIndex * width + xIndex;
		output[tid] = static_cast<unsigned char>(histogram[(int)input[tid]]);
	}
}

void equalization_host(const cv::Mat& input, cv::Mat& output){
	// Calculate total number of bytes of input and output image
	size_t bytes = input.step * input.rows;

	// Variables to store the input and output images and histograms
	unsigned char *d_input, *d_output;
	unsigned int *histogram, *h_s;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, bytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, bytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned int>(&histogram, 256 * sizeof(unsigned int)), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned int>(&h_s, 256 * sizeof(unsigned int)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), bytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size for image manipulation
	const dim3 block(32, 32);
	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

	// Specify a reasonable block size for histograms manipulation
	const dim3 block_2(256, 1);
	// Calculate grid size to cover the histogram
	const dim3 grid_2(1, 1);

	// Equalize image on device
	auto start_at = std::chrono::high_resolution_clock::now();
	// Kernel to get histogram from input image
	get_histogram<<<grid, block>>>(d_input, input.cols, input.rows, histogram);
	// Kernel to normalize histogram
	normalize_histogram<<<grid_2, block_2>>>(histogram, h_s, 255 / (float)(input.cols * input.rows));
	// Kernel to copy normalized histogram to output image
	copy_to_output<<<grid, block>>>(d_input, d_output, input.cols, input.rows, h_s);
	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
	auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  cout << "Equalize image on device elapsed: " << duration_ms.count() << " ms" << endl;

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, bytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
	SAFE_CALL(cudaFree(histogram), "CUDA Free Failed");
	SAFE_CALL(cudaFree(h_s), "CUDA Free Failed");
}

int main(int argc, char *argv[]) {
  string imagePath;

	if(argc < 2){
		cout << "Please enter image path" << std::endl;
    return -1;
  }
	else
		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);

  if (input.empty()){
		cout << "Image Not Found!" << std::endl;
		return -1;
	}

  // Create output image
	cv::Mat output(input.rows, input.cols, input.type());

	// Call the wrapper function
	equalization_host(input, output);

	// Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

  cv::resizeWindow("Input", 1200, 800);
  cv::resizeWindow("Output", 1200, 800);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

  return 0;
}
