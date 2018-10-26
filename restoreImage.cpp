#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;

void equalization(unsigned char* input, unsigned char* output, int width, int height){
  size_t i;
  const float n_constant = 255 / (float)(width * height);
  unsigned int histogram[256] = {0};
  for (i = 0; i < height * width; i++) {
    histogram[(int)input[i]] += 1;
  }
  for (i = 1; i < 256; i++) {
    histogram[i] += histogram[i - 1];
  }
  for (i = 0; i < 256; i++) {
    histogram[i] *= n_constant;
  }
  for (i = 0; i < height * width; i++) {
    output[i] = static_cast<unsigned char>(histogram[(int)input[i]]);
  }
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

	// Equalize image on host
	auto start_at = std::chrono::high_resolution_clock::now();
	equalization(input.ptr(), output.ptr(), input.cols, input.rows);
	auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  cout << "\Equalize image on host elapsed: " << duration_ms.count() << " ms" << endl;

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
