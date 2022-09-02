#include "MyFunc.h"

void PixelValue(Mat& img, int x, int y)
{
	int channel = img.channels();

	if (channel == 3)
	{
		int R = img.at<Vec3b>(y, x)[2];
		int G = img.at<Vec3b>(y, x)[1];
		int B = img.at<Vec3b>(y, x)[0];
		std::cout << "(" << x << ", " << y << ")" << ": ";
		std::cout << "[" << R << " " << G << " " << B << "]" << std::endl;
	}
	else
	{
		int Gray = img.at<uchar>(y, x);
		std::cout << "(" << x << ", " << y << ")" << ": " << Gray << std::endl;
	}
}