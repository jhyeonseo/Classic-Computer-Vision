#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::dnn;

class CVLAB
{
private:
	std::vector<std::vector<Mat>> storage;
public:
	CVLAB();
	void Insert(Mat img);
	void Editor();
};
void MOUSEINF(int event, int x, int y, int flags, void* MouseData);
void PixelValue(Mat img, int x, int y);
