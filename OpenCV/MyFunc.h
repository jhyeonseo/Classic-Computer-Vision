#pragma once
#include <iostream>
#include <cmath>

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
	void GRAY(Mat img, int x, int y, int BLK);
	void RESIZE(Mat img, int x, int y);
	void RESIZE(Mat img, double scalor);
	void ROTATE(Mat img, double angle);
};
void MOUSEINF(int event, int x, int y, int flags, void* MouseData);
void PixelValue(Mat img, int x, int y);
