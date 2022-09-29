#pragma once
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::dnn;

struct Histogram
{
	Mat orgin;
	double* data;
	int datasize;
	int bincount;
};

class CVLAB
{
private:
	std::vector<std::vector<Mat>> storage;
	std::vector<Histogram> histogram;
public:
	// 사용의 편리성을 위한 함수
	CVLAB();
	void Insert(Mat img);
	void Editor();

	// 이미지 분석을 위한 함수
	void PixelValue(Mat img, int x, int y);
	//void HOG(Mat input, int binsize = 9, int cellsize = 64, int blocksize = 4); // Histograms of oriented gradients
	void MYHOG(Mat input, int bincount = 9, int blocksize = 16, int interval = 8);

	// 이미지 변환을 위한 함수
	Mat GRAY(Mat img, int x, int y, int BLK = 0);
	Mat RESIZE(Mat img, double scalor, int option = 0);
	Mat ROTATE(Mat img, double angle, int option = 0);

	// 수학적 처리를 위한 함수
	Mat CONV(Mat input, Mat filter);
	Mat GRADIENT(Mat input);
	Mat MAGNITUDE(Mat gradient);
	Mat PHASE(Mat gradient);
	Mat NORMALIZE(Mat input);
	double DISTANCE(Mat base, Mat compare, int type = 0);  // type0: HOG distance
};

void MOUSEINF(int event, int x, int y, int flags, void* MouseData);

