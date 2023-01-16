#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include "utils.h"

#define Max 1000000

using namespace cv;
using namespace cv::dnn;

struct IMAGE
{
	Mat lena1 = imread("../data/pictures/original/lena1.bmp", 1);
	Mat lena2 = imread("../data/pictures/original/lena2.bmp", 1);
	Mat taylor1 = imread("../data/pictures/original/taylor1.jpg", 1);
	Mat taylor2 = imread("../data/pictures/original/taylor2.jpg", 1);
	Mat base = imread("../data/pictures/original/lecture3.bmp", 1);
	Mat compare = imread("../data/pictures/original/compare1.bmp", 1);
	Mat compare2 = imread("../data/pictures/original/compare2.bmp", 1);
	Mat cat = imread("../data/pictures/original/cat1.jpg", 1);
	Mat dog = imread("../data/pictures/original/dog1.jpg", 1);
	Mat text1 = imread("../data/pictures/original/text1.bmp", 1);
	Mat text2 = imread("../data/pictures/original/text2.bmp", 1);
	Mat text3 = imread("../data/pictures/original/text3.bmp", 1);
	Mat text4 = imread("../data/pictures/original/text4.bmp", 1);
	Mat text5 = imread("../data/pictures/original/text5.bmp", 1);
	Mat text6 = imread("../data/pictures/original/text6.bmp", 1);
	Mat text7 = imread("../data/pictures/original/text7.bmp", 1);
	Mat alphabet = imread("../data/pictures/alphabet/alphabet.bmp", 1);
	Mat alphabet2 = imread("../data/pictures/alphabet/alphabet2.bmp", 1);
	Mat example1 = imread("../data/pictures/alphabet/example1.bmp", 1);
	Mat example2 = imread("../data/pictures/alphabet/example2.bmp", 1);
	Mat example3 = imread("../data/pictures/alphabet/example3.bmp", 1);
	Mat example4 = imread("../data/pictures/alphabet/example4.bmp", 1);
	Mat face_ref = imread("../data/pictures/original/face_ref.bmp", 1);
	Mat face_tar = imread("../data/pictures/original/face_tar.bmp", 1);
};
struct VIDEO
{
	VideoCapture frog = VideoCapture("../data/videos/original/frog1.mp4");
	VideoCapture people = VideoCapture("../data/videos/original/people1.mp4");
	VideoCapture people2 = VideoCapture("../data/videos/original/people2.mp4");
	VideoCapture ship = VideoCapture("../data/videos/original/ship1.mp4");
	VideoCapture bloom = VideoCapture("../data/videos/original/bloom1.mp4");
	VideoCapture surfing = VideoCapture("../data/videos/original/surfing1.mp4");
};
struct FEATURE
{
	// Basic information
	Mat image;
	Mat grad;
	Mat magnitude;
	Mat phase;
	int hogsize;
	int lbpsize;
	
	// Feature information
	std::vector<Point> edge;
	std::vector<Point> corner;
	double* hog;
	double* lbp;
};

class CVLAB
{
private:
	std::vector<FEATURE>storage;
	CascadeClassifier cascade;

public:
	// 사용의 편리성을 위한 함수
	CVLAB();
	void INSERT(Mat img);

	// Application을 위한 함수
	Mat EDGE(Mat input);
	Mat CORNER(Mat input, double threshold, int option = 0);   // Option 0: Harris
	Mat LINKCORNER(Mat input1, Mat input2);
	Mat MYORB(Mat img1, Mat img2, Size window);
	int FACE_REGISTRATION(Mat img);    // 얼굴 등록
	void FACE_VERIFICATION(VideoCapture cap);  // 실시간 얼굴 비교 (Landmark 사용, 비교할 얼굴 직접 등록)
	void FACE_VERIFICATION(VideoCapture cap, Mat ref);  // 실시간 얼굴 비교 (Reference와 비교)

	// 이미지 feature 추출을 위한 함수
	void PixelValue(Mat img, int x, int y);
	double* HOG(Mat input, Size block, int interval = 8, int binsize = 9);  // Histograms of oriented gradients
	double* HOG(Mat input, Size block, std::vector<Point> point, int binsize = 9);
	std::vector<Point> HARRIS(Mat input, Size window, double threshold = 0.015);     // Corner point detection
	double* LBP(Mat img);   // Local Binary Pattern
	double* LBP(Mat img, std::vector<Point> point);
	std::vector<Point> FACE_DETECTION(Mat img);    // Face Position Detector
	std::vector<Point> FACE_LDMARK(Mat img);       // Face Landmark Detector

	// 이미지 변환을 위한 함수
	Mat GRAY(Mat img);
	Mat GRAY(Mat img, Point center, Size size);
	Mat RESIZE(Mat img, Size size, int option = 0);
	Mat RESIZE(Mat img, double scalor, int option = 0);
	Mat ROTATE(Mat img, double angle, int option = 0);
	Mat COMBINE(Mat img1, Mat img2);

	// 수학적 처리를 위한 함수
	Mat CONV(Mat input, Mat filter);
	Mat GRADIENT(Mat input);
	Mat MAGNITUDE(Mat gradient);
	Mat PHASE(Mat gradient);
	Mat NORMALIZE(Mat input, double range = 0);  // range 0: L2 normalization
	void NORMALIZE(double* input, double inputsize, double range = 0);
	double SIMILARITY(double* input1, double* input2, int size, int type = 0);

	// 프로젝트 함수
	double LETTER_COMPARE(double* input1, double* input2, int size);
	double* LETTER_LBP(Mat img);
	Mat threshold(Mat input, int color, int threshold);
	Mat refine(Mat input, int pontwidth);
	Mat paint(Mat input, Mat refined, int& count, int pontsize);
	int recur_fill(Mat input, Mat refined, int fill, int x, int y, int pontsize);
	void recur_remove(Mat input, int remove, int x, int y);
	std::vector<Mat> letters(Mat input, int color, int pontwidth);
};


const char lbp_lookup[256] = {
0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,         
11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,  
23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,  
29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };

