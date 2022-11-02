#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "MyFunc.h"

CVLAB mylab;
IMAGE image;
VIDEO video;
using namespace cv;
int main()
{
	int binsize = 18;
	int winw = 36;
	int winh = 36;

	double* ref_hog = mylab.HOG(image.face_ref, Size(winw, winh), 1, binsize);
	double* tar_hog = mylab.HOG(image.face_tar, Size(winw, winh), 1, binsize);

	int xblock = image.face_tar.cols - winw + 1;
	double* similarity = (double*)calloc(image.face_tar.cols * image.face_tar.rows, sizeof(double));
	double* temp = (double*)calloc(binsize, sizeof(double));
	for (int y = 0; y < image.face_tar.rows - winh + 1; y++)
	{
		for (int x = 0; x < image.face_tar.cols - winw + 1; x++)
		{
			for (int i = 0; i < binsize; i++)
			{
				temp[i] = tar_hog[y * xblock * binsize + x * binsize + i];
			}

			similarity[y * image.face_tar.cols + x] = mylab.SIMILARITY(ref_hog, temp, binsize);
		}
	}

	mylab.NORMALIZE(similarity, image.face_tar.cols * image.face_tar.rows, 255);
	Mat hog = Mat::zeros(image.face_tar.rows, image.face_tar.cols, CV_8UC1);
	for (int y = winh/2; y < image.face_tar.rows - winh/2; y++)
	{
		for (int x = winw/2; x < image.face_tar.cols - winw/2; x++)
		{
			hog.at<uchar>(y + winh / 2, x + winw / 2) = similarity[y * image.face_tar.cols + x];
			if (similarity[y * image.face_tar.cols + x] >= 248)
				rectangle(image.face_tar, Rect(Point(x, y), Point(x + winw, y + winh)), Scalar(0, 255, 0), 3, 8, 0);
		}
	}

	imshow("a", image.face_tar);
	imshow("b", hog);
	waitKey(0);

	return 0;
}
/*
* 가우시안필터 non-maxima suppression 구현해보기
* eigenvector의 의미 다시 공부
* 나만의 corner 함수 제작
* 기존 함수들의 컬러버전 만들기
*/