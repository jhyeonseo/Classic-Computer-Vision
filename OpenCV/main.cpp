#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("../pictures/orginal/text1.jpg", 0);
	Mat base= imread("../pictures/orginal/lecture3.bmp", 0);
	Mat compare1 = imread("../pictures/orginal/compare1.bmp", 0);
	Mat compare2 = imread("../pictures/orginal/compare2.bmp", 0);
	CVLAB MYLab;

	Mat grad = MYLab.GRADIENT(input1);
	Mat mag = MYLab.NORMALIZE(MYLab.MAGNITUDE(grad));
	Mat phase = MYLab.NORMALIZE(MYLab.PHASE(grad));

	/*
	imshow("orginal", input2);
	imshow("magnitude map", mag);
	imshow("phase map", phase);
	waitKey(1000);
	*/

	MYLab.Similiarity(base, compare1);
	MYLab.Similiarity(base, compare2);

	return 0;
}