#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("../pictures/orginal/dog1.jpg", 1);
	Mat input2 = imread("../pictures/orginal/text1.jpg", 0);
	Mat input3 = imread("../pictures/orginal/space1.jpg", 1);
	CVLAB MYLab;

	Mat grad = MYLab.GRADIENT(input2);
	Mat mag = MYLab.NORMALIZE(MYLab.MAGNITUDE(grad));
	Mat phase = MYLab.NORMALIZE(MYLab.PHASE(grad));

	imshow("orginal", input2);
	imshow("magnitude map", mag);
	imshow("phase map", phase);
	waitKey();

	return 0;
}