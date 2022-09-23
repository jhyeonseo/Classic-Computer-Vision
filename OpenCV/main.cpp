#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("../pictures/orginal/dog1.jpg", 1);
	Mat input2 = imread("../pictures/orginal/text1.jpg", 0);
	Mat input3 = imread("../pictures/orginal/space1.jpg", 1);
	CVLAB MYLab;

	double data_x[] = { -1,0,1,-1,0,1,-1,0,1 };
	double data_y[] = { -1,-1,-1, 0,0,0, 1,1,1 };

	Mat edge_x(3, 3, CV_64FC1, data_x);
	Mat edge_y(3, 3, CV_64FC1, data_y);

	Mat grad_x = CONV(input2, edge_x);
	Mat grad_y = CONV(input2, edge_y);

	Mat result_mag = NORMALIZE(MAG(grad_x, grad_y));
	Mat result_phase = NORMALIZE(PHASE(grad_x, grad_y));

	imshow("orgin", input2);
	imshow("mag", result_mag);
	imshow("phase", result_phase);
	waitKey();

	return 0;
}