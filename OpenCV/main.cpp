#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat base= imread("../pictures/orginal/lecture3.bmp", 0);
	Mat compare1 = imread("../pictures/orginal/compare1.bmp", 0);
	Mat compare2 = imread("../pictures/orginal/compare2.bmp", 0);
	CVLAB MYLab;

	double distance1, distance2;
	distance1 = MYLab.DISTANCE(base, compare1);
	distance2 = MYLab.DISTANCE(base, compare2);

	printf("Distance with lecture3 and compare1: %f\n", distance1);
	printf("Distance with lecture3 and compare2: %f\n", distance2);

	return 0;
}