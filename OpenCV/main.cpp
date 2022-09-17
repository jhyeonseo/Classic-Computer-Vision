#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("dog1.jpg", 1);
	Mat input2 = imread("cat1.jpg", 1);

	CVLAB MYLab;

	MYLab.Insert(input1);
	MYLab.Insert(input2);

	//MYLab.GRAY(input2, 400, 400, 100);
	//MYLab.Editor();
	//MYLab.RESIZE(input2, sqrt(3.141592));
	MYLab.ROTATE(input2, 50);

	return 0;
}