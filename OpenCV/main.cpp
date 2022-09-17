#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("dog1.jpg", 1);
	Mat input2 = imread("cat1.jpg", 1);

	CVLAB MYLab;

	//Mat out = MYLab.RESIZE(input2, 5 , 0);

	MYLab.ROTATE(input2, 50);

	return 0;
}