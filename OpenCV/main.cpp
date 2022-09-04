#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("dog1.jpg", 1);
	Mat input2 = imread("cat1.jpg", 1);

	CVLAB MYLab;
	MYLab.Insert(input1);
	MYLab.Insert(input2);
	MYLab.Editor();


	return 0;
}