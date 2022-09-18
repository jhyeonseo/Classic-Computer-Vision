#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("dog1.jpg", 1);
	Mat input2 = imread("cat1.jpg", 1);
	Mat input3 = imread("space1.jpg", 1);
	CVLAB MYLab;

	Mat resize_nn = MYLab.RESIZE(input3, 16 , 2);
	Mat resize_linear = MYLab.RESIZE(input3, 16);
	imwrite("resize_NN.bmp", resize_nn);
	imwrite("resize_Linear.bmp", resize_linear);

	Mat rotate_nn = MYLab.ROTATE(input3, 50, 2);
	Mat rotate_linear = MYLab.ROTATE(input3, 50);
	imwrite("rotate_NN.bmp", rotate_nn);
	imwrite("rotate_Linear.bmp", rotate_linear);

	return 0;
}