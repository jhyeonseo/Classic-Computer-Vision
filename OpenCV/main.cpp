#include <iostream>
#include "MyFunc.h"

int main()
{
	Mat input1 = imread("../pictures/orginal/dog1.jpg", 1);
	Mat input2 = imread("../pictures/orginal/cat1.jpg", 1);
	Mat input3 = imread("../pictures/orginal/space1.jpg", 1);
	CVLAB MYLab;

	Mat resize_linear = MYLab.RESIZE(input2, 7.15, 0);
	Mat resize_cubic= MYLab.RESIZE(input2, 7.15, 1);
	Mat resize_nn = MYLab.RESIZE(input2, 7.15, 3);
	imwrite("../pictures/converted/resize_linear.bmp", resize_linear);
	imwrite("../pictures/converted/resize_cubic.bmp", resize_cubic);
	imwrite("../pictures/converted/resize_nn.bmp", resize_nn);

	Mat rotate_linear = MYLab.ROTATE(input2, 71.5);
	Mat rotate_cubic = MYLab.ROTATE(input2, 71.5, 1);
	Mat rotate_nn = MYLab.ROTATE(input2, 71.5, 3);
	imwrite("../pictures/converted/rotate_linear.bmp", rotate_linear);
	imwrite("../pictures/converted/rotate_cubic.bmp", rotate_cubic);
	imwrite("../pictures/converted/rotate_nn.bmp", rotate_nn);

	return 0;
}