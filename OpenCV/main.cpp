#include <iostream>
#include "MyFunc.h"

int main()
{
	//Mat input = imread("dogcat.jpeg",0);
	Mat input = Mat::ones(500, 500, CV_8UC3);
	imshow("window", input);
	waitKey(0);
	for (int i = 0; i < input.cols; i++)
		for (int j = 0; j < input.rows; j++)
			PixelValue(input, i, j);





	waitKey(0);

	return 0;
}