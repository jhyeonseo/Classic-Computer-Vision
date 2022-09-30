#include <iostream>
#include "MyFunc.h"

CVLAB mylab;
int main()
{
	Mat input = imread("../pictures/orginal/ref.bmp", 0);
	Mat corner = mylab.HARRIS(input, Size(3, 3));

	Mat grad = mylab.NORMALIZE(mylab.GRADIENT(input), 1);
	
	Mat mag = mylab.MAGNITUDE(grad);
	Mat edge;
	mylab.NORMALIZE(mag, 255).convertTo(edge, CV_8UC1);

	namedWindow("orgin", 0);
	resizeWindow("orgin", 640, 480);
	namedWindow("corner", 0);
	resizeWindow("corner", 640, 480);
	imshow("orgin", input);
	imshow("corner", corner);
	waitKey();

	return 0;
}
/*
* 이미지의 features들을 효과적이게 저장하는 구조체 고안
* 이미지를 입력받아 corner을 중심으로 15 x 15 pixel에서 histogram을 만들고 연결하는 함수 구현 (과제)
* HOG 함수 재정비
* 가우시안필터, non-maxima suppression 구현해보기
* eigenvector의 의미 다시 공부
* 나만의 corner 함수 제작
* 기존 함수들의 컬러버전 만들기
*/