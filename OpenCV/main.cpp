#define _CRT_SECURE_NO_WARNINGS
#include "main.h"

CVLAB mylab;
IMAGE image;
VIDEO video;

int main()
{
	Mat orgin = image.text;
	Mat result;
	threshold(orgin, result, 155, 255, THRESH_BINARY);
	imshow("a", result);
	SLICsegmentation(result);
	waitKey();

	return 0;
}

/*
* 가우시안필터 non-maxima suppression 구현해보기
* eigenvector의 의미 다시 공부
* 나만의 corner 함수 제작
* 기존 함수들의 컬러버전 만들기
*/