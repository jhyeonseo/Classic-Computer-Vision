#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "MyFunc.h"


CVLAB mylab;
IMAGE image;
VIDEO video;
int main()
{
	VideoCapture capture(video.surfing);
	Mat frame;

	while (1)
	{
		capture >> frame;
		if (frame.empty())
		{
			waitKey();
			break;
		}
		imshow("ship", mylab.CORNER(frame));
		waitKey(1);
	}

	return 0;
}
/*
* 가우시안필터 non-maxima suppression 구현해보기
* eigenvector의 의미 다시 공부
* 나만의 corner 함수 제작
* 기존 함수들의 컬러버전 만들기
*/