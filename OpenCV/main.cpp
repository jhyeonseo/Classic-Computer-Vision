#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "MyFunc.h"

CVLAB mylab;
IMAGE image;
VIDEO video;
int main()
{
	Mat v1;
	Mat v2;
	while (1)
	{
		video.frog >> v1;
		video.people >> v2;
		if (v1.empty()||v2.empty())
			break;
		imshow("frog", mylab.MYORB(v1, v2, Size(640, 640)));
		waitKey(10);
	}

	return 0;
}
/*
* 가우시안필터 non-maxima suppression 구현해보기
* eigenvector의 의미 다시 공부
* 나만의 corner 함수 제작
* 기존 함수들의 컬러버전 만들기
*/