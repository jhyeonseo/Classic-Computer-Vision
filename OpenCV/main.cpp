#define _CRT_SECURE_NO_WARNINGS
#include "main.h"

CVLAB mylab;
IMAGE image;
VIDEO video;

int main()
{
	vector<Mat>cap = mylab.letters(image.alphabet, 0, 12);
	vector<Mat>lit = mylab.letters(image.alphabet2, 0, 6);
	double* ref[52];
	for (int i = 0; i < 26; i++)
	{
		ref[i * 2] = mylab.LETTER_LBP(cap[i]);
		ref[i * 2 + 1] = mylab.LETTER_LBP(lit[i]);
	}
	vector<Mat>words = mylab.letters(image.example4, 0, 16);
	double* temp;
	for (int i = 0; i < words.size(); i++)
	{
		temp = mylab.LETTER_LBP(words[i]);
		double max = 0;
		int maxindex = 0;
		for (int j = 0; j < 52; j++)
		{
			if (max < mylab.LETTER_COMPARE(ref[j], temp, 828495))
			{
				max = mylab.LETTER_COMPARE(ref[j], temp, 828495);
				maxindex = j;
			}
		}

		if (max > 0.5)
		{
			printf("감지 성공!\n%c : %f %\n", letter_lookup[maxindex / 2], max);
			imshow("compare", words[i]);
			if (maxindex % 2 == 0)
				imshow("ref", cap[maxindex / 2]);
			else
				imshow("ref", lit[maxindex / 2]);
			waitKey(0);
		}


	}
	

	return 0;
}
