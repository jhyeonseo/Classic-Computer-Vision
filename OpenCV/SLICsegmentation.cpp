#include "main.h"
#include "SLIC.h"

#define slicDraw

void SLICsegmentation(Mat image)
{
	SLIC slic;
	Scalar s;

	int x, y;
	int height, width;
	int numlabels; // Generated number of superpixels
	int m_spcount = 256; // Desired number of superpixels
	double m_compactness = 30;// 20.0; // compactness factor (1-40)

	height = image.rows;
	width = image.cols;

	unsigned int* ubuff = (unsigned int*)calloc(height * width, sizeof(unsigned int));
	int* labels = (int*)calloc(height * width, sizeof(int));

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			ubuff[y * width + x] = (int)image.at<Vec3b>(y, x)[0] + ((int)image.at<Vec3b>(y, x)[1] << 8) + ((int)image.at<Vec3b>(y, x)[2] << 16);
		}
	}

	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(ubuff, width, height, labels, numlabels, m_spcount, m_compactness);

	printf("# of superpixels = %d\n", numlabels);

#ifdef slicDraw
	Mat result(height, width, CV_8UC3);
	slic.DrawContoursAroundSegments(ubuff, labels, width, height, 0);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			result.at<Vec3b>(y, x)[0] = ubuff[y * width + x] & 0xff;
			result.at<Vec3b>(y, x)[1] = ubuff[y * width + x] >> 8 & 0xff;
			result.at<Vec3b>(y, x)[2] = ubuff[y * width + x] >> 16 & 0xff;
		}
	}

	//imwrite("./images/SLIC_segmentation.bmp", result);
#endif

	Mat labelImage(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			labelImage.at<uchar>(y, x) = labels[y * width + x];
		}
	}
	//imwrite("./images/labelImage.bmp", labelImage);
	imshow("result1", result);
	imshow("result2", labelImage);
	waitKey(0);
	free(ubuff);
	free(labels);

}