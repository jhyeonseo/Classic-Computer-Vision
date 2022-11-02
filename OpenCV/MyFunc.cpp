#define _CRT_SECURE_NO_WARNINGS
#include "MyFunc.h"

CVLAB::CVLAB()
{
}
void CVLAB::Insert(Mat input)
{
	FEATURE information;
	information.image = input;
	information.grad = GRADIENT(input);
	information.magnitude = MAGNITUDE(information.grad);
	information.phase = PHASE(information.grad);

	this->storage.push_back(information);
}
/*
void CVLAB::Editor()
{
	char command;
	std::cin >> command;

	int x, y, flags;
	int* event = new int[this->storage.size()];
	void*** MouseData = new void** [this->storage.size()];
	for (int i = 0; i < this->storage.size(); i++)
		MouseData[i] = new void* [4];


	for (int i = 0; i < this->storage.size(); i++)
	{
		event[i] = -1;
		MouseData[i][0] = &event[i];
		MouseData[i][1] = &x;
		MouseData[i][2] = &y;
		MouseData[i][3] = &flags;
		cv::namedWindow(std::to_string(i));
		setMouseCallback(std::to_string(i), MOUSEINF, MouseData[i]);
	}
	if (command == 'p')
	{
		while (1)
		{
			waitKey(1);

			for (int i = 0; i < this->storage.size(); i++)
			{
				imshow(std::to_string(i), this->storage[i][0]);

				if (event[i] == EVENT_LBUTTONDOWN)
				{
					std::cout << "Window " << i << " ";
					PixelValue(this->storage[i][0], x, y);
				}
				event[i] = -1;
			}
		}
	}
	else if (command == 'g')
	{
		Mat* gray = new Mat[this->storage.size()];
		while (1)
		{
			for (int i = 0; i < this->storage.size(); i++)
			{
				if (event[i] == EVENT_MOUSEMOVE)
				{
					gray[i] = GRAY(this->storage[i][0], x, y, 100);
					imshow(std::to_string(i), gray[i]);
				}
				waitKey(1);
				event[i] = -1;
			}
		}
		delete[]gray;
	}


	delete[]event;
	for (int i = 0; i < this->storage.size(); i++)
		delete[]MouseData[i];
	delete[]MouseData;

}
*/
void CVLAB::Print(Mat input)
{
	if (input.type() == CV_64FC1)
	{
		for (int y = 0; y < input.rows; y++)
		{
			printf("[ ");
			for (int x = 0; x < input.cols; x++)
			{
				printf("%f ", input.at<double>(y, x));
			}
			printf("]\n");
		}
	}
	else if (input.type() == CV_64FC2)
	{
		for (int y = 0; y < input.rows; y++)
		{
			printf("[ ");
			for (int x = 0; x < input.cols; x++)
			{
				printf("(%f, %f) ", input.at<Vec2d>(y, x)[0], input.at<Vec2d>(y, x)[1]);
			}
			printf("]\n");
		}
	}
	else if (input.type() == CV_8UC1)
	{
		for (int y = 0; y < input.rows; y++)
		{
			for (int x = 0; x < input.cols; x++)
			{
				printf("%d ", input.at<int>(y, x));
			}
			printf("\n");
		}
	}
}

// Application을 위한 함수
Mat CVLAB::EDGE(Mat img)
{
	Mat output;
	NORMALIZE(MAGNITUDE(GRADIENT(img)),255).convertTo(output, CV_8UC1);
	return output;
}
Mat CVLAB::CORNER(Mat img, int option)
{
	Mat copy;
	img.copyTo(copy);
	if (option == 0)
	{
		int width = img.cols;
		int height = img.rows;

		std::vector<Point> corner = HARRIS(img, Size(3, 3));
		for (int i = 0; i < corner.size(); i++)
			circle(copy, corner[i], 3, Scalar(0, 255, 255), 2, 8, 0);
	}

	return copy;
}
Mat CVLAB::LINKCORNER(Mat img1, Mat img2)
{
	std::vector<Point> corner1 = HARRIS(img1, Size(3, 3));
	std::vector<Point> corner2 = HARRIS(img2, Size(3, 3));
	double* histogram1 = (double*)calloc(corner1.size() * 9, sizeof(double));
	double* histogram2 = (double*)calloc(corner2.size() * 9, sizeof(double));
	double* similarity = (double*)calloc(corner1.size(), sizeof(double));
	int* similarity_index = (int*)calloc(corner1.size(), sizeof(int));
	double THRESHOLD = 0.125;
	int BLK = 15;
	int binsize = 9;

	if ((img1.rows != img2.rows) || (img1.cols != img2.cols))
		img2 = RESIZE(img2, Size(img1.cols, img1.rows));
	Mat output = COMBINE(img1, img2);
	histogram1 = HOG(img1, Size(BLK, BLK), corner1, binsize);
	histogram2 = HOG(img2, Size(BLK, BLK), corner2, binsize);
	for (int c1 = 0; c1 < corner1.size(); c1++)
	{
		int cx1 = corner1[c1].x;
		int cy1 = corner1[c1].y;
		if (cx1<BLK / 2 || cx1>img1.cols - BLK / 2 || cy1<BLK / 2 || cy1>img1.rows - BLK / 2)
			continue;
		for (int c2 = 0; c2 < corner2.size(); c2++)
		{
			// c1, c2 = 코너의 순서
			int cx2 = corner2[c2].x;
			int cy2 = corner2[c2].y;
			if (cx2<BLK / 2 || cx2>img2.cols - BLK / 2 || cy2<BLK / 2 || cy2>img2.rows - BLK / 2)
				continue;
			double value = 0;
			for (int i = 0; i < binsize; i++)
				value += (histogram1[c1 * binsize + i] - histogram2[c2 * binsize + i]) * (histogram1[c1 * binsize + i] - histogram2[c2 * binsize + i]);
			if (value > 0.0)
				value = 1 / std::sqrt(value);
			else
				value = INT_MAX;
			if (value > similarity[c1])
			{
				similarity[c1] = value;
				similarity_index[c1] = c2;
			}
		}
	}
	NORMALIZE(similarity, corner1.size());
	for (int i = 0; i < corner1.size(); i++)
		if (similarity[i] > THRESHOLD)
			line(output, corner1[i], corner2[similarity_index[i]] + Point(img1.rows, 0), Scalar(255, 0, 0), 2, 9, 0);

	free(histogram1);
	free(histogram2);
	free(similarity);
	free(similarity_index);
	return output;
}
Mat CVLAB::MYORB(Mat img1, Mat img2, Size window)
{
	// ORB settings
	int ORB_MAX_KPTS = 128;
	float ORB_SCALE_FACTOR = 1.2;
	int ORB_PYRAMID_LEVELS = 4;
	float ORB_EDGE_THRESHOLD = 31.0;
	int ORB_FIRST_PYRAMID_LEVEL = 0;
	int ORB_WTA_K = 2;
	int ORB_PATCH_SIZE = 31;
	// Some image matching options
	float MIN_H_ERROR = 2.50f; // Maximum error in pixels to accept an inlier
	float DRATIO = 0.80f;

	// Input image (color)
	Mat img1_rgb_orb = RESIZE(img1, window);
	Mat img2_rgb_orb = RESIZE(img2, window);
	// Input image (gray)
	Mat img1_gray, img2_gray;
	img1_gray = GRAY(img1_rgb_orb);
	img2_gray = GRAY(img2_rgb_orb);
	// Input image to float
	Mat img1_32, img2_32;
	img1_gray.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	img2_gray.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);
	// Output image (color)
	Mat img_com_orb = Mat(Size(img1_gray.cols * 2, img1_gray.rows), CV_8UC3);

	// Time create the L2 and L1 matchers
	double t1 = 0.0, t2 = 0.0;
	double torb = 0.0;

	// Corner Points
	std::vector<KeyPoint> kpts1_orb, kpts2_orb;
	// Descriptors
	Mat desc1_orb, desc2_orb;
	// Matches btw Descriptors
	std::vector<Point2f> matches_orb, inliers_orb;
	std::vector<std::vector<DMatch>> dmatches_orb;

	Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
	Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");

	// ORB Features
	Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
		ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB::HARRIS_SCORE,
		ORB_PATCH_SIZE);

	t1 = getTickCount();
	// Detect Corner and Calculate Descriptor
	orb->detectAndCompute(img1_gray, noArray(), kpts1_orb, desc1_orb, false);
	orb->detectAndCompute(img2_gray, noArray(), kpts2_orb, desc2_orb, false);

	// Find best matches using descriptor
	matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
	matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);
	compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);
	t2 = cv::getTickCount();

	// Draw corners and matches in image and show
	draw_keypoints(img1_rgb_orb, kpts1_orb);
	draw_keypoints(img2_rgb_orb, kpts2_orb);
	draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);

	// Show measures
	int nmatches_orb = 0, ninliers_orb = 0, noutliers_orb = 0;
	int nkpts1_orb = 0, nkpts2_orb = 0;
	float ratio_orb = 0.0;
	nkpts1_orb = kpts1_orb.size();
	nkpts2_orb = kpts2_orb.size();
	nmatches_orb = matches_orb.size() / 2;
	ninliers_orb = inliers_orb.size() / 2;
	noutliers_orb = nmatches_orb - ninliers_orb;
	ratio_orb = 100.0 * (float)(ninliers_orb) / (float)(nmatches_orb);
	torb = 1000.0 * (t2 - t1) / cv::getTickFrequency();
	std::cout << "ORB Results" << std::endl;
	std::cout << "**************************************" << std::endl;
	std::cout << "Number of Keypoints Image 1: " << nkpts1_orb << std::endl;
	std::cout << "Number of Keypoints Image 2: " << nkpts2_orb << std::endl;
	std::cout << "Number of Matches: " << nmatches_orb << std::endl;
	std::cout << "Number of Inliers: " << ninliers_orb << std::endl;
	std::cout << "Number of Outliers: " << noutliers_orb << std::endl;
	std::cout << "Inliers Ratio: " << ratio_orb << std::endl;
	std::cout << "ORB Features Extraction Time (ms): " << torb << std::endl; std::cout << std::endl;

	return img_com_orb;
}

// 이미지 feature 추출을 위한 함수
void CVLAB::PixelValue(Mat img, int x, int y)
{
	int channel = img.channels();
	if (channel == 3)
	{
		int R = img.at<Vec3b>(y, x)[2];
		int G = img.at<Vec3b>(y, x)[1];
		int B = img.at<Vec3b>(y, x)[0];
		std::cout << "(" << x << ", " << y << ")" << ": ";
		std::cout << "[" << R << " " << G << " " << B << "]" << std::endl;
	}
	else
	{
		int Gray = img.at<uchar>(y, x);
		std::cout << "(" << x << ", " << y << ")" << ": " << Gray << std::endl;
	}
}
double* CVLAB::HOG(Mat input, Size block, int interval, int binsize)
{
	int h = block.height;
	int w = block.width;
	double bininterval = 180 / binsize;
	int yblock = (input.rows - h) / interval + 1;
	int xblock = (input.cols - w) / interval + 1;
	double* output = (double*)calloc(xblock * yblock * binsize, sizeof(double));
	double* temp = (double*)calloc(binsize, sizeof(double));
	Mat phase = PHASE(GRADIENT(input));
	Mat mag = MAGNITUDE(GRADIENT(input));
	for (int y = 0; y < input.rows - h + 1; y += interval)
	{
		for (int x = 0; x < input.cols - w + 1; x += interval)
		{
			int bx = x / interval;
			int by = y / interval;
			// x, y = 이미지에서 block의 첫번째 픽셀에 해당하는 좌표
			// bx, by = block의 좌표
			for (int i = 0; i < h; i++)
			{
				for (int j = 0; j < w; j++)
				{
					int binindex;
					double degree = phase.at<double>(y + i, x + j) * 57.2958;
					if (degree < 0.0)
						degree += 180;
					if (degree >= 180.0)
						binindex = 0;
					else
						binindex = degree / bininterval;

					temp[binindex] += mag.at<double>(y + i, x + j);
				}

			}
			NORMALIZE(temp, binsize);
			for (int i = 0; i < binsize; i++)
			{
				output[by * xblock * binsize + bx * binsize + i] = temp[i];
				temp[i] = 0;
			}
		}
	}
	return output;
}
double* CVLAB::HOG(Mat input, Size block, std::vector<Point> point, int binsize)
{
	int h = block.height;
	int w = block.width;
	double bininterval = 180 / binsize;
	Mat phase = PHASE(GRADIENT(input));
	Mat mag = MAGNITUDE(GRADIENT(input));
	double* output = (double*)calloc(point.size() * binsize, sizeof(double));
	double* temp = (double*)calloc(binsize, sizeof(double));
	for (int p = 0; p < point.size(); p++)
	{
		int cx = point[p].x;
		int cy = point[p].y;
		for (int i = cy - h/2; i <= cy + h/2; i++)
		{
			for (int j = cx - w/2; j <= cx + w/2; j++)
			{
				// p = 좌표의 순번
				// cx, cy = block의 중심 좌표
				// i, j = block안의 픽셀의 좌표
				if (j < 0 || j >= input.cols || i < 0 || i >= input.rows)
					continue;
				int binindex;
				double degree = phase.at<double>(i, j) * 57.2958;
				if (degree < 0.0)
					degree += 180;
				if (degree >= 180.0)
					binindex = 0;
				else
					binindex = degree / bininterval;

				temp[binindex] += mag.at<double>(i, j);
			}
		}
		NORMALIZE(temp, binsize);
		for (int i = 0; i < binsize; i++)
		{
			output[p * binsize + i] = temp[i];
			temp[i] = 0;
		}
	}
	free(temp);
	return output;
}
std::vector<Point> CVLAB::HARRIS(Mat input, Size window, double threshold)
{
	Mat grad = NORMALIZE(GRADIENT(input), 1);

	std::vector<Point> corner;
	double M[2][2];
	double k = 0.05;
	int window_w = window.width;
	int window_h = window.height;

	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++) // window interval = 1
		{
			// x,y = window center index
			M[0][0] = M[0][1] = M[1][0] = M[1][1] = 0.0;
			for (int i = y - window_h / 2; i <= y + window_h / 2; i++)
			{
				for (int j = x - window_w / 2; j <= x + window_w / 2; j++)
				{
					// i,j = windows의 픽셀의 index
					if (i < 0 || i >= input.rows || j < 0 || j >= input.cols)
						continue;

					double ix = grad.at<Vec2d>(i, j)[0];
					double iy = grad.at<Vec2d>(i, j)[1];

					M[0][0] += ix * ix;
					M[0][1] += ix * iy;
					M[1][0] += ix * iy;
					M[1][1] += iy * iy;
				}
			}
			// 하나의 window에서 M 완성

			double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];  // determinant component
			double tr = M[0][0] + M[1][1];  // trace component
			double R = det - (k * tr * tr);

			if (R > threshold)
				corner.push_back(Point(x, y));	
		}
	}

	return corner;
}

// 이미지 변환을 위한 함수
Mat CVLAB::GRAY(Mat img)
{
	Mat gray(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < gray.rows; y++)
	{
		for (int x = 0; x < gray.cols; x++)
		{
			int value = (img.at<Vec3b>(y, x)[0] + img.at<Vec3b>(y, x)[1] + img.at<Vec3b>(y, x)[2]) / 3;
			gray.at<uchar>(y, x) = value;
		}
	}

	return gray;
}
Mat CVLAB::GRAY(Mat img, Point center, Size size)
{
	Mat gray = img.clone();
	int cx = center.x;
	int cy = center.y;
	int height = size.height;
	int width = size.width;

	for (int y = cy - height / 2; y <= cy + height / 2; y++)
	{
		if (y<0 || y>=gray.rows)
			continue;
		for (int x = cx - width / 2; x < cx + width / 2; x++)
		{
			if (x < 0 || x >= gray.cols)
				continue;

			int value = (img.at<Vec3b>(y, x)[0] + img.at<Vec3b>(y, x)[1] + img.at<Vec3b>(y, x)[2]) / 3;
			gray.at<Vec3b>(y, x)[0] = gray.at<Vec3b>(y, x)[1] = gray.at<Vec3b>(y, x)[2] = value;
		}
	}

	return gray;
}
Mat CVLAB::RESIZE(Mat img, Size size, int option)
{
	int width = round(size.width);
	int height = round(size.height);
	double ratio_x = (double)img.cols / (double)width;
	double ratio_y = (double)img.rows / (double)height;

	Mat resize(height, width, CV_8UC3);

	for (int x = 0; x < resize.cols; x++)
	{
		for (int y = 0; y < resize.rows; y++)
		{
			int x_orgin = std::floor(x * ratio_x);
			int y_orgin = std::floor(y * ratio_y);

			int x_left = floor(x_orgin);
			int x_right = x_left + 1;
			int y_bot = floor(y_orgin);
			int y_top = y_bot + 1;

			if (x_right >= img.cols || y_top >= img.rows || x_left < 0 || y_bot < 0)
				continue;

			Vec3d value = 0;
			if (option == 0)
			{
				Vec3d value1, value2, value3, value4 = 0;
				value1 += (Vec3d)img.at<Vec3b>(y_bot, x_left) * (1 - (x_orgin - x_left));
				value1 += (Vec3d)img.at<Vec3b>(y_bot, x_right) * (x_orgin - x_left);
				value1 = value1 * (1 - (y_orgin - y_bot));
				value2 += (Vec3d)img.at<Vec3b>(y_top, x_left) * (1 - (x_orgin - x_left));
				value2 += (Vec3d)img.at<Vec3b>(y_top, x_right) * (x_orgin - x_left);
				value2 = value2 * (y_orgin - y_bot);
				value3 += (Vec3d)img.at<Vec3b>(y_bot, x_left) * (1 - (y_orgin - y_bot));
				value3 += (Vec3d)img.at<Vec3b>(y_top, x_left) * (y_orgin - y_bot);
				value3 = value3 * (1 - (x_orgin - x_left));
				value4 += (Vec3d)img.at<Vec3b>(y_bot, x_right) * (1 - (y_orgin - y_bot));
				value4 += (Vec3d)img.at<Vec3b>(y_top, x_right) * (y_orgin - y_bot);
				value4 = value4 * (x_orgin - x_left);

				value = (value1 + value2 + value3 + value4) / 2;
			}
			else if (option == 1)
			{
				double normalize = 0;
				value += (Vec3d)img.at<Vec3b>(y_bot, x_left) * std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_bot, 2));
				value += (Vec3d)img.at<Vec3b>(y_bot, x_right) * std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_bot, 2));
				value += (Vec3d)img.at<Vec3b>(y_top, x_left) * std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_top, 2));
				value += (Vec3d)img.at<Vec3b>(y_top, x_right) * std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_top, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_bot, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_bot, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_top, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_top, 2));
				value = value / normalize;
			}
			else if (option == 2)
			{
				int x_neighbor = round(x_orgin);
				int y_neighbor = round(y_orgin);

				value = (Vec3d)img.at<Vec3b>(y_neighbor, x_neighbor);
			}
			else if (option == 3)
			{
				value = (Vec3d)img.at<Vec3b>(y_bot, x_left);
			}

			resize.at<Vec3b>(y, x) = (Vec3b)value;
		}
	}

	return resize;
}
Mat CVLAB::RESIZE(Mat img, double scalar, int option)
{
	scalar = sqrt(scalar);
	int width = round(img.cols * scalar);
	int height = round(img.rows * scalar);
	double ratio_x = (double)img.cols / (double)width;
	double ratio_y = (double)img.rows / (double)height;

	Mat resize(height, width, CV_8UC3);

	for (int x = 0; x < resize.cols; x++)
	{
		for (int y = 0; y < resize.rows; y++)
		{
			int x_orgin = std::floor(x * ratio_x);
			int y_orgin = std::floor(y * ratio_y);

			int x_left = floor(x_orgin);
			int x_right = x_left + 1;
			int y_bot = floor(y_orgin);
			int y_top = y_bot + 1;

			if (x_right >= img.cols || y_top >= img.rows || x_left < 0 || y_bot < 0)
				continue;

			Vec3d value = 0;
			if (option == 0)
			{
				Vec3d value1, value2, value3, value4 = 0;
				value1 += (Vec3d)img.at<Vec3b>(y_bot, x_left) * (1 - (x_orgin - x_left));
				value1 += (Vec3d)img.at<Vec3b>(y_bot, x_right) * (x_orgin - x_left);
				value1 = value1 * (1 - (y_orgin - y_bot));
				value2 += (Vec3d)img.at<Vec3b>(y_top, x_left) * (1 - (x_orgin - x_left));
				value2 += (Vec3d)img.at<Vec3b>(y_top, x_right) * (x_orgin - x_left);
				value2 = value2 * (y_orgin - y_bot);
				value3 += (Vec3d)img.at<Vec3b>(y_bot, x_left) * (1 - (y_orgin - y_bot));
				value3 += (Vec3d)img.at<Vec3b>(y_top, x_left) * (y_orgin - y_bot);
				value3 = value3 * (1 - (x_orgin - x_left));
				value4 += (Vec3d)img.at<Vec3b>(y_bot, x_right) * (1 - (y_orgin - y_bot));
				value4 += (Vec3d)img.at<Vec3b>(y_top, x_right) * (y_orgin - y_bot);
				value4 = value4 * (x_orgin - x_left);

				value = (value1 + value2 + value3 + value4) / 2;
			}
			else if (option == 1)
			{
				double normalize = 0;
				value += (Vec3d)img.at<Vec3b>(y_bot, x_left) * std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_bot, 2));
				value += (Vec3d)img.at<Vec3b>(y_bot, x_right) * std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_bot, 2));
				value += (Vec3d)img.at<Vec3b>(y_top, x_left) * std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_top, 2));
				value += (Vec3d)img.at<Vec3b>(y_top, x_right) * std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_top, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_bot, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_bot, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_top, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_top, 2));
				value = value / normalize;
			}
			else if (option == 2)
			{
				int x_neighbor = round(x_orgin);
				int y_neighbor = round(y_orgin);

				value = (Vec3d)img.at<Vec3b>(y_neighbor, x_neighbor);
			}
			else if (option == 3)
			{
				value = (Vec3d)img.at<Vec3b>(y_bot, x_left);
			}

			resize.at<Vec3b>(y, x) = (Vec3b)value;
		}
	}

	return resize;
}
Mat CVLAB::ROTATE(Mat img, double angle, int option)
{
	Mat rotate = Mat::zeros(img.rows, img.cols, CV_8UC3);
	angle = -angle * (3.141592 / 180);


	double x_center = rotate.cols / 2;
	double y_center = rotate.rows / 2;

	double ROT[2][2] = { {cos(angle),sin(angle)},{-sin(angle),cos(angle)} };

	for (int x = 0; x < rotate.cols; x++)
	{
		for (int y = 0; y < rotate.rows; y++)
		{
			double x_orgin = ROT[0][0] * (x - x_center) + ROT[0][1] * (y - y_center);
			double y_orgin = ROT[1][0] * (x - x_center) + ROT[1][1] * (y - y_center);

			x_orgin = x_orgin + x_center;
			y_orgin = y_orgin + y_center;

			int x_left = floor(x_orgin);
			int x_right = x_left + 1;
			int y_bot = floor(y_orgin);
			int y_top = y_bot + 1;

			if (x_right >= img.cols || y_top >= img.rows || x_left < 0 || y_bot < 0)
				continue;

			Vec3d value = 0;
			if (option == 0)
			{
				Vec3d value1, value2, value3, value4 = 0;
				value1 += (Vec3d)img.at<Vec3b>(y_bot, x_left) * (1 - (x_orgin - x_left));
				value1 += (Vec3d)img.at<Vec3b>(y_bot, x_right) * (x_orgin - x_left);
				value1 = value1 * (1 - (y_orgin - y_bot));
				value2 += (Vec3d)img.at<Vec3b>(y_top, x_left) * (1 - (x_orgin - x_left));
				value2 += (Vec3d)img.at<Vec3b>(y_top, x_right) * (x_orgin - x_left);
				value2 = value2 * (y_orgin - y_bot);
				value3 += (Vec3d)img.at<Vec3b>(y_bot, x_left) * (1 - (y_orgin - y_bot));
				value3 += (Vec3d)img.at<Vec3b>(y_top, x_left) * (y_orgin - y_bot);
				value3 = value3 * (1 - (x_orgin - x_left));
				value4 += (Vec3d)img.at<Vec3b>(y_bot, x_right) * (1 - (y_orgin - y_bot));
				value4 += (Vec3d)img.at<Vec3b>(y_top, x_right) * (y_orgin - y_bot);
				value4 = value4 * (x_orgin - x_left);

				value = (value1 + value2 + value3 + value4) / 2;
			}
			else if (option == 1)
			{
				double normalize = 0;
				value += (Vec3d)img.at<Vec3b>(y_bot, x_left) * std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_bot, 2));
				value += (Vec3d)img.at<Vec3b>(y_bot, x_right) * std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_bot, 2));
				value += (Vec3d)img.at<Vec3b>(y_top, x_left) * std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_top, 2));
				value += (Vec3d)img.at<Vec3b>(y_top, x_right) * std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_top, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_bot, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_bot, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_left, 2) + std::pow(y_orgin - y_top, 2));
				normalize += std::sqrt(std::pow(x_orgin - x_right, 2) + std::pow(y_orgin - y_top, 2));
				value = value / normalize;
			}
			else if (option == 2)
			{
				int x_neighbor = round(x_orgin);
				int y_neighbor = round(y_orgin);

				value = (Vec3d)img.at<Vec3b>(y_neighbor, x_neighbor);
			}
			else if (option == 3)
			{
				value = (Vec3d)img.at<Vec3b>(y_bot, x_left);
			}

			rotate.at<Vec3b>(y, x) = (Vec3b)value;
		}
	}

	return rotate;
}
Mat CVLAB::COMBINE(Mat img1, Mat img2)
{
	if ((img1.rows != img2.rows) || (img1.cols != img2.cols))
		img2 = RESIZE(img1, Size(img1.cols, img1.rows));

	Mat output(img1.rows, img1.cols * 2, img1.type());
	if (img1.type() == CV_8UC3)
	{
		for (int y = 0; y < img1.rows; y++)
		{
			for (int x = 0; x < img1.cols; x++)
			{
				output.at<Vec3b>(y, x) = img1.at<Vec3b>(y, x);
				output.at<Vec3b>(y, x + img1.cols) = img2.at<Vec3b>(y, x);
			}
		}
	}
	else if (img1.type() == CV_8UC1)
	{
		for (int y = 0; y < img1.rows; y++)
		{
			for (int x = 0; x < img1.cols; x++)
			{
				output.at<uchar>(y, x) = img1.at<uchar>(y, x);
				output.at<uchar>(y, x + img1.cols) = img2.at<uchar>(y, x);
			}
		}
	}
	return output;
}

// 수학적 처리를 위한 함수
Mat CVLAB::CONV(Mat input, Mat filter)
{
	Mat output = Mat::zeros(input.rows, input.cols, CV_64FC1);
	int fcx = filter.cols / 2;
	int fcy = filter.rows / 2;

	for (int cy = 0; cy < input.rows; cy++)
	{
		for (int cx = 0; cx < input.cols; cx++)
		{
			double value = 0;
			for (int i = -fcy; i <= fcy; i++)
			{
				for (int j = -fcx; j <= fcx; j++)
				{
					// fcx, fcy = 필터의 중심
					// i, j = input, filter의 중심으로부터 떨어진 칸수
					if (cx + j < 0 || cx + j >= input.cols || cy + i < 0 || cy + i >= input.rows)
						continue;
					if (fcx + j < 0 || fcx + j >= filter.cols || fcy + i < 0 || fcy + i >= filter.rows)
						continue;

					value += input.at<uchar>(cy + i, cx + j) * filter.at<double>(fcy + i, fcx + j);
				}
				output.at<double>(cy, cx) = value;
			}
		}
	}

	return output;
}
Mat CVLAB::GRADIENT(Mat input)
{
	if (input.channels() == 3)
		input = GRAY(input);
	Mat output(input.rows, input.cols, CV_64FC2);
	double difference[] = { -1,0,1 };
	Mat edge_x(1, 3, CV_64FC1, difference);
	Mat edge_y(3, 1, CV_64FC1, difference);
	Mat grad_x = CONV(input, edge_x);
	Mat grad_y = CONV(input, edge_y);
	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			output.at<Vec2d>(y, x) = Vec2d(grad_x.at<double>(y, x) / 2, grad_y.at<double>(y, x) / 2);
		}
	}

	return output;
}
Mat CVLAB::MAGNITUDE(Mat gradient)
{
	Mat result(gradient.rows, gradient.cols, CV_64FC1);

	for (int y = 0; y < gradient.rows; y++)
	{
		for (int x = 0; x < gradient.cols; x++)
		{
			double fx = gradient.at<Vec2d>(y, x)[0];
			double fy = gradient.at<Vec2d>(y, x)[1];

			result.at<double>(y, x) = std::sqrt(fx * fx + fy * fy);
		}
	}

	return result;
}
Mat CVLAB::PHASE(Mat gradient)
{
	Mat result(gradient.rows, gradient.cols, CV_64FC1);

	for (int y = 0; y < gradient.rows; y++)
	{
		for (int x = 0; x < gradient.cols; x++)
		{
			double fx = gradient.at<Vec2d>(y, x)[0];
			double fy = gradient.at<Vec2d>(y, x)[1];

			result.at<double>(y, x) = atan2(fy, fx);
		}
	}

	return result;
}
Mat CVLAB::NORMALIZE(Mat input, double range)
{
	int type = input.type();
	if (type == CV_64FC1)
	{
		Mat output(input.rows, input.cols, type);
		if (range != 0)
		{
			double max = 0;
			double min = 0;
			double ratio = range;  // normalize factor
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					double value = input.at<double>(y, x);
					if (value > max)
						max = value;
					else if (value < min)
						min = value;
				}
			}
			if (max != min)
				ratio = range / (max - min);
			
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					output.at<double>(y, x) = input.at<double>(y, x) * ratio;
				}
			}
			return output;
		}
		else
		{
			double value = 0;
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					value += input.at<double>(y, x)* input.at<double>(y, x);
				}
			}
			value = std::sqrt(value);
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					output.at<double>(y, x) = input.at<double>(y, x) / value;
				}
			}
			return output;
		}
	}
	else if(type == CV_64FC2)
	{
		Mat output(input.rows, input.cols, type);
		if (range != 0)
		{
			double xmax = 0;
			double xmin = 0;
			double ymax = 0;
			double ymin = 0;
			double xratio = range;  // normalize factor
			double yratio = range;
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					double xvalue = input.at<Vec2d>(y, x)[0];
					if (xvalue > xmax)
						xmax = xvalue;
					else if (xvalue < xmin)
						xmin = xvalue;

					double yvalue = input.at<Vec2d>(y, x)[1];
					if (yvalue > ymax)
						ymax = yvalue;
					else if (yvalue < ymin)
						ymin = yvalue;
				}
			}
			if (xmax != xmin)
				xratio = range / (xmax - xmin);
			if (ymax != ymin)
				yratio = range / (ymax - ymin);

			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					output.at<Vec2d>(y, x)[0] = input.at<Vec2d>(y, x)[0] * xratio;
					output.at<Vec2d>(y, x)[1] = input.at<Vec2d>(y, x)[1] * yratio;
				}
			}
			return output;
		}
		else
		{
			double value = 0;
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					value += input.at<Vec2d>(y, x)[0] * input.at<Vec2d>(y, x)[0] + input.at<Vec2d>(y, x)[1] * input.at<Vec2d>(y, x)[1];
				}
			}
			value = std::sqrt(value);
			for (int y = 0; y < input.rows; y++)
			{
				for (int x = 0; x < input.cols; x++)
				{
					output.at<Vec2d>(y, x)[0] = input.at<Vec2d>(y, x)[0] / value;
					output.at<Vec2d>(y, x)[1] = input.at<Vec2d>(y, x)[1] / value;
				}
			}
			return output;
		}
		return output;
	}
}
void CVLAB::NORMALIZE(double* input, double inputsize, double range)
{
	if (range != 0)
	{
		double max = input[0];
		double min = input[0];
		double ratio = range;  // normalize factor
		for (int i = 0; i < inputsize; i++)
		{
			if (input[i] > max)
				max = input[i];
			if (input[i] < min)
				min = input[i];
		}
		if (max != min)
			ratio = range / (max - min);
		else
			return;
		for (int i = 0; i < inputsize; i++)
		{
			input[i] = (input[i]-min) * ratio;
		}
	}
	else
	{
		double value = 0;
		for (int i = 0; i < inputsize; i++)
			value += input[i] * input[i];
		if (value != 0)
		{
			value = std::sqrt(value);
			for (int i = 0; i < inputsize; i++)
				input[i] = input[i] / value;
		}
	}
}
double CVLAB::SIMILARITY(double* input1, double* input2, int size, int type)
{
	double similarity = 0;
	if (type == 0)
	{
		double inner = 0;
		double anorm = 0;
		double bnorm = 0;
		for (int i = 0; i < size; i++)
		{
			inner += input1[i] * input2[i];
			anorm += input1[i] * input1[i];
			bnorm += input2[i] * input2[i];
		}
		if (anorm == 0 || bnorm == 0)
			similarity = 0;
		else
		{
			similarity = inner / (std::sqrt(anorm) * sqrt(bnorm));
		}
	}
	else if (type == 1)
	{
		double distance = 0;
		for (int i = 0; i < size; i++)
			distance += (input1[i] - input2[i]) * (input1[i] - input2[i]);

		similarity = 1 / std::sqrt(distance);
	}

	return similarity;
}

void MOUSEINF(int event, int x, int y, int flags, void* MouseData)
{
	int** inf = (int**)MouseData;

	*inf[0] = event;
	*inf[1] = x;
	*inf[2] = y;
	*inf[3] = flags;

	return;
}
