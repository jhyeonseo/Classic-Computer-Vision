#include "MyFunc.h"

CVLAB::CVLAB()
{
}
void CVLAB::Insert(Mat img)
{
	std::vector<Mat> vec;
	vec.push_back(img);
	this->storage.push_back(vec);
}
void CVLAB::Editor()
{
	char command;
	std::cin >> command;

	int x, y, flags;
	int* event = new int[this->storage.size()];
	void*** MouseData = new void** [this->storage.size()];
	for (int i = 0; i < this->storage.size(); i++)
		MouseData[i] = new void* [4];

	if (command == 'p')
	{
		for (int i = 0; i < this->storage.size(); i++)
		{
			event[i] = -1;
			MouseData[i][0] = &event[i];
			MouseData[i][1] = &x;
			MouseData[i][2] = &y;
			MouseData[i][3] = &flags;
			cv::namedWindow(std::to_string(i), 0);
			setMouseCallback(std::to_string(i), MOUSEINF, MouseData[i]);
		}
		
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
		cv::namedWindow("GRAY", 0);

		for (int i = 0; i < this->storage.size(); i++)
		{
			event[i] = -1;
			MouseData[i][0] = &event[i];
			MouseData[i][1] = &x;
			MouseData[i][2] = &y;
			MouseData[i][3] = &flags;
			setMouseCallback("GRAY", MOUSEINF, MouseData[i]);
		}

		while (1)
		{
			waitKey(1);

			for (int i = 0; i < this->storage.size(); i++)
			{
				if (event[i] == EVENT_MOUSEMOVE)
				{
					GRAY(this->storage[i][0], x, y, 100);
				}
				event[i] = -1;
			}
		}
	}
	else if (command == 's')
	{
		// CALL RESIZE FUNCTION!
	}

	delete[]event;
	for (int i = 0; i < this->storage.size(); i++)
		delete[]MouseData[i];
	delete[]MouseData;
	
}
void CVLAB::GRAY(Mat img, int x, int y, int BLK)
{
	Mat gray = img.clone();


	for (int i = x - BLK; i < x + BLK; i++)
	{
		if (i<0 || i>img.cols)
			continue;

		for (int j = y - BLK; j < y + BLK; j++)
		{
			if (j<0 || i>img.rows)
				continue;

			int value = (img.at<Vec3b>(j, i)[2] + img.at<Vec3b>(j, i)[0] + img.at<Vec3b>(j, i)[1]) / 3;

			gray.at<Vec3b>(j, i)[0] = gray.at<Vec3b>(j, i)[1] = gray.at<Vec3b>(j, i)[2] = value;
		}
	}

	imshow("GRAY", gray);
	waitKey(1);
}
Mat CVLAB::RESIZE(Mat img,  double scalor, int option)
{
	scalor = sqrt(scalor * scalor);
	int x = round(img.cols * scalor);
	int y = round(img.rows * scalor);
	double ratio_x = (double)img.rows / (double)x;
	double ratio_y = (double)img.cols / (double)y;
	Mat resize(y, x, CV_8UC3);

	for (int i = 0; i < x; i++)
	{
		int i_orgin = std::floor(i * ratio_x);
		if (i_orgin + 1 == img.cols)
			break;

		for (int j = 0; j < y; j++)
		{
			int j_orgin = std::floor(j * ratio_y);
			if (j_orgin + 1 == img.rows)
				break;

			Vec3d value = 0;
			if (option == 0)
			{
				Vec3d value1, value2, value3, value4 = 0;
				value1 += (Vec3d)img.at<Vec3b>(j_orgin, i_orgin) * (1 - (i * ratio_x - i_orgin));
				value1 += (Vec3d)img.at<Vec3b>(j_orgin, i_orgin + 1) * (i * ratio_x - i_orgin);
				value1 = value1 * (1 - (j * ratio_y - j_orgin));
				value2 += (Vec3d)img.at<Vec3b>(j_orgin + 1, i_orgin) * (1 - (i * ratio_x - i_orgin));
				value2 += (Vec3d)img.at<Vec3b>(j_orgin + 1, i_orgin + 1) * (i * ratio_x - i_orgin);
				value2 = value2 * (j * ratio_y - j_orgin);
				value3 += (Vec3d)img.at<Vec3b>(j_orgin, i_orgin) * (1 - (j * ratio_y - j_orgin));
				value3 += (Vec3d)img.at<Vec3b>(j_orgin + 1, i_orgin) * (j * ratio_y - j_orgin);
				value3 = value3 * (1 - (i * ratio_x - i_orgin));
				value4 += (Vec3d)img.at<Vec3b>(j_orgin, i_orgin + 1) * (1 - (j * ratio_y - j_orgin));
				value4 += (Vec3d)img.at<Vec3b>(j_orgin + 1, i_orgin + 1) * (j * ratio_y - j_orgin);
				value4 = value4 * (i * ratio_x - i_orgin);

				value = (value1 + value2 + value3 + value4) / 2;
			}
			else if (option == 1)
			{
				double normalize = 0;
				value += (Vec3d)img.at<Vec3b>(j_orgin, i_orgin) * std::sqrt(std::pow(i * ratio_x - i_orgin, 2) + std::pow(j * ratio_y - j_orgin, 2));
				value += (Vec3d)img.at<Vec3b>(j_orgin, i_orgin + 1) * std::sqrt(std::pow(i * ratio_x - i_orgin + 1, 2) + std::pow(j * ratio_y - j_orgin, 2));
				value += (Vec3d)img.at<Vec3b>(j_orgin + 1, i_orgin) * std::sqrt(std::pow(i * ratio_x - i_orgin, 2) + std::pow(j * ratio_y - j_orgin + 1, 2));
				value += (Vec3d)img.at<Vec3b>(j_orgin + 1, i_orgin + 1) * std::sqrt(std::pow(i * ratio_x - i_orgin + 1, 2) + std::pow(j * ratio_y - j_orgin + 1, 2));
				normalize += std::sqrt(std::pow(i * ratio_x - i_orgin, 2) + std::pow(j * ratio_y - j_orgin, 2));
				normalize += std::sqrt(std::pow(i * ratio_x - i_orgin + 1, 2) + std::pow(j * ratio_y - j_orgin, 2));
				normalize += std::sqrt(std::pow(i * ratio_x - i_orgin, 2) + std::pow(j * ratio_y - j_orgin + 1, 2));
				normalize += std::sqrt(std::pow(i * ratio_x - i_orgin + 1, 2) + std::pow(j * ratio_y - j_orgin + 1, 2));
				value = value / normalize;
			}
			else if (option == 2)
			{
				int neighbor_x = round(i * ratio_x);
				int neighbor_y = round(j * ratio_y);

				value = (Vec3d)img.at<Vec3b>(neighbor_y, neighbor_x);
			}

			resize.at<Vec3b>(j, i) = (Vec3b)value;
		}
	}

	imwrite("Bi-Linear.bmp", resize);
	return resize;
}
void CVLAB::ROTATE(Mat img, double angle, int option)
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

			if (x_orgin < 0.0 || x_orgin >= (double)img.cols || y_orgin < 0.0 || y_orgin >= (double)img.rows)
				continue;

			int x_left = floor(x_orgin);
			int x_right = x_left + 1;
			int y_bot = floor(y_orgin);
			int y_top = y_bot + 1;
			
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
			
			rotate.at<Vec3b>(y, x) = (Vec3b)value;
		}
	}


	imshow("ROTATE", rotate);
	waitKey(0);
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
void PixelValue(Mat img, int x, int y)
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
