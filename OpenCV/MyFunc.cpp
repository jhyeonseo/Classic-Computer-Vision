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
Mat CVLAB::GRAY(Mat img, int x, int y, int BLK)
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

			double value = (img.at<Vec3b>(j, i)[2] + img.at<Vec3b>(j, i)[0] + img.at<Vec3b>(j, i)[1]) / 3;
			gray.at<Vec3b>(j, i)[0] = gray.at<Vec3b>(j, i)[1] = gray.at<Vec3b>(j, i)[2] = round(value);
		}
	}

	return gray;
}
Mat CVLAB::RESIZE(Mat img, double scalor, int option)
{
	scalor = sqrt(scalor);
	int width = round(img.cols * scalor);
	int height = round(img.rows * scalor);
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
Mat CONV(Mat input, Mat filter) // size=(x_size, y_size)
{
	Mat output = Mat::zeros(input.rows, input.cols, CV_64FC1); // FOR GRAY IMAGE

	for (int cx = 0; cx < input.cols; cx++)
	{
		for (int cy = 0; cy < input.rows; cy++)
		{
			double value = 0;
			for (int i = 0; i < filter.cols; i++)
			{
				for (int j = 0; j < filter.rows; j++)
				{
					int x_left = cx - filter.cols / 2;
					int y_bot = cy - filter.rows / 2;
					if (x_left + i < 0 || x_left + i >= input.cols || y_bot + i < 0 || y_bot + i >= input.rows)
						continue;

					value += input.at<uchar>(y_bot + j, x_left + i) * filter.at<double>(j, i);
				}
				output.at<double>(cy, cx) = value;
			}
		}
	}

	return output;
}
Mat NORMALIZE(Mat input)
{
	double max = 0;
	double min = 0;
	Mat output(input.rows, input.cols, CV_8UC1);

	for (int i = 0; i < input.cols; i++)
	{
		for (int j = 0; j < input.rows; j++)
		{
			double value = input.at<double>(j, i);

			if (value > max)
				max = value;
			else if (value < min)
				min = value;
		}
	}

	double ratio = 255 / (max - min);

	for (int i = 0; i < input.cols; i++)
	{
		for (int j = 0; j < input.rows; j++)
		{
			output.at<uchar>(j, i) = input.at<double>(j, i) * ratio;
		}
	}

	return output;
}
Mat MAG(Mat fx, Mat fy)
{
	Mat result(fx.rows, fx.cols, CV_64FC1);

	for (int i = 0; i < fx.cols; i++)
	{
		for (int j = 0; j < fx.rows; j++)
		{
			double value = std::sqrt(fx.at<double>(j, i) * fx.at<double>(j, i) + fy.at<double>(j, i) * fy.at<double>(j, i));
			//printf("%f\n", value);
			result.at<double>(j, i) = value * value;
		}
	}

	return result;
}
Mat PHASE(Mat fx, Mat fy)
{
	Mat result(fx.rows, fx.cols, CV_64FC1);

	for (int i = 0; i < fx.cols; i++)
	{
		for (int j = 0; j < fx.rows; j++)
		{
			double angle = atan2(fy.at<double>(j, i), fx.at<double>(j, i));
			angle = angle * (180 / 3.141592);

			if (angle < 0)
				angle = angle + 180;
			
			result.at<double>(j, i) = angle;
		}
	}

	return result;
}