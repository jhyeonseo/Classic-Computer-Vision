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
void CVLAB::HOG(Mat input, int bincount, int cellsize, int blocksize)
{
	Mat grad = GRADIENT(input);
	Mat mag = MAGNITUDE(grad);
	Mat phase = PHASE(grad);

	cellsize = std::sqrt(cellsize);
	blocksize = std::sqrt(blocksize);
	double interval = 180.0 / (double)bincount;

	int cellnum_x = input.cols / cellsize;
	int cellnum_y = input.rows / cellsize;
	int cellnum_total = cellnum_x * cellnum_y;
	double** cell = new double* [cellnum_total];
	for (int i = 0; i < cellnum_total; i++)
	{
		cell[i] = new double[bincount];
		for (int j = 0; j < bincount; j++)
			cell[i][j] = 0;
	}
	int blocknum_x = cellnum_x - blocksize + 1;
	int blocknum_y = cellnum_y - blocksize + 1;
	int blocknum_total = blocknum_x * blocknum_y;
	int block_bincount = bincount * blocksize * blocksize;
	double** block = new double* [blocknum_total];
	for (int i = 0; i < blocknum_total; i++)
	{
		block[i] = new double[block_bincount];
		for (int j = 0; j < block_bincount; j++)
			block[i][j] = 0;
	}


	for (int cellindex_y = 0; cellindex_y < cellnum_y; cellindex_y++)
	{
		int y_first = cellindex_y * cellsize;
		for (int cellindex_x = 0; cellindex_x < cellnum_x; cellindex_x++)
		{
			int x_first = cellindex_x * cellsize;
			int cellindex = cellindex_y * cellnum_x + cellindex_x;

			for (int i = 0; i < cellsize; i++)
			{
				for (int j = 0; j < cellsize; j++)
				{
					double degree = phase.at<double>(y_first + i, x_first + j) * 57.2958;

					if (degree < 0.0)
						degree += 180.f;

					int binindex = (int)round(degree / interval);
					if (binindex >= bincount)
						binindex = bincount - 1;
					cell[cellindex][binindex] += mag.at<double>(y_first + i, x_first + j);
				}
			}
		}
	}

	for (int blockindex_y = 0; blockindex_y < blocknum_y; blockindex_y++)
	{
		for (int blockindex_x = 0; blockindex_x < blocknum_x; blockindex_x++)
		{
			int blockindex = blockindex_y * blocknum_x + blockindex_x;
			int cellindex = blockindex_y * cellnum_x + blockindex_x;
			double total = 0;
			for (int i = 0; i < blocksize; i++)
			{
				for (int j = 0; j < blocksize; j++)
				{
					for (int k = 0; k < bincount; k++)
					{
						block[blockindex][k + (i * blocksize + j) * bincount] += cell[cellindex + (i * cellnum_x + j)][k];
					}
				}
			}
			for (int k = 0; k < block_bincount; k++)
				total += block[blockindex][k] * block[blockindex][k];

			for (int k = 0; k < block_bincount; k++)
				block[blockindex][k] = block[blockindex][k] / std::sqrt(total);
		}
	}

	struct Histogram output;
	output.orgin = input;
	output.data = block;
	output.bincount = block_bincount;
	output.datacount = blocknum_total;

	this->histogram.push_back(output);
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

Mat CVLAB::CONV(Mat input, Mat filter)
{
	Mat output = Mat::zeros(input.rows, input.cols, CV_64FC1);

	for (int cy = 0; cy < input.rows; cy++)
	{
		for (int cx = 0; cx < input.cols; cx++)
		{
			double value = 0;
			for (int i = 0; i < filter.rows; i++)
			{
				for (int j = 0; j < filter.cols; j++)
				{
					int x_first = cx - filter.cols / 2;
					int y_first = cy - filter.rows / 2;
					if (x_first + j < 0 || x_first + j >= input.cols || y_first + i < 0 || y_first + i >= input.rows)
						continue;

					value += input.at<uchar>(y_first + i, x_first + j) * filter.at<double>(i, j);
				}
				output.at<double>(cy, cx) = value;
			}
		}
	}

	return output;
}
Mat CVLAB::GRADIENT(Mat input)
{
	Mat output(input.rows, input.cols, CV_64FC2);

	double data_x[] = { -1,0,1,-1,0,1,-1,0,1 };
	double data_y[] = { -1,-1,-1, 0,0,0, 1,1,1 };
	Mat edge_x(3, 3, CV_64FC1, data_x);
	Mat edge_y(3, 3, CV_64FC1, data_y);
	Mat grad_x = CONV(input, edge_x);
	Mat grad_y = CONV(input, edge_y);

	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			output.at<Vec2d>(i, j) = Vec2d(grad_x.at<double>(i, j), grad_y.at<double>(i, j));
		}
	}

	return output;
}
Mat CVLAB::MAGNITUDE(Mat gradient)
{
	Mat result(gradient.rows, gradient.cols, CV_64FC1);

	for (int i = 0; i < gradient.rows; i++)
	{
		for (int j = 0; j < gradient.cols; j++)
		{
			double fx = gradient.at<Vec2d>(i, j)[0];
			double fy = gradient.at<Vec2d>(i, j)[1];

			result.at<double>(i, j) = std::sqrt(fx * fx + fy * fy);
		}
	}

	return result;
}
Mat CVLAB::PHASE(Mat gradient)
{
	Mat result(gradient.rows, gradient.cols, CV_64FC1);

	for (int i = 0; i < gradient.rows; i++)
	{
		for (int j = 0; j < gradient.cols; j++)
		{
			double fx = gradient.at<Vec2d>(i, j)[0];
			double fy = gradient.at<Vec2d>(i, j)[1];

			result.at<double>(i, j) = atan2(fy, fx);
		}
	}

	return result;
}
Mat CVLAB::NORMALIZE(Mat input)
{
	int type = input.type();
	if (type == CV_64FC1)
	{
		Mat output(input.rows, input.cols, CV_8UC1);
		double max = 0;
		double min = 0;

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
	else
	{
		Mat output(input.rows, input.cols, CV_8UC3);
		/*
		채널수 조정 
		타입 조정
		*/
		return output;
	}
}
double CVLAB::DISTANCE(Mat base, Mat compare, int type)
{
	double difference = 0;

	if (type == 0)
	{
		HOG(base);
		HOG(compare);

		double** basedata = this->histogram[0].data;
		double** comparedata = this->histogram[1].data;

		for (int i = 0; i < this->histogram[0].datacount; i++)
		{
			for (int j = 0; j < this->histogram[0].bincount; j++)
				difference += (basedata[i][j] - comparedata[i][j]) * (basedata[i][j] - comparedata[i][j]);
		}
		this->histogram.pop_back();
		this->histogram.pop_back();
		difference = sqrt(difference);
	}


	return difference;
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
