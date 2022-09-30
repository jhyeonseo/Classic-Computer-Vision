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
/*
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

	double** block = new double* [blocknum_total];
	for (int i = 0; i < blocknum_total; i++)
	{
		block[i] = new double[bincount];
		for (int j = 0; j < bincount; j++)
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
						block[blockindex][k] += cell[cellindex + (i * cellnum_x + j)][k];    // Block을 구성하는 cell들의 histogram을 합친다
					}
				}
			}
			for (int k = 0; k < bincount; k++)
				total += block[blockindex][k] * block[blockindex][k];

			for (int k = 0; k < bincount; k++)
				block[blockindex][k] = block[blockindex][k] / std::sqrt(total);
		}
	}

	struct Histogram output;
	output.orgin = input;
	output.data = block;
	output.bincount = bincount;
	output.datacount = blocknum_total;

	this->histogram.push_back(output);
	
}
*/
void CVLAB::MYHOG(Mat input, int bincount, int blocksize, int interval)
{
	Mat grad = GRADIENT(input);
	Mat mag = MAGNITUDE(grad);
	Mat phase = PHASE(grad);
	int blocknum_x = (input.cols - interval) / interval;
	int blocknum_y = (input.rows - interval) / interval;
	double* histogram = new double[bincount * blocknum_x * blocknum_y];
	for (int i = 0; i < bincount * blocknum_x * blocknum_y; i++)
		histogram[i] = 0;

	double bin_interval = 180.0 / bincount;
	int blockindex = 0;

	for (int y = 0; y <= input.rows-blocksize; y+=interval)
	{
		for (int x = 0; x <= input.cols-blocksize; x+=interval)
		{
			// x, y = first index in block
			double total = 0;
			for (int i = y; i < y + blocksize; i++)
			{
				for (int j = x; j < x + blocksize; j++)
				{
					if (i >= input.rows || j >= input.cols)
						continue;
					// i, j = pixel index
					double degree = phase.at<double>(i, j) * 57.2958;
					if (degree < 0.0)
						degree += 180.f;
					int binindex = degree / bin_interval;

					if (binindex >= bincount)
						binindex = bincount - 1;

					histogram[blockindex * bincount + binindex] += mag.at<double>(i, j);
				}
			}
			for (int i = blockindex * bincount; i < blockindex * bincount + bincount; i++)
				total += histogram[i];

			for (int i = blockindex * bincount; i < blockindex * bincount + bincount; i++)
				histogram[i] = histogram[i] / total;                  // Normalize with L2
			
			blockindex++;
		}
	}

	struct Histogram output;
	output.orgin = input;
	output.data = histogram;
	output.bincount = bincount;
	output.datasize = blocknum_x * blocknum_y * bincount;

	this->histogram.push_back(output);
	
	return;
}
Mat CVLAB::HARRIS(Mat input, Size window, double threshold)
{
	Mat output;
	input.copyTo(output);
	Mat grad = NORMALIZE(GRADIENT(input), 1);
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
			{
				output.at<uchar>(y, x) = 255;
			}
		}
	}

	return output;
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
double CVLAB::DISTANCE(Mat base, Mat compare)
{
	double distance = 0;
	for (int y = 0; y < base.rows; y++)
	{
		for (int x = 0; x < base.cols; x++)
		{
			distance += (base.at<double>(y, x) - compare.at<double>(y, x)) * (base.at<double>(y, x) - compare.at<double>(y, x));
		}
	}

	return std::sqrt(distance);
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
