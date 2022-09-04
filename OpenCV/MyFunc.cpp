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

	for (int i = 0; i < this->storage.size(); i++)
		imshow(std::to_string(i), this->storage[i][0]);
	int command = waitKey(0);

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
			setMouseCallback(std::to_string(i), MOUSEINF, MouseData[i]);
		}
		
		while (1)
		{
			waitKey(1);

			for (int i = 0; i < this->storage.size(); i++)
			{
				if (event[i] == EVENT_MOUSEMOVE)
				{
					std::cout << "Window " << i << " ";
					PixelValue(this->storage[i][0], x, y);
				}
				event[i] = -1;
			}

		}
	}

	delete[]event;
	for (int i = 0; i < this->storage.size(); i++)
		delete[]MouseData[i];
	delete[]MouseData;
	
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
