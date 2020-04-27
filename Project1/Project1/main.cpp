#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv_dnn_centerface.h"

using namespace cv;
using namespace std;

int main() {
	string model_path = "D:/Joestar/CFace/Project1/x64/Release/models/onnx/centerface.onnx";
	
	VideoCapture cap;
	cap.open(0);
	
	if (!cap.isOpened())
		return 0;
	
	int w = cap.get(CAP_PROP_FRAME_WIDTH);
	int h = cap.get(CAP_PROP_FRAME_HEIGHT);

	Mat frame;
	Centerface centerface(model_path, 640, 480);
	while (cap.isOpened())
	{
		cap >> frame;
		if (frame.empty()) {
			continue;
		}
		vector<FaceInfo> face_info;

		centerface.detect(frame, face_info);

		for (int i = 0; i < face_info.size(); i++) {
			rectangle(frame, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2), cv::Scalar(0, 255, 0), 2);
			for (int j = 0; j < 2; j++) {
				circle(frame, cv::Point(face_info[i].landmarks[2 * j], face_info[i].landmarks[2 * j + 1]), 2, cv::Scalar(255, 255, 0), 2);
			}
		}
		cv::imshow("out", frame);
		if (waitKey(20) > 0)
			break;
	}
	cap.release();
	destroyAllWindows();
	return 0;
}
