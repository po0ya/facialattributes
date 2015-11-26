/*
 * detect.cpp
 *
 *  Created on: Jun 25, 2015
 *      Author: Pouya Samangouei
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <stddef.h>
#include <types.hpp>
#include <utils.hpp>
#include "detect.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>

#include"confs.h"
#include <vector>

using namespace std;
using namespace cv;

#define PI 3.14159265
String faceDetectorPath = "statics/ocvhaars/haarcascade_frontalface_default.xml";
String eyesDetectorPath = "statics/ocvhaars/haarcascade_eye_tree_eyeglasses.xml";
String noseDetectorPath = "statics/ocvhaars/haarcascade_mcs_nose.xml";
String mouthDetectorPath = "statics/ocvhaars/haarcascade_mcs_mouth.xml";

CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
CascadeClassifier mouthCascade;
CascadeClassifier noseCascade;

bool cascades_initialized = false;

bool init_detectors() {
	if (cascades_initialized)
		return true;
	if (!faceCascade.load(faceDetectorPath)) {
		printf("Error loading facedetector\n");
		return false;
	}
	if (!eyesCascade.load(eyesDetectorPath)) {
		printf("Error loading eyes detector\n");
		return false;
	}
	if (!noseCascade.load(noseDetectorPath)) {
		printf("Error loading nose detector\n");
		return false;
	}
	if (!mouthCascade.load(mouthDetectorPath)) {
		printf("Error loading mouth detector\n");
		return false;
	}
	cascades_initialized = true;
	return true;
}

Point2f canonicalPoints[] = { Point2f(61, 87), Point2f(39, 75), Point2f(81, 75),
		Point2f(59, 101), Point2f(60, 118) };



Mat alignCropFace(Mat& src) {
	auto s = getStandardSize(src);
	auto resized = Mat(s.height, s.width, src.type());

	resize(src, resized, s);
	Mat second;
	Mat third;
	flip(resized, second, 1);
	transpose(second, second);
	transpose(resized, third);
	Mat allRotations[] = { resized, second, third };
	size_t maxInd = 0;
	size_t maxCount = 0;
	std::vector<Rect> allParts[3];

	for (int ir = 0; ir < 3; ir++) {
		allParts[ir] = detectAll(allRotations[ir]);
		size_t temp = 0;

		for (auto it = allParts[ir].begin(); it != allParts[ir].end(); it++) {
			if (it->width != 0)
				temp++;
		}
		if (temp > maxCount) {
			maxCount = temp;
			maxInd = ir;
		}
	}

	std::vector<Rect> parts = allParts[maxInd];
	src = allRotations[maxInd];
	Mat res;
	if (parts.size() == 0)
		return res;

	std::vector<Point2f> srcPoints;
	std::vector<Point2f> dstPoints;

	size_t pointsCounter = 0;
	for (size_t i = 1; i < parts.size(); i++) {
		if (parts[i].width != 0) {
			dstPoints.push_back(canonicalPoints[i]);
			srcPoints.push_back(
					Point2f(parts[i].x + parts[i].width / 2,
							parts[i].y + parts[i].height / 2));
			pointsCounter++;
		}
	}
	if (pointsCounter == 1) {
		srcPoints.push_back(
				Point2f(parts[0].x + parts[0].width / 2,
						parts[0].y + parts[0].height / 2));
		dstPoints.push_back(canonicalPoints[4]);
	}
	if (pointsCounter == 0)
		return src(parts[0]);

	//first ROTATE:
	auto digDst = atan2(srcPoints[0].y - srcPoints[1].y,
			srcPoints[0].x - srcPoints[1].x);
	auto digSrc = atan2(dstPoints[0].y - dstPoints[1].y,
			dstPoints[0].x - dstPoints[1].x);

	auto rotDigRad = digDst - digSrc;
	double resDig = rotDigRad * 180 / PI;

	auto lenDst = distPoints(srcPoints[0], srcPoints[1]);
	auto lenSrc = distPoints(dstPoints[0], dstPoints[1]);
	Mat rot_mat(2, 3, CV_32FC1);

	rot_mat = getRotationMatrix2D(srcPoints[0], resDig, lenSrc / lenDst);
	res = Mat::zeros(src.rows, src.cols, src.type());
	warpAffine(src, src, rot_mat, src.size());

	cout << rot_mat << endl;
	int sumX = 0;
	int sumY = 0;
	for (int i = 1; i < srcPoints.size(); i++) {
		cout << srcPoints[i] << endl;
		srcPoints[i] = mulPointMat(rot_mat, srcPoints[i]);
		cout << srcPoints[i] << endl;
		sumX += srcPoints[i].x - dstPoints[i].x;
		sumY += srcPoints[i].y - dstPoints[i].y;
	}
	sumX /= srcPoints.size() - 1;
	sumY /= srcPoints.size() - 1;
	res = src(Rect(sumX, sumY, IMG_W, IMG_H));
	//then translate

	return res;
}

Rect detect(Mat& image, FaceComponent comp) {

}
/*
 * Detects all parts and if something is not detected it will be (0,0,0,0)
 */
std::vector<cv::Rect> detectAll(Mat& image) {
	init_detectors();
	Mat intermediate, result;
	std::vector<Rect> faceRects, eyeRects, noseRects, mouthRects;
	Rect leftEye(0, 0, 0, 0);
	Rect rightEye(0, 0, 0, 0);
	Rect nose(0, 0, 0, 0);
	Rect mouth(0, 0, 0, 0);
	/*Mat gray2(image.rows, image.cols, image.type());
	 cvtColor(image, gray2, COLOR_BGR2GRAY);
	 Mat gray(image.rows, image.cols, image.type());
	 equalizeHist(gray2, gray);
	 imshow("s",gray);
	 waitKey(0);*/
	faceCascade.detectMultiScale(image, faceRects);
	//No faces detected
	if (faceRects.size() == 0) {
		std::vector<Rect> dummy;
		return dummy;
	}

	std::vector<std::vector<Rect>> allParts;
	for (int facei = 0; facei < faceRects.size(); facei++) {
		Point2f faceCenter(faceRects[facei].width / 2,
				faceRects[facei].height / 2);

		//Just work on the first detected face
		intermediate = image(faceRects[facei]);
		/*imshow("s",intermediate);
		 waitKey(0);*/
		bool eyesDetected = false;
		bool leftDetected = false;
		bool rightDetected = false;
		eyesCascade.detectMultiScale(intermediate, eyeRects);
		Rect bothEyes;
		for (auto eyeit = eyeRects.begin(); eyeit < eyeRects.end(); eyeit++) {
			if (eyeit->x < faceCenter.x && eyeit->y < faceCenter.y) {
				leftEye = *eyeit;
				leftDetected = true;
			} else if (eyeit->x > faceCenter.x && eyeit->y < faceCenter.y) {
				rightEye = *eyeit;
				rightDetected = true;
			}
		}
		if (leftDetected && rightDetected) {
			bothEyes.x = leftEye.x;
			bothEyes.width = rightEye.x + rightEye.width - leftEye.x;
			bothEyes.y = leftEye.y;
			bothEyes.height =
					leftEye.height > rightEye.height ?
							leftEye.height : rightEye.height;
			eyesDetected = true;
		}

		//LEFT EYE

		//MOUTH
		mouthCascade.detectMultiScale(intermediate, mouthRects);
		for (auto it = mouthRects.begin(); it != mouthRects.end(); it++) {
			bool flag = false;

			if (eyesDetected)
				if (it->x > bothEyes.x && it->y > bothEyes.y + bothEyes.height)
					flag = true;
				else if (leftDetected)
					if (it->x > leftEye.x && it->y > leftEye.y + leftEye.height)
						flag = true;
					else if (rightDetected)
						if (it->x < rightEye.x
								&& it->y > rightEye.y + rightEye.height)
							flag = true;
						else if (it->x < faceCenter.x && it->y > faceCenter.y)
							flag = true;

			if (flag) {
				mouth = *it;

				mouth.x = mouth.x + faceRects[facei].x;
				mouth.y = mouth.y + faceRects[facei].y;
			}
		}

		//NOSE
		noseCascade.detectMultiScale(intermediate, noseRects);
		for (auto it = noseRects.begin(); it != noseRects.end(); it++) {
			bool flag = false;

			if (eyesDetected)
				if (it->x > bothEyes.x
						&& it->y > bothEyes.y + bothEyes.height / 2)
					flag = true;
				else if (leftDetected)
					if (it->x > leftEye.x
							&& it->y > leftEye.y + leftEye.height / 2)
						flag = true;
					else if (rightDetected)
						if (it->x < rightEye.x
								&& it->y > rightEye.y + rightEye.height / 2)
							flag = true;

			if (flag) {
				nose = *it;

				nose.x = nose.x + faceRects[facei].x;
				nose.y = nose.y + faceRects[facei].y;
			}
		}

		if (leftDetected) {
			leftEye.x += faceRects[facei].x;
			leftEye.y += faceRects[facei].y;
		}

		//RIGHT EYE
		if (rightDetected) {
			rightEye.x += faceRects[facei].x;
			rightEye.y += faceRects[facei].y;
		}

		std::vector<Rect> parts(5);
		parts[0] = faceRects[facei];
		parts[1] = leftEye;
		parts[2] = rightEye;
		parts[3] = nose;
		parts[4] = mouth;
		allParts.push_back(parts);
	}

	size_t maxInd = 0;
	size_t maxCount = 0;
	for (int i = 0; i < allParts.size(); i++) {
		size_t temp = 0;
		for (auto it = allParts[i].begin(); it != allParts[i].end(); it++) {
			if (it->width != 0)
				temp++;
		}
		if (temp > maxCount) {
			maxCount = temp;
			maxInd = i;
		}
	}

	return allParts[maxInd];
}

