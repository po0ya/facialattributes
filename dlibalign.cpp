#include <detect.h>
#include <dlib/geometry/point_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>

#include"confs.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/all/source.cpp>
#include <stddef.h>
#include <utils.hpp>
#include <iostream>
#include "types.hpp"
#include "dlibalign.hpp"
using namespace dlib;
using namespace cv;
using namespace std;

#define SHAPEDAT "/cfarhomes/pouya/glue_scratch/eclipseProj/dlib-18.17/shape_predictor_68_face_landmarks.dat"

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

 std::vector<dlib::point> basePoints;
unsigned long int alignInds[] = { 36, 39, 42, 45, 48, 54 };

dlib::shape_predictor sp;
void init_basePoint() {
	if (basePoints.size() == 0) {
		dlib::deserialize(SHAPEDAT) >> sp;
		basePoints.push_back(dlib::point(25.034 * 2 - 17, 34.158 * 2 + 7));
		basePoints.push_back(dlib::point(34.18 * 2 - 17, 34.16 * 2 + 7));
		basePoints.push_back(dlib::point(44.19 * 2 - 17, 34.09 * 2 + 7));
		basePoints.push_back(dlib::point(53.46 * 2 - 17, 33.80 * 2 + 7));
		basePoints.push_back(dlib::point(31.14 * 2 - 17, 53.02 * 2 + 7));
		basePoints.push_back(dlib::point(47.87 * 2 - 17, 52.8 * 2 + 7));
	}
}

cv::Mat cropAlignFace(Mat& src,Rect& faceRect) {
	init_basePoint();
	auto s = getStandardSize(src);
	auto resized = Mat(s.height, s.width, src.type());
	cv::resize(src, resized, s);
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

	Mat selectedSrc = allRotations[maxInd];
	Mat res;
	if (parts.size() == 0)
		return res;
	cv_image<bgr_pixel> img = cv_image<bgr_pixel>(selectedSrc);
	dlib::rectangle facedlibrect(parts[0].x, parts[0].y,
			parts[0].x + parts[0].width, parts[0].y + parts[0].height);
	full_object_detection shape = sp(img, facedlibrect);

	std::vector<dlib::point> srcPoints;
	for (int i = 0; i < 6; i++) {
		srcPoints.push_back(shape.part(alignInds[i]));
	}

	point_transform_affine transform = find_similarity_transform(srcPoints,
			basePoints);
	auto m = transform.get_m();
	auto b = transform.get_b();
	Mat rotScale = (Mat_<float>(2, 2) << m(0, 0), m(0, 1), m(1, 0), m(1, 1));
	Mat translate = (Mat_<float>(2, 1) << b(0), b(1));

	Mat finalTrnasform(2, 3, rotScale.type());

	hconcat(rotScale, translate, finalTrnasform);
	point upper(parts[0].x,parts[0].y);
	point lower(parts[0].x+parts[0].width,parts[0].y+parts[0].height);

	auto transformedUpper = transform(upper);
	auto transformedLower = transform(lower);

	faceRect = Rect(transformedUpper.x(),transformedUpper.y(),transformedLower.x()- transformedUpper.x(),transformedLower.y()-transformedUpper.y());
	warpAffine(selectedSrc, selectedSrc, finalTrnasform, src.size());
	Rect region(0, 0, IMG_W, IMG_H);
	return selectedSrc(region);

}

std::vector<int> left_eye { 18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42 };
std::vector<int> right_eye { 23, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48 };
std::vector<int> nose { 28, 29, 30, 31, 32, 33, 34, 35, 36 };
std::vector<int> mouth { 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
		63, 64, 65, 66 };
std::vector<int> jaws_and_chin { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
		15, 16, 17 };
std::vector<int> left_eye_interior { 37, 38, 39, 40, 41, 42 };
std::vector<int> right_eye_interior { 43, 44, 45, 46, 47, 48 };

cv::Mat getPart(cv::Mat aligned, FaceComponent fc,cv::Rect faceRect,cv::Rect& outRect) {
	dlib::rectangle facedlibrect(faceRect.x,faceRect.y,faceRect.x+faceRect.width,faceRect.y+faceRect.height);
	//dlib::rectangle facedlibrect(0,0,IMG_W,IMG_H);
	cv_image<bgr_pixel> img = cv_image<bgr_pixel>(aligned);
	full_object_detection shape = sp(img, facedlibrect);
	if(shape.num_parts()==0){
		Mat dummy;
		return dummy;
	}
	std::vector<int> selectedLands;
	int offsets[] = { 4, 4, 4, 4 }; //{up,down,left,right}
	std::vector<std::vector<int>> impParts;
	switch (fc) {
	case BOTH_EYES:
		impParts.push_back(left_eye);
		impParts.push_back(right_eye);
		offsets[UPIND] -= 4;
		offsets[DOWNIND] += 4;
		break;
	case NOSE:
		impParts.push_back(nose);
		offsets[LEFTIND] += 2;
		offsets[RIGHTIND] += 2;
		offsets[UPIND] -= 2;
		break;
	case MOUTH:
		impParts.push_back(mouth);
		offsets[UPIND] += 8;
		break;
	case HAIR:
		impParts.push_back(left_eye);
		impParts.push_back(right_eye);
		impParts.push_back(nose);
		impParts.push_back(mouth);
		break;
	case INNER:
		impParts.push_back(left_eye);
		impParts.push_back(right_eye);
		impParts.push_back(nose);
		impParts.push_back(mouth);
		break;
	case FULL:
		return aligned;
		break;
	}
	std::vector<float> allx;
	std::vector<float> ally;

	for (auto it = impParts.begin(); it < impParts.end(); it++) {
		for (auto itt = it->begin(); itt < it->end(); itt++) {
			allx.push_back(shape.part(*itt-1).x());
			ally.push_back(shape.part(*itt-1).y());
		}
	}

	float x0 = *min_element(allx.begin(), allx.end()) - offsets[LEFT];
	x0 = x0 < 0 ? 0 : x0;
	float x1 = *max_element(allx.begin(), allx.end()) + offsets[RIGHT];
	x1 = x1 > IMG_W ? IMG_W : x1;
	float y0 = *min_element(ally.begin(), ally.end()) - offsets[UP];
	y0 = y0 < 0 ? 0 : y0;
	float y1 = *max_element(ally.begin(), ally.end()) + offsets[DOWN];
	y1 = y1 > IMG_H ? IMG_H : y1;
	Rect region(x0,y0,x1-x0,y1-y0);

	outRect = Rect(x0,y0,x1-x0,y1-y0);
	if(fc == HAIR){
		Mat outter = aligned.clone();
		outter(region) = Mat::zeros(region.height,region.width,outter.type());
		return outter;
	} else {
		return aligned(region);
	}
}

