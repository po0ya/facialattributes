#ifndef __DETECT__
#define __DETECT__

#include<opencv/cv.h>
#include<iostream>
#include"types.hpp"
//parts = {'lefteyewbrow','righteyewbrow','eyeswitheyebrows','mouth','nose','hair','full'};


bool init_detectors();
cv::Rect detect(cv::Mat& image,FaceComponent comp);
cv::vector<cv::Rect> detectAll(cv::Mat& image);
cv::Mat alignCropFace(cv::Mat& src);
#endif
