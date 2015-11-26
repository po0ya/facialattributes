/*
 * utils.cpp
 *
 *  Created on: Jun 29, 2015
 *      Author: pouya
 */

#include<opencv/cv.h>
#include<iostream>
#include<fstream>
#include"confs.h"
#include"utils.hpp"
#include"stdlib.h"
#include"time.h"
#include"types.hpp"
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#define FLOAT_THRESHOLD 1e-6

cv::Size getStandardSize(cv::Mat s) {

	int height = s.size[0];
	int width = s.size[1];
	int newHeight, newWidth;
	if (height < MAX_SIZE && width < MAX_SIZE) {
		newHeight = height;
		newWidth = width;
	} else {
		if (width > height) {
			newHeight = (MAX_SIZE / float(width)) * height;
			newWidth = MAX_SIZE;
		} else {
			newWidth = (MAX_SIZE / float(height)) * width;
			newHeight = MAX_SIZE;
		}
	}
	return cv::Size(newWidth, newHeight);
}
cv::Size getStandardSize2(int width,int height) {

	int newHeight, newWidth;
	if (height<100 || width <100)
	if (height < MAX_SIZE/2 || width < MAX_SIZE/2) {
		newHeight = height;
		newWidth = width;
	} else {
		if (width > height) {
			newHeight = (MAX_SIZE / float(width)) * height;
			newWidth = MAX_SIZE;
		} else {
			newWidth = (MAX_SIZE / float(height)) * width;
			newHeight = MAX_SIZE;
		}
	}
	return cv::Size(newWidth, newHeight);
}
double distPoints(cv::Point2f p1,cv::Point2f p2){
	return sqrt(pow((p1.x-p2.x),2)+pow(p1.y-p2.y,2));
}
std::string getFileNeme(const char* pathChar) {
	std::string path(pathChar);
	auto sepBSlash = path.find_last_of('/');
	auto sepDot = path.find_last_of('.');

	return path.substr(sepBSlash+1,sepDot-sepBSlash-1);
}

std::string getFileNemePartial(const char* pathChar) {
	std::string path(pathChar);
	auto sepBSlash = path.find_last_of('/');
	auto sepDot = path.find_last_of('_');

	return path.substr(sepBSlash+1,sepDot-sepBSlash-1);
}

std::string getFileDir(const char* pathChar) {
	std::string path(pathChar);
	auto sepBSlash = path.find_last_of('/');
	return path.substr(0,sepBSlash);
}

void thresholdFloatVec(std::vector<float> floats){
	for(size_t i=0;i<floats.size();i++){
		if(floats[i]<FLOAT_THRESHOLD)
			floats[i]=0;
	}
}

void eigenPCA(Mat data, PCA& destPCA){

}


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void drawRect(Mat&img,Point2f center){
	rectangle(img,Rect(center.x-1,center.y-1,10,10),(255,0,0));
}
cv::Point2f mulPointMat(cv::Mat M, const cv::Point2f& p)
{
    cv::Mat_<double> src(3/*rows*/,1 /* cols */);

    src(0,0)=p.x;
    src(1,0)=p.y;
    src(2,0)=1.0;

    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA
    return cv::Point2f(dst(0,0),dst(1,0));
}

void predictAll(string src,string dst,AllAttributes& allAttributes){
	//Open files
	ifstream srcFile;
	srcFile.open(src.c_str());
	ofstream dstFile;
	dstFile.open(dst.c_str());
	while(srcFile.good()){
		string imgPath;

		getline(srcFile,imgPath);
		if(imgPath.find(".jpg")==string::npos)
			continue;
		try{
		Mat img = imread(imgPath,CV_LOAD_IMAGE_COLOR);
		float * res = allAttributes.getAllAttributes(img);
		dstFile << imgPath <<",";
		for(size_t i=0;i<allAttributes.numberOfCls;i++){
			float f = floor(res[i] * 100000)/100000;
			dstFile << f;
			if(i!=allAttributes.numberOfCls-1)
				dstFile<<",";
			else
				dstFile<<endl;
		}

		//cout<<"Attributes are extracted for: "<<imgPath<<endl;
		} catch (cv::Exception e){
			cerr<<"Couldn't get attributes for: "<<imgPath<<endl;
		}

	}
	srcFile.close();
	dstFile.close();

}
