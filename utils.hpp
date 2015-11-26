#ifndef __UTILS__
#define __UTILS__

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include "types.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
cv::Size getStandardSize(cv::Mat img);
cv::Size getStandardSize2(int width,int height);
std::string getFileNeme(const char* path);
void predictAll(std::string src,std::string dst,AllAttributes& allAttributes);
template<typename T> T ** vec2DtoArr2D(std::vector<std::vector<T> > vec2D) {
	auto row = vec2D.size();
	auto col = vec2D[0].size();

	T ** t2d = new T*[row];

	size_t i, j;
	for (i = 0; i < row; i++) {
		t2d[i] = new T[col];
		for (j = 0; j < col; j++) {

			t2d[i][j] = vec2D[i][j];
			if(t2d[i][j]<1e-5)
				t2d[i][j] = 0;
		}
	}

	return t2d;
}
template<typename T> void print2DArr(T** vec2D, int row, int col) {
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++){
			std::cout << vec2D[j][i] << " ";
		}
		std::cout << std::endl;
	}
}
void drawRect(cv::Mat&img,cv::Point2f center);
double distPoints(cv::Point2f p1,cv::Point2f p2);
cv::Point2f mulPointMat(cv::Mat M, const cv::Point2f& p);
void thresholdFloatVec(std::vector<float> floats);
std::string getFileNemePartial(const char* pathChar);
std::string getFileDir(const char* pathChar);
std::vector<std::string> split(const std::string &s, char delim);
#endif
