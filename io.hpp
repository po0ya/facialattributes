#ifndef __IOO__
#define __IOO__

#include<opencv/cv.h>
#include<iostream>
#include "detect.h"

bool loadCSV(const std::string & csvPath, std::vector<cv::Mat> & outMat,
		std::vector<std::vector<cv::Rect> > & outParts,
		std::map<std::string, int>& paths);
bool loadCSVLFW(const std::string & csvPath, std::vector<cv::Mat> & outMat,
		std::vector<std::vector<cv::Rect> > & outParts,
		std::map<std::string, int>& paths);
bool readAttributesCSV(const std::string &,
		std::vector<std::vector<int> >& attributes,
		std::vector<std::string>& filenameMap);
bool readAttributesCSVLFW(const std::string &,
		std::vector<std::vector<int> >& attributes,
		std::vector<std::string>& filenameMap);
bool loadCSVPartial(const std::string & csvPath, std::vector<cv::Mat> & outMat,
		std::map<std::string, int> & paths);
void loadAll(const std::string& partsPath, const std::string& attrsPath,
		TrainingIO & io);
void loadAllLFW(const std::string& partsPath, const std::string& attrsPath,
		TrainingIO & io);
void loadAllLFWPartial(const std::string& partsPath, const std::string& attrsPath,
		TrainingIO & io);
void savePCA(const std::string &file_name, cv::PCA pca_);
void loadPCA(const std::string &file_name, cv::PCA &pca_);
void saveTrainingData(const std::string& file_name, TrainingData all);
void loadTrainingData(const std::string& file_name, TrainingData& all);
std::string prepareForLFW(std::string name);
#endif
