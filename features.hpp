/*
 * features.hpp
 *
 *  Created on: Jul 8, 2015
 *      Author: pouya
 */

#ifndef SRC_FEATURES_HPP_
#define SRC_FEATURES_HPP_
#include "types.hpp"
#include <opencv2/core/core.hpp>


void hogForTraining(TrainingIO & io, int fc, int desiredAttr,cv::Mat& outHogs, float*& labelsint,float scale=1);
void lbpForTraining(TrainingIO & io, int fc, int desiredAttr,cv::Mat& outHogs, float*& labelsint,float scale=1);
cv::PCA compressPCA(cv::Mat in, double retainedEnergy, cv::Mat& out);
cv::Mat formatHogForPCA(const std::vector<cv::Mat > &data);

void lbpFromImgs(TrainingIO & io, int desiredAttr, cv::Mat& outHogs,float*& labelsint, float scale = 1);
void getTrainingFeats(TrainingIO & io, int desiredAttr, cv::Mat& outHogs,float*& labelsint);
void putTrainingFeatures(TrainingIO & io, int fc, float scale);
#endif
