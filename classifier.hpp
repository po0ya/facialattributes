/*
 * classifier.hpp
 *
 *  Created on: Jul 8, 2015
 *      Author: pouya
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "types.hpp"
#ifndef SRC_CLASSIFIER_HPP_
#define SRC_CLASSIFIER_HPP_

void trainSVM(std::vector<cv::Mat>& imgs,const cv::Mat& labels,CvSVMParams& params ,const char* savePath);
void testSVM(Classifier cl,TrainingData data);
float evaluate(cv::Mat& predicted, cv::Mat& actual);
float testSVM(cv::SVM &cl,TrainingData data);
void splitData(TrainingData& all,TrainingData& train, TrainingData& test,float fraction);
#endif /* SRC_CLASSIFIER_HPP_ */
