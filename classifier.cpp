/*
 * classifier.cpp
 *
 *  Created on: Jul 8, 2015
 *      Author: pouya
 */

#ifndef SRC_CLASSIFIER_CPP_
#define SRC_CLASSIFIER_CPP_

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <types.hpp>
#include <utils.hpp>
#include <lbp/histogram.hpp>
#include <lbp/lbp.hpp>

#include"confs.h"
#include "classifier.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
void trainSVM(cv::Mat imgs, const cv::Mat& labels, CvSVMParams& params,
		const char* savePath) {
	CvSVM SVM;
	SVM.train(imgs, labels, cv::Mat(), cv::Mat(), params);

}

void splitData(TrainingData& all, TrainingData& train, TrainingData& test,
		float fraction) {
	auto n = all.data.rows;
	int trainSize = (n / 2) * fraction;

	Mat pos, neg, posLabel, negLabel;
	pos = all.data.rowRange(0, trainSize);
	neg = all.data.rowRange(n / 2, (n / 2) + trainSize);

	posLabel = all.labels.colRange(0, trainSize);
	negLabel = all.labels.colRange(n / 2, (n / 2) + trainSize);

	vconcat(pos, neg, train.data);
	hconcat(posLabel, negLabel, train.labels);

	Mat post, negt, posLabelt, negLabelt;
	post = all.data.rowRange(trainSize, n / 2);
	negt = all.data.rowRange((n / 2) + trainSize, n);

	posLabelt = all.labels.colRange(trainSize, n / 2);
	negLabelt = all.labels.colRange((n / 2) + trainSize, n);

	vconcat(post, negt, test.data);
	hconcat(posLabelt, negLabelt, test.labels);

}

Classifier::Classifier(std::string& svm_path, cv::PCA& pca, int featureType,
		int fc, int avgWidth, int avgHeight, int scale) {
	this->svm_path = svm_path;
	this->svm.load(svm_path.c_str());
	this->pca = pca;
	this->featureType = featureType;
	this->fc = fc;
	this->avgHeight = avgHeight;
	this->avgWidth = avgWidth;
	this->scale = scale;
	this->hg = new HOGDescriptor(Size(this->avgWidth, this->avgHeight),
			Size(8, 8), Size(8, 8), Size(4, 4), 15, -10.2, true, 64);
	this->isPCA = 1;
}
Classifier::Classifier(std::string& svm_path, int featureType, int fc,
		int avgWidth, int avgHeight, int scale) {
	this->svm_path = svm_path;
	this->svm.load(svm_path.c_str());
	this->pca = pca;
	this->featureType = featureType;
	this->fc = fc;
	this->avgHeight = avgHeight;
	this->avgWidth = avgWidth;
	this->scale = scale;
	this->hg = new HOGDescriptor(Size(this->avgWidth, this->avgHeight),
			Size(8, 8), Size(8, 8), Size(4, 4), 15, -10.2, true, 64);
	this->isPCA = 0;
}

template<typename _Tp> inline void readFileNodeList(const FileNode& fn,
		vector<_Tp>& result) {
	if (fn.type() == FileNode::SEQ) {
		for (FileNodeIterator it = fn.begin(); it != fn.end();) {
			_Tp item;
			it >> item;
			result.push_back(item);
		}
	}
}

float testSVM(SVM &cl, TrainingData data) {
	Mat res;
	Mat testData = data.data;

	cv::Mat predicted(testData.rows, 1, CV_32F);

	for (int i = 0; i < testData.rows; i++) {
		cv::Mat sample = testData.row(i);

		float x = sample.at<float>(0, 0);
		float y = sample.at<float>(0, 1);

		predicted.at<float>(i, 0) = cl.predict(sample);
	}
	Mat transposed;
	transpose(data.labels, transposed);
	auto acc = evaluate(predicted, transposed);
	std::cout << std::endl;
	std::cout << "Accuracy: " << acc;
	return acc;
}

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i, 0);
		float a = actual.at<float>(i, 0);
		if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 && a <= 0.0)) {
			t++;
		} else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}
Classifier::Classifier(const Classifier& cl) {
	this->svm.load(cl.svm_path.c_str());
	this->svm_path = cl.svm_path;
	this->pca = cl.pca;
	this->avgWidth = cl.avgWidth;
	this->avgHeight = cl.avgHeight;
	this->fc = cl.fc;
	this->featureType = cl.featureType;
	this->weight = cl.weight;
	this->isPCA = cl.isPCA;
	this->scale = cl.scale;
}

Classifier::Classifier(const string& fileName) {
	FileStorage fs(fileName, FileStorage::READ);
	fs["svm_path"] >> this->svm_path;
	fs["is_pca"] >> this->isPCA;
	if (this->isPCA) {
		fs["mean"] >> this->pca.mean;
		fs["e_vectors"] >> this->pca.eigenvectors;
		fs["e_values"] >> this->pca.eigenvalues;
	}
	fs["feature_type"] >> this->featureType;

	fs["avgWidth"] >> this->avgWidth;
	fs["avgHeight"] >> this->avgHeight;
	fs["weight"] >> this->weight;
	fs["face_component"] >> this->fc;
	fs["scale"] >> this->scale;
	this->hg = new HOGDescriptor(Size(this->avgWidth, this->avgHeight),
			Size(8, 8), Size(8, 8), Size(4, 4), 15, -10.2, true, 64);
	this->svm.load(this->svm_path.c_str());
	fs.release();
}

Classifier::Classifier(const string& fileName, const string& svmPath) {
	FileStorage fs(fileName, FileStorage::READ);
	fs["svm_path"] >> this->svm_path;
	fs["is_pca"] >> this->isPCA;
	if (this->isPCA) {
		fs["mean"] >> this->pca.mean;
		fs["e_vectors"] >> this->pca.eigenvectors;
		fs["e_values"] >> this->pca.eigenvalues;
	}
	fs["feature_type"] >> this->featureType;

	fs["avgWidth"] >> this->avgWidth;
	fs["avgHeight"] >> this->avgHeight;
	fs["weight"] >> this->weight;
	fs["face_component"] >> this->fc;
	fs["scale"] >> this->scale;

	this->hg = new HOGDescriptor(Size(this->avgWidth, this->avgHeight),
			Size(8, 8), Size(8, 8), Size(4, 4), 15, -10.2, true, 64);
	this->svm_path = svmPath;
	this->svm.load(svmPath.c_str());
	fs.release();
}

std::string Classifier::getSVMPath() {
	return this->svm_path;
}

bool Classifier::setSVMPath(std::string path) {
	try {
		this->svm.load(path.c_str());
		return true;
	} catch (Exception e) {
		return false;
	}
}

bool Classifier::save(const string& fileName) {

	FileStorage fs(fileName, FileStorage::WRITE);
	fs << "svm_path" << this->svm_path;

	fs << "is_pca" << this->isPCA;
	if (this->isPCA) {
		fs << "mean" << this->pca.mean;
		fs << "e_vectors" << this->pca.eigenvectors;
		fs << "e_values" << this->pca.eigenvalues;
	}
	fs << "avgWidth" << this->avgWidth;
	fs << "avgHeight" << this->avgHeight;
	fs << "weight" << this->weight;
	fs << "feature_type" << this->featureType;
	fs << "face_component" << this->fc;
	fs.release();
}

void Classifier::getFeats(cv::Mat in,cv::Mat& out) {

	cv::Mat dest(this->avgHeight, this->avgWidth, in.type());
	vector<cv::Point> locations;
	vector<float> descVals;
	//TODO GET PART

	// RESIZE
	cv::resize(in, dest, Size(this->avgWidth, this->avgHeight));
	vector<Mat> channel(3);
	Mat gray;
	gray.create(this->avgHeight, this->avgWidth, CV_32F);
	cvtColor(dest, gray, COLOR_RGB2GRAY);
	gray.convertTo(gray, CV_8U);
	cv::split(dest, channel);

	if (this->featureType == 1) {
		hg->compute(gray, descVals, Size(0, 0), Size(0, 0), locations);
		for (int j = 0; j < 3; j++) {
			vector<float> channelVals;
			Mat t;
			((Mat) channel[j]).convertTo(t, CV_8U);
			hg->compute(t, channelVals, Size(0, 0), Size(0, 0), locations);
			descVals.insert(descVals.end(), channelVals.begin(),
					channelVals.end());
		}
		thresholdFloatVec(descVals);
		Mat descMat(1, descVals.size(), CV_32FC1, descVals.data());
		out.create(descMat.rows,descMat.cols,descMat.type());
		descMat.convertTo(out,descMat.type());
	} else {

		Mat lbp_image = lbp::ELBP(gray, LBP_RADIUS, LBP_NEIGHBORS);
		Mat query = lbp::spatial_histogram(lbp_image, /* lbp_image */
		static_cast<int>(std::pow(2.0, static_cast<double>(LBP_NEIGHBORS))), /* number of possible patterns */
		8, /* grid size x */
		8, /* grid size y */
		true /* normed histograms */);
		query.convertTo(query, CV_32F);
		Mat descMat = query.clone();
		//		hg.compute(gray, descVals, Size(0, 0), Size(0, 0), locations);
		for (int j = 0; j < 3; j++) {
			vector<float> channelVals;
			channel[j].convertTo(channel[j], CV_8U);

			lbp_image = lbp::ELBP(channel[j], LBP_RADIUS, LBP_NEIGHBORS);
			query = lbp::spatial_histogram(lbp_image, /* lbp_image */
			static_cast<int>(std::pow(2.0, static_cast<double>(LBP_NEIGHBORS))), /* number of possible patterns */
			8, /* grid size x */
			8, /* grid size y */
			true /* normed histograms */);
			query.convertTo(query, CV_32F);
			Mat t = query.clone();
			hconcat(descMat, t, descMat);
		}

		out= descMat.clone();
	}
}

float Classifier::predict(cv::Mat in,int isFeat) {
	Mat descMat;
	if(isFeat==0){
		getFeats(in,descMat);
	} else {
		descMat = in;
	}
	//PCA
	cout<<"DIM: "<<descMat.cols<<endl;
	Mat out;
	Mat vec = descMat.row(0);
	if (isPCA) {
		out.create(descMat.rows, pca.eigenvalues.rows, descMat.type());
		Mat coeffs = out.row(0);
		pca.project(vec, coeffs);
		return this->svm.predict(coeffs);
	} else {
		return this->svm.predict(vec,1);
	}
}

#endif /* SRC_CLASSIFIER_CPP_ */
