/*

 * types.hpp
 *
 *  Created on: Jul 6, 2015
 *      Author: pouya
 */

#ifndef _TYPES_HPP_
#define _TYPES_HPP_
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include"confs.h"
#include<list>

enum FaceComponent{LEFT_EYE=0,RIGHT_EYE=1,BOTH_EYES=2,MOUTH=3,NOSE=4,HAIR=5,INNER=6,FULL=7};
enum FeatureType{HOG,CHOG,LBP,CLBP};

struct TrainingIO{
	std::vector<std::vector<int> > attributes;
	std::vector<std::string> attrsPaths;
	std::vector<cv::Mat> imgs;
	cv::Mat imgsFeats;
	std::vector<std::vector<cv::Rect> > parts;
	std::vector<std::vector<cv::Rect> > negParts;
	std::map<std::string,int> partsPathsMap;
	int avgWidth;
	int avgHeight;
};

struct TrainingData{
	cv::Mat data;
	cv::Mat labels;
};

class Classifier{
	private:
		cv::HOGDescriptor * hg;
	public:
		cv::SVM svm;
		std::string svm_path;
		cv::PCA pca;
		int avgWidth;
		int avgHeight;
		int fc;
		int featureType;
		float weight;
		int isPCA;
		int scale;
		Classifier(std::string& svm_path,cv::PCA& pca,int featureType,int fc,int avgWidth,int avgHeight,int scale=1);
		Classifier(std::string& svm_path,int featureType,int fc,int avgWidth,int avgHeight,int scale=1);
		Classifier(const std::string& fileName);
		Classifier(const std::string& fileName,const std::string&);
		Classifier(const Classifier&);


		bool save(const std::string& fileName);
		std::string getSVMPath();
		bool setSVMPath(std::string);
		float predict(cv::Mat in,int isFeat=1);
		void getFeats(cv::Mat in,cv::Mat& out);
		~Classifier(){
			delete this->hg;
		}

};


class AllAttributesParts {

public:
	static const std::string attributeNames[];
	Classifier ** classifiers;
	std::list<Classifier*> sortedClassifiers[44];
	std::vector<Classifier*> selectedClassifier[44];
	std::vector<cv::Mat> featuresOfParts[6];
	std::map<int,int> scaleFeaturesMap[6];
	int numberOfCls;
	AllAttributesParts(std::vector<std::string> allClassifiers);
	AllAttributesParts(std::vector<std::string> allClassifiers,std::vector<std::string> svmPaths);
	float* getAllAttributes(cv::Mat img);
	cv::Mat allAttributesOfImages(std::vector<cv::Mat> imgs);

	~AllAttributesParts();

};

class AllAttributes {

public:
	static const std::string attributeNames[];
	Classifier ** classifiers;
	int numberOfCls;
	AllAttributes(std::vector<std::string> allClassifiers);
	AllAttributes(std::vector<std::string> allClassifiers,std::vector<std::string> svmPaths);
	float* getAllAttributes(cv::Mat img);
	cv::Mat allAttributesOfImages(std::vector<cv::Mat> imgs);

	~AllAttributes();

};

typedef void (*function)();
#endif /* SRC_STATICS_TYPES_HPP_ */
