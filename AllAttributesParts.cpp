/*
 * AllAttributesParts.cpp
 *
 *  Created on: Aug 19, 2015
 *      Author: pouya
 */

#include "types.hpp"
#include "detect.h"
#include<iostream>
#include"utils.hpp"
#include"confs.h"
using namespace std;
using namespace cv;
const string AllAttributesParts::attributeNames[] = { "Arched_Eyebrows",
		"Asian", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
		"Black", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows",
		"Child", "Chubby", "Curly_Hair", "Double_Chin", "Eyeglasses",
		"Flushed_Face", "Goatee", "Gray_Hair", "Indian", "Male", "Middle_Aged",
		"Mustache", "Narrow_Eyes", "No_Beard", "No_Eyewear",
		"Obstructed_Forehead", "Oval_Face", "Pale_Skin",
		"Partially_Visible_Forehead", "Pointy_Nose", "Receding_Hairline",
		"Round_Face", "Round_Jaw", "Senior", "Shiny_Skin", "Sideburns",
		"Straight_Hair", "Strong_Nose-Mouth_Lines", "Sunglasses", "Wavy_Hair",
		"White", "Youth" };

AllAttributesParts::AllAttributesParts(vector<string> allClassifiers) {
	this->numberOfCls = allClassifiers.size();
	this->classifiers = new Classifier*[numberOfCls];
	for (int i = 0; i < this->numberOfCls; i++) {
		this->classifiers[i] = new Classifier(allClassifiers[i]);
		string fname = getFileNeme(allClassifiers[i].c_str());
		auto splits = split(fname, 'x');
		auto attrStr = splits[0].substr(1);
		int scale = stoi(splits[2]);
		int attr = stoi(attrStr);
		this->classifiers[i]->scale = scale;
		auto it = sortedClassifiers[attr].begin();
		auto end = sortedClassifiers[attr].end();
		for (; it != end; it++) {
			if ((*it)->weight > this->classifiers[i]->weight) {
				sortedClassifiers[attr].insert(it, this->classifiers[i]);
			}
		}
		if (it == sortedClassifiers[attr].end())
			sortedClassifiers[attr].insert(it, this->classifiers[i]);
	}

	for (int i = 0; i < 44; i++) {
		int size =
				sortedClassifiers[i].size() < 5 ?
						sortedClassifiers[i].size() : 5;
		auto it = sortedClassifiers[i].begin();
		for (int j = 0; j < size; j++) {
			selectedClassifier[i].push_back(*it);
			it++;
		}
		//clean up unused ones
		while (it != sortedClassifiers[i].end()) {
			delete *it;
			it++;
		}
	}
}
AllAttributesParts::AllAttributesParts(vector<string> allClassifiers,
		vector<string> svmPaths) {
	this->numberOfCls = allClassifiers.size();
	this->classifiers = new Classifier*[numberOfCls];
	for (int i = 0; i < this->numberOfCls; i++) {
		this->classifiers[i] = new Classifier(allClassifiers[i], svmPaths[i]);
	}
}

float* AllAttributesParts::getAllAttributes(cv::Mat img) {
	float* ret = new float[this->numberOfCls];
	cv::Mat alignedImg = alignCropFace(img);
	for (int i = 0; i < 44; i++) {
		ret[i] = 0;
	}
	vector<Mat> feats;
	map<int, int> partScaleFeat[6];
	float val;
	int i = 0;
	Mat feat;
	for (int i = 0; i < 44; i++) {
		for (auto it = selectedClassifier[i].begin();
				it < selectedClassifier[i].end(); it++) {
			try {
				feat = feats[partScaleFeat[classifiers[i]->fc - 2].at(
						classifiers[i]->scale)];
			} catch (std::out_of_range e) {
				partScaleFeat[classifiers[i]->fc - 2][classifiers[i]->scale] =
						feats.size();
				classifiers[i]->getFeats(alignedImg,feat);
				feats.push_back(feat);
			}
			ret[i] += this->classifiers[i]->predict(feat)
					* this->classifiers[i]->weight;
		}
		ret[i]/=selectedClassifier[i].size();
	}
	return ret;
}

AllAttributesParts::~AllAttributesParts() {
	for (int i = 0; i < 44; i++) {
		for (int j = 0; j < selectedClassifier[i].size(); j++)
			for (auto it = selectedClassifier[j].begin();
					it < selectedClassifier[j].end(); it++)
				delete *it;
	}
	delete this->classifiers;
}

