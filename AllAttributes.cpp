/*
 * AllAttributes.cpp
 *
 *  Created on: Aug 19, 2015
 *      Author: pouya
 */

#include "types.hpp"
#include "detect.h"
#include<iostream>
using namespace std;
const string AllAttributes::attributeNames[]= {"Arched_Eyebrows", "Asian",
		    "Bags_Under_Eyes",
		    "Bald",
		    "Bangs",
		    "Big_Lips",
		    "Big_Nose",
		    "Black",
		    "Black_Hair",
		    "Blond_Hair",
		    "Brown_Hair",
		    "Bushy_Eyebrows",
		    "Child",
		    "Chubby",
		    "Curly_Hair",
		    "Double_Chin",
		    "Eyeglasses",
		    "Flushed_Face",
		    "Goatee",
		    "Gray_Hair",
		    "Indian",
		    "Male",
		    "Middle_Aged",
		    "Mustache",
		    "Narrow_Eyes",
		    "No_Beard",
		    "No_Eyewear",
		    "Obstructed_Forehead",
		    "Oval_Face",
		    "Pale_Skin",
		    "Partially_Visible_Forehead",
		    "Pointy_Nose",
		    "Receding_Hairline",
		    "Round_Face",
		    "Round_Jaw",
		    "Senior",
		    "Shiny_Skin",
		    "Sideburns",
		    "Straight_Hair",
		    "Strong_Nose-Mouth_Lines",
		    "Sunglasses",
		    "Wavy_Hair",
		    "White",
		    "Youth"};

AllAttributes::AllAttributes(vector<string> allClassifiers) {
	this->numberOfCls = allClassifiers.size();
	this->classifiers = new Classifier*[numberOfCls];
	for(int i=0;i<this->numberOfCls;i++){
		this->classifiers[i] = new Classifier(allClassifiers[i]);
	}
}
AllAttributes::AllAttributes(vector<string> allClassifiers,vector<string> svmPaths) {
	this->numberOfCls = allClassifiers.size();
	this->classifiers = new Classifier*[numberOfCls];
	for(int i=0;i<this->numberOfCls;i++){
		this->classifiers[i] = new Classifier(allClassifiers[i],svmPaths[i]);
	}
}

float* AllAttributes::getAllAttributes(cv::Mat img){
	float* ret=new float[this->numberOfCls];
	//cv::Mat alignedImg = alignCropFace(img);
	float val;
	int i=0;
	cv::Mat feat;
	classifiers[0]->getFeats(img,feat);
	for(int i=0;i<this->numberOfCls;i++){
		ret[i]=this->classifiers[i]->predict(feat);
	}
	return ret;
}

AllAttributes::~AllAttributes(){
	for(int i=0;i<this->numberOfCls;i++){
		delete this->classifiers[i];
	}
	delete this->classifiers;
}

