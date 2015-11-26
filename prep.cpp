#include"prep.hpp"
#include <assert.h>
#include<stdio.h>
#include"types.hpp"
#include"confs.h"

bool cropAllParts(std::vector<cv::Mat>& imgs,std::vector<std::vector<cv::Rect> > & parts,int fc,std::vector<cv::Mat>& dst,int& avgWidth,int& avgHeight){
	bool res = false;
	int numImgs = imgs.size();
	avgWidth =0;
	avgHeight =0;
	for(int i=0;i<numImgs;i++){
		cv::Mat cropped;
		if (fc==HAIR){
			cv::Mat outter = imgs[i].clone();
			cv::Rect region = parts[i][INNER];
			outter(region) = cv::Mat::zeros(region.height,region.width,outter.type());
			cropped = outter;
		}else {
		cropped = imgs[i](parts[i][fc]);
		}
		avgWidth += cropped.cols;
		avgHeight += cropped.rows;
		dst.push_back(cropped);
	}

	avgWidth /= imgs.size();
	avgHeight /= imgs.size();

	avgWidth += 8-avgWidth%8;
	avgHeight += 8-avgHeight%8;
	return res;
}
