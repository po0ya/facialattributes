#ifndef __PREP__
#define __PREP__

#include<iostream>
#include<opencv2/core/core.hpp>
#include "types.hpp"

bool cropAllParts(std::vector<cv::Mat>& imgs,std::vector<std::vector<cv::Rect> > & parts,int fc,std::vector<cv::Mat>& dst,int& avgWidth,int& avgHeight);
#endif
