/*
 * dlibalign.hpp
 *
 *  Created on: Aug 26, 2015
 *      Author: pouya
 */

#ifndef _DLIBALIGN_HPP_
#define _DLIBALIGN_HPP_
#include <opencv2/core/core.hpp>
#include "types.hpp"
void init_basePoint();
cv::Mat cropAlignFace(cv::Mat& src);
cv::Mat cropAlignFace(cv::Mat& src,cv::Rect& faceRect);
cv::Mat getPart(cv::Mat aligned, FaceComponent fc,cv::Rect faceRect,cv::Rect& outRect);


#endif /* DLIBALIGN_HPP_ */
