#include<opencv2/core/core.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include"types.hpp"
#include"utils.hpp"
#include"prep.hpp"
#include <lbp/lbp.hpp>
#include <lbp/histogram.hpp>

#include"confs.h"
#include <stdexcept>      // std::out_of_range
using namespace std;
using namespace cv;
using namespace lbp;

Mat formatHogForPCA(const vector<Mat> &data) {
	Mat dst(static_cast<int>(data.size()), data[0].rows * data[0].cols, CV_32F);
	for (unsigned int i = 0; i < data.size(); i++) {
		Mat image_row = data[i].clone().reshape(1, 1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	return dst;
}

void putTrainingFeatures(TrainingIO & io, int fc, float scale) {
	vector<cv::Mat> cropped;
	cropAllParts(io.imgs, io.parts, fc, cropped, io.avgWidth, io.avgHeight);
	cout << "Average width:" << io.avgWidth << " Average height:"
			<< io.avgHeight << endl;
	size_t scaledWidth = floor(io.avgWidth * scale);
	scaledWidth -= (scaledWidth - ((scaledWidth >> 3) << 3));
	size_t scaledHeight = floor(io.avgHeight * scale);
	scaledHeight -= (scaledHeight - ((scaledHeight >> 3) << 3));
	cout << "Scaled width:" << scaledWidth << " Scaled height:" << scaledHeight
			<< endl;

	HOGDescriptor hg(Size(scaledWidth, scaledHeight), Size(8, 8), Size(8, 8),
			Size(4, 4), 15, -10.2, true, 64);

	io.avgWidth = scaledWidth;
	io.avgHeight = scaledHeight;

	vector<Mat> hogs;
	vector<float> labels;
	for (size_t i = 0; i < cropped.size(); i++) {
		cv::Mat dest;
		vector<cv::Point> locations;
		vector<float> descVals;
		cv::resize(cropped[i], dest, Size(scaledWidth, scaledHeight));
		Mat channel[3];
		for (int j = 0; j < 3; j++) {
			channel[j].create(scaledHeight, scaledWidth, CV_32F);
		}
		Mat gray;
		gray.create(scaledHeight, scaledWidth, CV_32F);
		cvtColor(dest, gray, COLOR_RGB2GRAY);
		gray.convertTo(gray, CV_8U);
		cv::split(dest, channel);
		hg.compute(gray, descVals, Size(0, 0), Size(0, 0), locations);
		for (int j = 0; j < 3; j++) {
			vector<float> channelVals;
			channel[j].convertTo(channel[j], CV_8U);
			hg.compute(channel[j], channelVals, Size(0, 0), Size(0, 0),
					locations);
			descVals.insert(descVals.end(), channelVals.begin(),
					channelVals.end());
		}
		thresholdFloatVec(descVals);
		Mat descMat(1, descVals.size(), CV_32FC1, descVals.data());

		if (i == 0) {
			io.imgsFeats.create(static_cast<int>(cropped.size()),
					cropped.size() * descVals.size(), CV_32FC1);
		}

		descMat.row(0).copyTo(io.imgsFeats.row(i));

	}

}

void getTrainingFeats(TrainingIO & io, int desiredAttr, Mat& outHogs,
		float*& labelsint) {

	vector<int> posInds;
	vector<int> negInds;
	auto attrLabels = io.attributes[desiredAttr];
	vector<int> posImgs;
	vector<int> negImgs;
	size_t i = 0;
	for (i = 0; i < attrLabels.size(); i++) {
		if (attrLabels[i] != 0) {
			if (attrLabels[i] == 1)
				posInds.push_back(i);
			else if (attrLabels[i] == -1)
				negInds.push_back(i);
		}
	}
	for (i = 0; i < posInds.size(); i++) {
		auto imgPath = io.attrsPaths[posInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			posImgs.push_back(imgInd);
		} catch (std::out_of_range oor) {

		}
	}

	for (i = 0; i < negInds.size(); i++) {
		auto imgPath = io.attrsPaths[negInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			negImgs.push_back(imgInd);
		} catch (out_of_range ex) {

		}
	}

	size_t minSize =
			negImgs.size() > posImgs.size() ? posImgs.size() : negImgs.size();
	//outHogs.create(static_cast<int>(2 * minSize),	2 * minSize * io.imgsFeats.cols, CV_32FC1);

//Labels are set here:
	labelsint = new float[2 * minSize];
	vector<Mat> imgs;
	for (size_t i = 0; i < 2 * minSize; i++) {
		if (i < minSize) {
			//imgs.push_back(posImgs[i]);
			outHogs.row(i) = io.imgsFeats.row(posImgs[i]);
			labelsint[i] = 1;
		} else {
			//imgs.push_back(negImgs[i - minSize]);

			outHogs.row(i) = io.imgsFeats.row(negImgs[i - minSize]);
			labelsint[i] = -1;
		}
	}

}

void hogForTraining(TrainingIO & io, int fc, int desiredAttr, Mat& outHogs,
		float*& labelsint, float scale = 1) {
	vector<int> posInds;
	vector<int> negInds;
	auto attrLabels = io.attributes[desiredAttr];
	vector<cv::Mat> posImgs;
	vector<cv::Mat> negImgs;
	size_t i = 0;
	for (i = 0; i < attrLabels.size(); i++) {
		if (attrLabels[i] != 0) {
			if (attrLabels[i] == 1)
				posInds.push_back(i);
			else if (attrLabels[i] == -1)
				negInds.push_back(i);
		}
	}
	for (i = 0; i < posInds.size(); i++) {
		auto imgPath = io.attrsPaths[posInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			posImgs.push_back(io.imgs[imgInd]);
		} catch (std::out_of_range oor) {

		}
	}

	for (i = 0; i < negInds.size(); i++) {
		auto imgPath = io.attrsPaths[negInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			negImgs.push_back(io.imgs[imgInd]);
		} catch (out_of_range ex) {

		}
	}

	size_t minSize =
			negImgs.size() > posImgs.size() ? posImgs.size() : negImgs.size();

//Labels are set here:
	labelsint = new float[2 * minSize];
	vector<Mat> imgs;
	for (size_t i = 0; i < 2 * minSize; i++) {
		if (i < minSize) {
			imgs.push_back(posImgs[i]);
			labelsint[i] = 1;
		} else {
			imgs.push_back(negImgs[i - minSize]);
			labelsint[i] = -1;
		}
	}
	cout << "Training size: " << 2 * minSize << endl;

	vector<cv::Mat> cropped;
	cropAllParts(imgs, io.parts, fc, cropped, io.avgWidth, io.avgHeight);
	cout << "Average width:" << io.avgWidth << " Average height:"
			<< io.avgHeight << endl;
	size_t scaledWidth = floor(io.avgWidth * scale);
	scaledWidth -= (scaledWidth - ((scaledWidth >> 3) << 3));
	size_t scaledHeight = floor(io.avgHeight * scale);
	scaledHeight -= (scaledHeight - ((scaledHeight >> 3) << 3));
	cout << "Scaled width:" << scaledWidth << " Scaled height:" << scaledHeight
			<< endl;

	HOGDescriptor hg(Size(scaledWidth, scaledHeight), Size(8, 8), Size(8, 8),
			Size(4, 4), 15, -10.2, true, 64);

	vector<Mat> hogs;
	vector<float> labels;
	for (i = 0; i < cropped.size(); i++) {
		cv::Mat dest;
		vector<cv::Point> locations;
		vector<float> descVals;
		cv::resize(cropped[i], dest, Size(scaledWidth, scaledHeight));
		Mat channel[3];
		for (int j = 0; j < 3; j++) {
			channel[j].create(scaledHeight, scaledWidth, CV_32F);
		}
		Mat gray;
		gray.create(scaledHeight, scaledWidth, CV_32F);
		cvtColor(dest, gray, COLOR_RGB2GRAY);
		gray.convertTo(gray, CV_8U);
		cv::split(dest, channel);
		hg.compute(gray, descVals, Size(0, 0), Size(0, 0), locations);
		for (int j = 0; j < 3; j++) {
			vector<float> channelVals;
			channel[j].convertTo(channel[j], CV_8U);
			hg.compute(channel[j], channelVals, Size(0, 0), Size(0, 0),
					locations);
			descVals.insert(descVals.end(), channelVals.begin(),
					channelVals.end());
		}
		thresholdFloatVec(descVals);
		Mat descMat(1, descVals.size(), CV_32FC1, descVals.data());
		hogs.push_back(descMat.clone());
	}
//Mat is set here
	outHogs = formatHogForPCA(hogs);
}

void lbpForTraining(TrainingIO & io, int fc, int desiredAttr, Mat& outHogs,
		float*& labelsint, float scale = 1) {
	vector<int> posInds;
	vector<int> negInds;
	auto attrLabels = io.attributes[desiredAttr];
	vector<cv::Mat> posImgs;
	vector<cv::Mat> negImgs;
	size_t i = 0;
	for (i = 0; i < attrLabels.size(); i++) {
		if (attrLabels[i] != 0) {
			if (attrLabels[i] == 1)
				posInds.push_back(i);
			else if (attrLabels[i] == -1)
				negInds.push_back(i);
		}
	}
	for (i = 0; i < posInds.size(); i++) {
		auto imgPath = io.attrsPaths[posInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			posImgs.push_back(io.imgs[imgInd]);
		} catch (std::out_of_range oor) {

		}
	}

	for (i = 0; i < negInds.size(); i++) {
		auto imgPath = io.attrsPaths[negInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			negImgs.push_back(io.imgs[imgInd]);
		} catch (out_of_range ex) {

		}
	}

	size_t minSize =
			negImgs.size() > posImgs.size() ? posImgs.size() : negImgs.size();

//Labels are set here:
	labelsint = new float[2 * minSize];
	vector<Mat> imgs;
	for (size_t i = 0; i < 2 * minSize; i++) {
		if (i < minSize) {
			imgs.push_back(posImgs[i]);
			labelsint[i] = 1;
		} else {
			imgs.push_back(negImgs[i - minSize]);
			labelsint[i] = -1;
		}
	}
	cout << "Training size: " << 2 * minSize << endl;

	vector<cv::Mat> cropped;
	cropAllParts(imgs, io.parts, fc, cropped, io.avgWidth, io.avgHeight);
	cout << "Average width:" << io.avgWidth << " Average height:"
			<< io.avgHeight << endl;
	size_t scaledWidth = floor(io.avgWidth * scale);
	scaledWidth -= (scaledWidth - ((scaledWidth >> 3) << 3));
	size_t scaledHeight = floor(io.avgHeight * scale);
	scaledHeight -= (scaledHeight - ((scaledHeight >> 3) << 3));
	cout << "Scaled width:" << scaledWidth << " Scaled height:" << scaledHeight
			<< endl;

	vector<Mat> hogs;
	vector<float> labels;
	for (i = 0; i < cropped.size(); i++) {
		cv::Mat dest;
		vector<cv::Point> locations;
		vector<float> descVals;
		cv::resize(cropped[i], dest, Size(scaledWidth, scaledHeight));
		Mat channel[3];
		for (int j = 0; j < 3; j++) {
			channel[j].create(scaledHeight, scaledWidth, CV_32F);
		}
		Mat gray;
		gray.create(scaledHeight, scaledWidth, CV_32F);
		cvtColor(dest, gray, COLOR_RGB2GRAY);
		gray.convertTo(gray, CV_8U);
		cv::split(dest, channel);

		Mat lbp_image = ELBP(gray, LBP_RADIUS, LBP_NEIGHBORS);
		Mat query = spatial_histogram(lbp_image, /* lbp_image */
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

			lbp_image = ELBP(channel[j], LBP_RADIUS, LBP_NEIGHBORS);
			query = spatial_histogram(lbp_image, /* lbp_image */
			static_cast<int>(std::pow(2.0, static_cast<double>(LBP_NEIGHBORS))), /* number of possible patterns */
			8, /* grid size x */
			8, /* grid size y */
			true /* normed histograms */);
			query.convertTo(query, CV_32F);
			Mat t = query.clone();
			hconcat(descMat, t, descMat);
		}
		hogs.push_back(descMat.clone());
	}
//Mat is set here
	outHogs = formatHogForPCA(hogs);
}

void lbpFromImgs(TrainingIO & io, int desiredAttr, Mat& outHogs,
		float*& labelsint, float scale = 1) {
	vector<int> posInds;
	vector<int> negInds;
	auto attrLabels = io.attributes[desiredAttr];
	vector<cv::Mat> posImgs;
	vector<cv::Mat> negImgs;
	size_t i = 0;
	for (i = 0; i < attrLabels.size(); i++) {
		if (attrLabels[i] != 0) {
			if (attrLabels[i] == 1)
				posInds.push_back(i);
			else if (attrLabels[i] == -1)
				negInds.push_back(i);
		}
	}
	for (i = 0; i < posInds.size(); i++) {
		auto imgPath = io.attrsPaths[posInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			posImgs.push_back(io.imgs[imgInd]);
		} catch (std::out_of_range oor) {

		}
	}

	for (i = 0; i < negInds.size(); i++) {
		auto imgPath = io.attrsPaths[negInds[i]];
		try {
			auto imgInd = io.partsPathsMap.at(imgPath);
			negImgs.push_back(io.imgs[imgInd]);
		} catch (out_of_range ex) {

		}
	}

	size_t minSize =
			negImgs.size() > posImgs.size() ? posImgs.size() : negImgs.size();

//Labels are set here:
	labelsint = new float[2 * minSize];
	vector<Mat> imgs;
	io.avgWidth =0;
	io.avgHeight = 0;
	for (size_t i = 0; i < 2 * minSize; i++) {
		if (i < minSize) {
			imgs.push_back(posImgs[i]);
			labelsint[i] = 1;
		} else {
			imgs.push_back(negImgs[i - minSize]);
			labelsint[i] = -1;
		}
		io.avgWidth += imgs[i].cols;
		io.avgHeight += imgs[i].rows;
	}
	io.avgWidth = io.avgWidth/(2*minSize);
	io.avgHeight = io.avgHeight/(2*minSize);

	cv::Size s = getStandardSize2(io.avgWidth,io.avgHeight);
	io.avgWidth = 48;
	io.avgHeight = 48;

	cout << "Training size: " << 2 * minSize << endl;

	cout << "Average width:" << io.avgWidth << " Average height:"
			<< io.avgHeight << endl;
	size_t scaledWidth = floor(io.avgWidth * scale);
	scaledWidth -= (scaledWidth - ((scaledWidth >> 3) << 3));
	size_t scaledHeight = floor(io.avgHeight * scale);
	scaledHeight -= (scaledHeight - ((scaledHeight >> 3) << 3));
	cout << "Scaled width:" << scaledWidth << " Scaled height:" << scaledHeight
			<< endl;

	vector<Mat> hogs;
	vector<float> labels;
	for (i = 0; i < imgs.size(); i++) {
		cv::Mat dest;
		vector<cv::Point> locations;
		vector<float> descVals;
		cv::resize(imgs[i], dest, Size(scaledWidth, scaledHeight));
		Mat channel[3];
		for (int j = 0; j < 3; j++) {
			channel[j].create(scaledHeight, scaledWidth, CV_32F);
		}
		Mat gray;
		gray.create(scaledHeight, scaledWidth, CV_32F);
		cvtColor(dest, gray, COLOR_RGB2GRAY);
		gray.convertTo(gray, CV_8U);
		cv::split(dest, channel);

		Mat lbp_image = ELBP(gray, LBP_RADIUS, LBP_NEIGHBORS);
		Mat query = spatial_histogram(lbp_image, /* lbp_image */
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
			lbp_image = ELBP(channel[j], LBP_RADIUS, LBP_NEIGHBORS);
			query = spatial_histogram(lbp_image, /* lbp_image */
			static_cast<int>(std::pow(2.0, static_cast<double>(LBP_NEIGHBORS))), /* number of possible patterns */
			8, /* grid size x */
			8, /* grid size y */
			true /* normed histograms */);
			query.convertTo(query, CV_32F);
			Mat t = query.clone();
			hconcat(descMat, t, descMat);
		}
		hogs.push_back(descMat.clone());
	}
//Mat is set here
	outHogs = formatHogForPCA(hogs);
}

//OpenCV example modified to serve my need

cv::PCA compressPCA(cv::Mat in, double retainedEnergy, Mat& out) {
	cv::PCA pca(in, cv::Mat(), CV_PCA_DATA_AS_ROW, retainedEnergy);

	out.create(in.rows, pca.eigenvalues.rows, in.type());
	Mat reconstructed;
	for (int i = 0; i < in.rows; i++) {
		Mat vec = in.row(i), coeffs = out.row(i);
		pca.project(vec, coeffs);
	}
	return pca;
}

