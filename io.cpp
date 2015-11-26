#include "io.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utils.hpp>
#include <math.h>
#include <dlibalign.hpp>
#include"confs.h"

using namespace std;
using namespace cv;

void loadAll(const std::string& partsPath, const std::string& attrsPath,
		TrainingIO & io) {
	readAttributesCSV(attrsPath, io.attributes, io.attrsPaths);
	loadCSV(partsPath, io.imgs, io.parts, io.partsPathsMap);
}

bool loadCSVPartial(const std::string & csvPath, std::vector<cv::Mat> & outMat,
		std::map<std::string, int> & paths) {
	string line;
	string columnStr;
	ifstream myfile(csvPath.c_str());
	auto prefix = getFileNeme(csvPath.c_str());
	int i = 0;
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			vector<cv::Rect> parts;
			cv::Mat img;
			img = imread(line, CV_LOAD_IMAGE_COLOR);
			string fname = prefix;
			paths[fname.append("_").append(getFileNemePartial(line.c_str()))] =
					i;
			outMat.push_back(img);

			i++;
		}
		myfile.close();
	} else
		cout << "Unable to open file";
	return true;
}

void loadAllLFW(const std::string& partsPath, const std::string& attrsPath,
		TrainingIO & io) {
	readAttributesCSVLFW(attrsPath, io.attributes, io.attrsPaths);
	loadCSV(partsPath, io.imgs, io.parts, io.partsPathsMap);
}

void loadAllLFWPartial(const std::string& partsPath, const std::string& attrsPath,
		TrainingIO & io) {
	readAttributesCSVLFW(attrsPath, io.attributes, io.attrsPaths);
	loadCSVPartial(partsPath, io.imgs, io.partsPathsMap);
}

bool loadCSV(const std::string & csvPath, std::vector<cv::Mat> & outMat,
		std::vector<std::vector<cv::Rect> > & outParts,
		std::map<std::string, int> & paths) {
	string line;
	string columnStr;
	ifstream myfile(csvPath.c_str());
	auto prefix = getFileNeme(csvPath.c_str());
	int i = 0;
	if (myfile.is_open()) {
		while (getline(myfile, line)) {
			vector<cv::Rect> parts;
			cv::Mat img;
			stringstream lineStream(line);
			bool flag = true;
			int fc;
			while (getline(lineStream, columnStr, ',')) {
				if (flag) {
					flag = false;
					img = imread(columnStr, CV_LOAD_IMAGE_COLOR);
					string fname = prefix;
					paths[fname.append("_").append(
							getFileNeme(columnStr.c_str()))] = i;
					fc = 0;
				} else {
					fc++;
					stringstream colStream(columnStr);
					float x, y, w, h;
					colStream >> x >> y >> w >> h;
					int offsets[] = { 2, 2, 2, 2 }; //{up,down,left,right}
					switch (fc) {
					case MOUTH:
						offsets[UPIND] += 8;
						break;
					}

					x -= offsets[LEFTIND];
					x = x < 0 ? 0 : x;

					y -= offsets[UPIND];
					y = y < 0 ? 0 : y;

					w += offsets[RIGHTIND];
					w = x + w > IMG_W ? IMG_W - x : w;

					h += offsets[DOWNIND];
					h = y + h > IMG_H ? IMG_H - y : h;

					cv::Rect part(x, y, w, h);
					parts.push_back(part);
				}
			}
			outMat.push_back(img);
			outParts.push_back(parts);
			i++;
		}
		myfile.close();
	} else
		cout << "Unable to open file";
	return true;
}

/*
 * This function extracts the number at the end of the name in LFW file and convert it to LFW Path compatible format
 */
string prepareForLFW(string name) {
	auto parts = split(name, '_');
	auto number = parts[parts.size() - 1];
	stringstream temp(number);
	int intNum;

	temp >> intNum;

	int nZeros = 3 - floor(log10(intNum));
	string newName = "";
	for (int i = 0; i < nZeros; i++)
		newName.append("0");
	newName.append(number);
	string final = "";
	for (int i = 0; i < parts.size() - 1; i++)
		final.append(parts[i]).append("_");
	final.append(newName);

	return final;
}
bool readAttributesCSVLFW(const std::string &csvPath,
		std::vector<std::vector<int> >& attributes,
		vector<std::string>& filenameMap) {
	string line;
	string columnStr;
	ifstream myfile(csvPath.c_str());
	auto prefix = getFileNeme(csvPath.c_str());
	int attri = 0;
	if (myfile.is_open()) {
		bool flag = true; //to recognize the first line
		while (getline(myfile, line)) {
			stringstream lineStream(line);
			if (flag) {
				flag = false;
				string name;

				while (lineStream >> name) {
					string s = prefix;
					name = prepareForLFW(name);
					filenameMap.push_back(s.append("_").append(name));
				}
			} else {
				int val;
				vector<int> t;
				attributes.push_back(t);
				while (lineStream >> val) {
					attributes[attri].push_back(val);
				}
				attri++;
			}
		}
		myfile.close();
	} else
		cout << "Unable to open file";
	return true;
}
bool readAttributesCSV(const std::string &csvPath,
		std::vector<std::vector<int> >& attributes,
		vector<std::string>& filenameMap) {
	string line;
	string columnStr;
	ifstream myfile(csvPath.c_str());
	auto prefix = getFileNeme(csvPath.c_str());
	int attri = 0;
	if (myfile.is_open()) {
		bool flag = true; //to recognize the first line
		while (getline(myfile, line)) {
			stringstream lineStream(line);
			if (flag) {
				flag = false;
				string name;

				while (lineStream >> name) {
					string s = prefix;
					filenameMap.push_back(s.append("_").append(name));
				}
			} else {
				int val;
				vector<int> t;
				attributes.push_back(t);
				while (lineStream >> val) {
					attributes[attri].push_back(val);
				}
				attri++;
			}
		}
		myfile.close();
	} else
		cout << "Unable to open file";
	return true;
}

void savePCA(const string &file_name, cv::PCA pca_) {
	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "mean" << pca_.mean;
	fs << "e_vectors" << pca_.eigenvectors;
	fs << "e_values" << pca_.eigenvalues;
	fs.release();
}

void loadPCA(const string &file_name, cv::PCA &pca_) {
	FileStorage fs(file_name, FileStorage::READ);
	if (!fs.isOpened()) {
		cerr << "Failed to open " << file_name << endl;
		exit(0);
	}
	fs["mean"] >> pca_.mean;
	fs["e_vectors"] >> pca_.eigenvectors;
	fs["e_values"] >> pca_.eigenvalues;
	fs.release();

}

void saveTrainingData(const string& file_name, TrainingData all) {
	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "data" << all.data;
	fs << "labels" << all.labels;
	fs.release();
}

void loadTrainingData(const string& file_name, TrainingData& all) {
	FileStorage fs(file_name, FileStorage::READ);
	fs["data"] >> all.data;
	fs["labels"] >> all.labels;
	fs.release();
}
