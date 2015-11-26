#include <types.hpp>
#include <utils.hpp>
#include <classifier.hpp>
#include <fstream>
using namespace std;

int main(int argc, char* argv[]) {
	ifstream allcslf(argv[0]);
	vector<string> paths;
	while (allcslf.good()) {
		string imgPath;
		getline(allcslf, imgPath);
		if (imgPath.find(".cl") != string::npos) {
			cout << imgPath << endl;
			paths.push_back(imgPath);
		}
	}
	AllAttributes allattrs(paths);
	predictAll(argv[1],
			argv[2], allattrs);
}

