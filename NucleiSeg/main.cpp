#include "initializationPhase.h"

#include <string>
#include <cstdio>
#include <iostream>

using namespace std;

int main() {

	string filename;
	for (int i = 1; i <= 40; i++)
	{   
		if(i<=9)
		filename = "G:/Ankita_Workspace/GSOC/NucleiSeg/512image/0" + to_string(i) + ".tif";
		else
			filename = "G:/Ankita_Workspace/GSOC/NucleiSeg/512image/" + to_string(i) + ".tif";
		cout << filename << endl;
		Mat input = imread(filename);
		//cout << input.size() << endl;
		if (!input.data)
		{
			cout << "Image not present" << endl;
			exit(-1);

		}
		initializationPhase ip = initializationPhase(input);
		Mat M = (Mat_<double>(3, 3) << 0.554688,0.380814,0,0.781334,0.87215,0,0.286075,0.307141,0);
		//cout << "M = " << endl << " " << M << endl << endl;
		vector<Mat> deconv;
		deconv=ip.colordeconv(input, M, Mat::ones(Size(3,1), CV_8UC1));
		cout << deconv.size() <<endl;
	    imshow("Color Deconvolved Image", deconv[0]);
		imshow("Input Image", input);
		waitKey(0);
	}
	


}