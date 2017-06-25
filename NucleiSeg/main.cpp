#include "initializationPhase.h"

#include <string>
#include <cstdio>
#include <iostream>

using namespace std;

int main() {

	string filename;
	for (int i = 1; i <=40; i++)
	{   
		if(i<=9)
		filename = "G:/Ankita_Workspace/GSOC/NucleiSeg/512image/0" + to_string(i) + ".tif";
		else
		filename = "G:/Ankita_Workspace/GSOC/NucleiSeg/512image/" + to_string(i) + ".tif";

		cout << filename << endl;
		Mat input = imread(filename);
		if (!input.data)
		{
			cout << "Image not present" << endl;
			exit(-1);

		}
		initializationPhase ip = initializationPhase(input);
		Mat M = (Mat_<double>(3, 3) << 0.5547,0.3808,0,0.7813,0.8721,0,0.2861,0.3071,0);
		vector<Mat> deconv;
   		deconv=ip.colordeconv(input, M, Mat::ones(Size(3,1), CV_8UC1));
		imwrite("Hemat_" + to_string(i) + ".png", deconv[0]);
		imwrite("Eosin_" + to_string(i) + ".png", deconv[1]);
		Mat im = imread("Hemat_" + to_string(i) + ".png");
		Mat result=ip.preprocess_hemat_generate_vote(im);
		result = ip.im_32f_or_64f_to_8u(result);
		imwrite("voting_map_" + to_string(i) + ".png", result);
		
	}
	


}