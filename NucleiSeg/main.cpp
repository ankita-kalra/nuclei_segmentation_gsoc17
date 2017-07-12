#include "initializationPhase.h"

#include <string>
#include <cstdio>
#include <iostream>
#include <set>
using namespace std;

int main() {

	string filename;
	for (int i = 1; i <=1; i++)
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
		//Mat M = (Mat_<double>(3, 3) << 0.5547,0.3808,0,0.7813,0.8721,0,0.2861,0.3071,0);
		Mat M = (Mat_<int>(3, 3) << 1,2, 9, 5, 9, 2, 8, 3, 1);
		cout << " M before = " << endl << M << endl;
		vector<Mat> deconv;
   		/*deconv=ip.colordeconv(input, M, Mat::ones(Size(3,1), CV_8UC1));
		imwrite("Hemat_" + to_string(i) + ".png", deconv[0]);
		imwrite("Eosin_" + to_string(i) + ".png", deconv[1]);
		Mat im = imread("Hemat_" + to_string(i) + ".png");
		Mat result=ip.preprocess_hemat_generate_vote(im);
		result = ip.im_32f_or_64f_to_8u(result);
		imwrite("voting_map_" + to_string(i) + ".png", result);*/
		
		//cv::sort(M, M, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
		Mat M_orig = M.clone();
		M = ip.matlab_reshape(M.t(), M.cols*M.rows, 1, 1);
		Mat mt = M;
		mt.convertTo(mt, CV_8UC1);
		//cout << mt << endl;
		vector<int> array(mt.rows*mt.cols);
		if (mt.isContinuous()) {
				array.assign(mt.datastart,mt.dataend);
				cout << "one shot " << endl;
			}
			else {
				for (int i = 0; i < mt.rows; ++i) {
					array.insert(array.end(), mt.ptr<int>(i),mt.ptr<int>(i) + mt.cols);
				}
			}

			for (int i=0; i<array.size(); i++) cout << array.at(i) << " ";
			cout << endl;
			std::set<int> c(array.begin(),array.end());

			std::vector<int> u;
			u.reserve(array.size());

			std::transform(array.begin(), array.end(), std::back_inserter(u),
				[&](int x)
			{
				return (std::distance(c.begin(), c.find(x)));
			});

		
		set<int>::iterator citer;
		mt.release();
		for (citer = c.begin(); citer != c.end(); citer++) {
			mt.push_back(*citer);
		}
		//cout << mt.at<int>(0, 1) << endl;
		
		Mat mtrev = mt.clone();
		cout << mt.type() << endl;
		mtrev.convertTo(mtrev, CV_32F);
		cv::sort(mt, mt, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
		cout << mt.type() << endl;
		cout << " v after = " << endl << mt << endl;
		Mat vnorm = ( mtrev- mt.at<int>(0, mt.rows - 1)) / (float)(mt.at<int>(0, 0) - mt.at<int>(0, mt.rows - 1));
		cout << "vnorm" << endl << vnorm << endl;
		cout << mt.at<int>(0, 0)<< endl;
		Mat a =( M_orig >= (mt.at<int>(0,0)));
		a /= 255;
		cout << "a =" << endl << a << endl;

		// Create an image
		const int color_white = 255;
		Mat src = Mat::zeros(600, 800, CV_8UC1);

		rectangle(src, Point(100, 100), Point(200, 200), color_white, CV_FILLED);
		rectangle(src, Point(500, 150), Point(600, 450), color_white, CV_FILLED);
		rectangle(src, Point(350, 250), Point(359, 251), color_white, CV_FILLED);
		rectangle(src, Point(354, 246), Point(355, 255), color_white, CV_FILLED);
		circle(src, Point(300, 400), 75, color_white, CV_FILLED);

		imshow("Original", src);

		// Get connected components and stats
		const int connectivity_4 = 4;
		Mat labels, stats, centroids;

		int nLabels = connectedComponentsWithStats(src, labels, stats, centroids, connectivity_4, CV_32S);

		cout << "Number of connected components = " << nLabels << endl << endl;

		
		// Statistics
		cout << "Show statistics and centroids:" << endl;
		cout << "stats:" << endl << "(left,top,width,height,area)" << endl << stats << endl << endl;
		cout << "centroids:" << endl << "(x, y)" << endl << centroids << endl << endl;

		// Print individual stats for component 1 (component 0 is background)
		cout << "Component 1 stats:" << endl;
		cout << "CC_STAT_LEFT   = " << stats.at<int>(1, CC_STAT_LEFT) << endl;
		cout << "CC_STAT_TOP    = " << stats.at<int>(1, CC_STAT_TOP) << endl;
		cout << "CC_STAT_WIDTH  = " << stats.at<int>(1, CC_STAT_WIDTH) << endl;
		cout << "CC_STAT_HEIGHT = " << stats.at<int>(1, CC_STAT_HEIGHT) << endl;
		cout << "CC_STAT_AREA   = " << stats.at<int>(1, CC_STAT_AREA) << endl;

		// Create image with only component 2
		Mat only2;
		compare(labels, 2, only2, CMP_EQ);

		imshow("Component 2", only2);

		waitKey(0);

	}
	


}