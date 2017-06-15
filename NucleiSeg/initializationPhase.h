
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>

#include <iostream>
#include <limits>
#include <cmath>

using namespace std;
using namespace cv;

class initializationPhase {

	Mat colordeconvim;
	Mat input;

  public:
    
	initializationPhase(Mat im);
	vector<Mat> colordeconv(Mat I,Mat M,Mat stains);

  private:
	Mat im2vec(Mat I);
	Mat colordeconv_normalize(Mat data);
	Mat colordeconv_denormalize(Mat data);
	Mat matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch);




};
