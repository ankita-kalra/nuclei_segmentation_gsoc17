
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

	Mat result_intermediate;
	Mat input;

  public:
    
	initializationPhase(Mat im);
	vector<Mat> colordeconv(Mat I,Mat M,Mat stains);
	void preprocess_hemat(Mat hemat);

  private:
	
	Mat colordeconv_normalize(Mat data);
	Mat im2vec(Mat I);
	Mat matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch);
	Mat colordeconv_denormalize(Mat data);
	Mat complement_contrast_smoothen(Mat hemat);
	Mat diff_image(Mat smoothened);
	Mat voting_map_const(Mat pp);
	string type2str(int type);

};
