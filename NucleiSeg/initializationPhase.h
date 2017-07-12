
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
	Mat preprocess_hemat_generate_vote(Mat hemat);
	Mat merge1(Mat input,Mat vote);
	Mat merge2(Mat input,Mat im);
	Mat im_32f_or_64f_to_8u(Mat _fpImage);
	Mat matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch);

  private:
	
	Mat colordeconv_normalize(Mat data);
	Mat im2vec(Mat I);
	Mat colordeconv_denormalize(Mat data);
	Mat complement_contrast_smoothen(Mat hemat);
	Mat diff_image(Mat smoothened);
	Mat voting_map_const(Mat pp);
	Mat peaks;
	string type2str(int type);
	Mat bwareaopen(Mat img, int size);
	template <class T>
	Mat findValue(const cv::Mat &mat, T value);

};
