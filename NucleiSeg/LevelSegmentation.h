#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <limits>
#include <cmath>

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;

class LevelSegmentation {

	Mat result_intermediate;
	Mat input;
	Mat cu, cb;
	Mat transform;
	Mat trainingset;
	Mat allU;
	Mat peaks;
	vector<Mat> u,transform;
	Mat g;
	double mu, timestep;
	int xi, omega, nu, sigma,lambdaU, lambdaB, iter_outer, iter_inner, epsilon, c0;
public:

	LevelSegmentation(Mat im);
	void lse(Mat input, Mat trainingMat);

private:

	void updateLSF(Mat g,vector<Mat> transform);
	void updateF();
	void updateSR();
	Mat readMat(string filename, string variable_name);
	Mat Heaviside(Mat x, int epsilon);
	Mat Dirac(Mat x, double epsilon);
	Mat NeumannBoundCond(Mat f);
	pair<Mat, Mat> gradient(Mat & img, float spaceX, float spaceY);
	Mat div_norm(Mat in);
	Mat distReg_p2(Mat phi);
	Mat divergence(Mat X, Mat Y);
	Mat post_process(Mat u, Mat peakX, Mat peakY);

};