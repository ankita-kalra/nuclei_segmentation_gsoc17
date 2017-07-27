#include "LevelSegmentation.h"
#include "mat.h"
#include <opencv2\matlab\bridge.hpp>
#include <opencv2\matlab\mxarray.hpp>
#include <opencv2\matlab\map.hpp>
#include <opencv2\matlab\transpose.hpp>

LevelSegmentation::LevelSegmentation(Mat im)
{
	input = im;
	mu = 0.07 * 255*255; // coefficient of arc length term % 0.001
	timestep = 0.1;
	xi = 2;  //coefficient of penalize term. for distance regularization term(regularize the level set function)
	omega = 2000;  //coefficient of exclusive term
	nu = 3000; //coefficient of sparse term
	sigma = 4; //scale parameter that specifies the size of the neighborhood 4
	lambdaU = 1;
	lambdaB = 1; 
	iter_outer = 30;
	iter_inner = 5;   //inner iteration for level set evolution
	epsilon = 1;   //for Heaviside function
	c0 = 5;
	peaks = readMat("./01.tif.mat", "peaks");
	trainingset = readMat("./trainingShape_v3.mat","peaks");
	cvtColor(input, input, CV_RGB2GRAY);
	int nPeaks = peaks.rows;
	Mat allinitialLSF = -c0*(Mat::ones( im.size(),im.type()));
	Mat allBW;
	int initialRadius = 5;
	int nDim = peaks.cols;
	Mat allBW = Mat::zeros(input.size(), input.type());
	Mat1i colGrid, rowGrid;
	Mat edgeBW;
	meshgridTest(Range(1, input.cols), Range(1, input.rows), colGrid, rowGrid);
	vector<Mat> initPhi;
	for (int iPeaks = 0; iPeaks < nPeaks; iPeaks++)
	{
		Mat BW;
		sqrt(rowGrid - (peaks.col(1)).mul(peaks.col(1)) + colGrid - (peaks.col(0)).mul(peaks.col(0)),BW);
		allBW += BW;
		Sobel(BW, edgeBW,BW.type(),0,1);
		Mat dm = Mat::zeros(edgeBW.size(), edgeBW.type());
		distanceTransform(edgeBW, dm, CV_DIST_L2, 3);
		//--
		initPhi.push_back(dm);
	}
	allinitialLSF(allBW);
	allU = allinitialLSF;
	lse(input, trainingset);
}

void LevelSegmentation::lse(Mat input, Mat trainingMat)
{ 
	updateF();
	updateLSF( g,transform);
	//TODO - sparse solver
	updateSR();
}

void LevelSegmentation::updateLSF(Mat g,vector<Mat> transform)
{
	int nPhi = u.size();
	for (int iIter = 0; iIter < nPhi; iIter++)
	{
		for (int iPhi = 0; iPhi < nPhi; iPhi++)
		{
			u.at(iPhi) = NeumannBoundCond(u.at(iPhi));
			Mat B = Mat::ones(input.size(), input.type()) - Heaviside(allU, epsilon) + Heaviside(u.at(iPhi), epsilon);
			Mat DiracU = Dirac(u.at(iPhi), epsilon);
			Mat e1 = (input - Mat::ones(input.size(), input.type()))*(cu.at<int>(Point(0, iPhi)));
			e1 = e1.mul(e1);
			Mat e2 = (input - Mat::ones(input.size(), input.type()))*cb;
			e2 = e2.mul(e2);
			e2 = e2.mul(B);
			Mat e1Term = lambdaU*e1;
			Mat e2Term = lambdaB*e2;
			Mat ImageTerm = -DiracU.mul(e1Term - e2Term);
			Mat K = div_norm(u.at(iPhi));
			pair<Mat, Mat> g_grad = gradient(g,0,0);
			pair<Mat, Mat> u_grad = gradient(u.at(iPhi),0,0);
			Mat gx = g_grad.first, gy = g_grad.second;
			Mat ux = u_grad.first; Mat uy = u_grad.second;
			Mat temp= ux.mul(ux)+ uy.mul(uy);
			double constt = pow(10, -10);
			temp += constt;
			Mat normDu;
			cv::sqrt(ux, normDu);
			double epsd = numeric_limits<double>::epsilon();
			Mat Nx = ux.mul(1 / (normDu + epsd));
			Mat Ny = uy.mul(1 / (normDu + epsd));
			Mat lengthTerm = mu*DiracU.mul(gx.mul(Nx) + gy.mul(Ny) + g.mul(K));
			Mat penalizeTerm = xi*distReg_p2(u.at(iPhi));
			Mat exclusiveTerm = -omega*DiracU.mul((Mat::ones(input.size(),input.type())) - B));
			Mat y = u.at(iPhi);
			y /= norm(y, NORM_L2);

			u.at(iPhi) += timestep + ImageTerm + lengthTerm + penalizeTerm + exclusiveTerm;
			Mat peakX = peaks.col(0);
			Mat peakY = peaks.col(1);
            

		}//inner for
		c0 = 5;
		allU = -c0*Mat::ones(input.size(), input.type());
		Mat utemp;
		for (int iPhi = 0; iPhi < nPhi; iPhi++)
		{
			utemp = u.at(iPhi);
			Mat mask = utemp > 0;
			allU(Rect()) = c0;
		}

	}//outer for

}

void LevelSegmentation::updateF()
{
	int nPhi = u.size();
	Mat Hu, Hb, NuMat, NbMat; Mat Nu, Du, cu;
	for (int i = 0; i < nPhi; i++)
	{
		Hu = Heaviside(u.at(i), epsilon);
		NuMat = input.mul(Hu);
		Nu.push_back(sum(NuMat)[0]);
		Du.push_back(sum(Hu)[0]);
		cu.at<int>(i,0) = Nu.at<int>(i,0) / Du.at<int>(i,0);

	}
	Hb = 1 - Heaviside(allU, epsilon);
	NbMat = input.mul(Hb);
	double Nb = sum(NbMat)[0];
	double Db = sum(Hb)[0];
	double cb = Nb / Db

}

void LevelSegmentation::updateSR()
{ 
	//TODO - Using Dlib c++ or ceres solver
}

Mat LevelSegmentation::readMat(string filename, string variable_name)
{
	MATFile *pmat = matOpen(filename.c_str(), "r");
	if (pmat == NULL)
	{
		cerr << "Error opening file " << filename << endl;
	}
	else
	{
		int numVars;
		char** namePtr = matGetDir(pmat, &numVars);
		cout << filename << " contains vars " << endl;
		for (int idx = 0; idx < numVars; idx++)
		{
			std::cout << "                     " << namePtr[idx] << " ";
			mxArray* m = matGetVariable(pmat, namePtr[idx]);
			matlab::MxArray mArray(m);
			cv::bridge::Bridge bridge(mArray);
			cv::Mat mat = bridge.toMat();
			if (namePtr[idx] == variable_name.c_str())
				return mat;
		}
	}

}

Mat LevelSegmentation::Heaviside(Mat x, int epsilon)
{   
	Mat h = x > 0;
	return h;
}

Mat LevelSegmentation::Dirac(Mat x, double epsilon)
{
	Mat f;
	double const1 = epsilon/M_PI;
	double const2 = epsilon*epsilon;
	Mat temp = x.mul(x);
	temp += epsilon;
	f = (Mat::ones(temp.size(), temp.type())).mul(const1 / temp);
}

//Make a function satisfy Neumann boundary condition
Mat LevelSegmentation::NeumannBoundCond(Mat f)
{
	int nrow = f.rows;
	int ncol = f.cols;
	Mat g = f;
	Mat mask1 = Mat::zeros(g.size(), g.type());
	Mat mask2 = Mat::ones(Size(ncol - 4, nrow - 4), g.type());
	mask2.copyTo(mask1(Rect(2,2,ncol-4,nrow-4)));
	f(Rect(2, 2, ncol - 4, nrow - 4)).copyTo(g(Rect(0,0,ncol-4,nrow-4)));
	Mat temp = g;
	temp(Rect(1, 2, ncol - 3, nrow-5)).copyTo(g(Rect(1, 0, ncol - 3, nrow-5)));
	temp = g;
	temp(Rect(2, 1, ncol - 5, nrow - 3)).copyTo(g(Rect(0, 1, ncol - 5, nrow - 3)));
	return g;
}

Mat LevelSegmentation::div_norm(Mat in)
{
	//compute curvature for u with central difference scheme
    pair<Mat,Mat> u_grad = gradient(in,1,1);
	Mat ux = u_grad.first, uy = u_grad.second;
	double constt = pow(10, -10);
	Mat temp = ux.mul(ux) + uy.mul(uy) + constt;
	Mat normDu; sqrt(temp, normDu);
	double epsd = numeric_limits<double>::epsilon();
	Mat Nx = ux.mul(1/(normDu + epsd));
	Mat Ny = uy.mul(1/(normDu + epsd));
	return divergence(Nx, Ny);
}

Mat LevelSegmentation::distReg_p2(Mat phi)
{
	//compute the distance regularization term with the double - well potential p2 in eqaution(16)
	pair<Mat, Mat> phi_grad = gradient(phi, 0, 0);
	Mat phi_x = phi_grad.first, phi_y = phi_grad.second;
	Mat s;
	sqrt(phi_x.mul(phi_x) + phi_y.mul(phi_y), s);
	Mat a;
	bitwise_and(s>=0,s<=1,a);
	Mat b = (s>1);
	Mat sin_s = 2*M_PI*s.mul(2*M_PI*s - 3) / 6;
	Mat ps = a.mul(sin_s) /( (2 * M_PI) + b.mul(s - 1));  //compute first order derivative of the double - well potential p2 in eqaution(16)
	double epsd = numeric_limits<double>::epsilon();
	Mat dps = ps.mul(1/ (s + epsd));
	pair<Mat, Mat>dps_grad = gradient(dps,0,0);
	Mat dps_x = dps_grad.first, dps_y=dps_grad.second;
	Mat temp;
	Laplacian(phi, temp,phi.type());
	Mat f = (dps_x.mul(phi_x) + dps_y.mul(phi_y) + 4 * dps.mul(temp);
}

Mat LevelSegmentation::divergence(Mat X, Mat Y) {
	
	//These are just 1, 2, 3, 4...till number of columns/rows
	Mat retval_x = gradientX(X, 1);
	Mat temp_y = Y.t();
	Mat retval_y = gradientY(temp_y, 1).t();
	Mat retval = retval_x + retval_y;
	return retval;
}

Mat LevelSegmentation::post_process(Mat u, Mat peakX, Mat peakY)
{
	return u;
}

pair<Mat, Mat> LevelSegmentation::gradient(Mat & img, float spaceX, float spaceY) {

	Mat gradY = gradientY(img, spaceY);
	Mat gradX = gradientX(img, spaceX);
	pair<Mat, Mat> retValue(gradX, gradY);
	return retValue;
}

/// Internal method to get numerical gradient for x components. 
/// @param[in] mat Specify input matrix.
/// @param[in] spacing Specify input space.
static Mat gradientX(Mat & mat, float spacing) {
	Mat grad = Mat::zeros(mat.cols, mat.rows, CV_32F);

	/*  last row */
	int maxCols = mat.cols;
	int maxRows = mat.rows;

	/* get gradients in each border */
	/* first row */
	Mat col = (-mat.col(0) + mat.col(1)) / (float)spacing;
	col.copyTo(grad(Rect(0, 0, 1, maxRows)));

	col = (-mat.col(maxCols - 2) + mat.col(maxCols - 1)) / (float)spacing;
	col.copyTo(grad(Rect(maxCols - 1, 0, 1, maxRows)));

	/* centered elements */
	Mat centeredMat = mat(Rect(0, 0, maxCols - 2, maxRows));
	Mat offsetMat = mat(Rect(2, 0, maxCols - 2, maxRows));
	Mat resultCenteredMat = (-centeredMat + offsetMat) / (((float)spacing)*2.0);

	resultCenteredMat.copyTo(grad(Rect(1, 0, maxCols - 2, maxRows)));
	return grad;
}

/// Internal method to get numerical gradient for y components. 
/// @param[in] mat Specify input matrix.
/// @param[in] spacing Specify input space.
static Mat gradientY(Mat & mat, float spacing) {
	Mat grad = Mat::zeros(mat.cols, mat.rows, CV_32F);

	/*  last row */
	const int maxCols = mat.cols;
	const int maxRows = mat.rows;

	/* get gradients in each border */
	/* first row */
	Mat row = (-mat.row(0) + mat.row(1)) / (float)spacing;
	row.copyTo(grad(Rect(0, 0, maxCols, 1)));

	row = (-mat.row(maxRows - 2) + mat.row(maxRows - 1)) / (float)spacing;
	row.copyTo(grad(Rect(0, maxRows - 1, maxCols, 1)));

	/* centered elements */
	Mat centeredMat = mat(Rect(0, 0, maxCols, maxRows - 2));
	Mat offsetMat = mat(Rect(0, 2, maxCols, maxRows - 2));
	Mat resultCenteredMat = (-centeredMat + offsetMat) / (((float)spacing)*2.0);

	resultCenteredMat.copyTo(grad(Rect(0, 1, maxCols, maxRows - 2)));
	return grad;
}

static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat1i &X, cv::Mat1i &Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}


static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
	cv::Mat1i &X, cv::Mat1i &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}


