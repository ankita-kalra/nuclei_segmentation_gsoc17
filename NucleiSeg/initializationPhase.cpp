#include "initializationPhase.h"
#include "frangi.h"

initializationPhase::initializationPhase(Mat im)
{
	input = im;
}

Mat initializationPhase::im_32f_or_64f_to_8u(Mat _fpImage) {
	
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(_fpImage, &minVal, &maxVal, &minLoc, &maxLoc);
	_fpImage -= minVal;
	Mat _8ucImage;
	_fpImage.convertTo(_8ucImage, CV_8U, 255 / (maxVal - minVal));
	
	return _8ucImage;
}

vector<Mat> initializationPhase::colordeconv(Mat I, Mat M, Mat stains)
{   
	Mat diff_checker; vector<Mat> test;
	for (int i = 0; i < 3; i++)
	{
		if (norm(M.col(i)))
			M.col(i) /= norm(M.col(i));
	}
	if (norm(M.col(2)) == 0)
	{   
		double x1 =  pow(M.at<double>(0, 0),2);
		double x2 = pow(M.at<double>(0, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(0, 2) = 0;
		else
		{
			M.at<double>(0, 2) = sqrt(1-(x1+x2));
		}
		 x1 = pow(M.at<double>(1, 0), 2);
		 x2 = pow(M.at<double>(1, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(1, 2) = 0;
		else
		{
			M.at<double>(1, 2) = sqrt(1 - (x1 + x2));
		}
		x1 = pow(M.at<double>(2, 0), 2);
		x2 = pow(M.at<double>(2, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(2, 2) = 0;
		else
		{
			M.at<double>(2, 2) = sqrt(1 -( x1 + x2));
		}
		M.col(2)/= norm(M.col(2));
	}
	cout << "M= " << endl << M << endl;
	Mat Q = (Mat_<double>(3, 3) << 4.8869, - 0.7311, - 3.9831, -4.3780 ,   1.8015  ,  3.5684, -0.0688 ,- 0.4440  ,  1.3462);
	Q = M.inv(DECOMP_LU);
	cvtColor(I, I, CV_BGR2RGB);
	split(I, test);
	merge(test, I);
	Mat temp1 = im2vec(I),temp1_1,temp1_2;
	temp1.convertTo(temp1_2,CV_32F);
	Mat y_OD = colordeconv_normalize(temp1_2); 
	y_OD.convertTo(y_OD, CV_64FC1);

	Q.convertTo(Q, CV_32FC1);
	Q.convertTo(Q, CV_64FC1);
	
	Mat C = Q*y_OD;
	Mat channel = colordeconv_denormalize(C);
	int m = I.rows; int n = I.cols;
	Mat intensity=Mat::zeros(I.size(),CV_32FC3),temp2;
	vector<Mat> splitCh;
	cv::split(intensity, splitCh);
	for (int i = 0; i < stains.cols; i++)
	{   
		temp2 = channel.row(i);
		temp2.convertTo(temp2, CV_8UC1);
		splitCh[i] = matlab_reshape(temp2, m, n, 1);
	}
	merge(splitCh, intensity);

	vector<Mat> colorStainImages;
	Mat stain_OD, stain_RGB,temp3;
	for (int i = 0; i <3; i++)
	{  
		stain_OD = M.col(i)*C.row(i);
		stain_RGB = colordeconv_denormalize(stain_OD);
		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		minMaxLoc(stain_RGB, &minVal, &maxVal, &minLoc, &maxLoc);
		stain_RGB -= minVal;
		stain_RGB.convertTo(stain_RGB, CV_8U, 255 / (maxVal - minVal));
		minMaxLoc(stain_RGB, &minVal, &maxVal, &minLoc, &maxLoc);
		splitCh[0] = matlab_reshape(stain_RGB.row(0), m, n, 1);
		splitCh[1] = matlab_reshape(stain_RGB.row(1), m, n, 1);
		splitCh[2] = matlab_reshape(stain_RGB.row(2), m, n, 1);
		merge(splitCh, temp3);
		colorStainImages.push_back(temp3);
		temp3.release();
	}
	std::cout << colorStainImages.size() << endl;
	return colorStainImages;
}

Mat initializationPhase::preprocess_hemat_generate_vote(Mat hemat)
{  
    Mat CCS= complement_contrast_smoothen(hemat);
	Mat diff = diff_image(CCS);
	CCS.release();
	diff.convertTo(diff, CV_32F);
	Mat vote_map = voting_map_const(diff);
	return vote_map;
}

Mat initializationPhase::im2vec(Mat I)
{ 
	int M = I.rows;
	int N = I.cols;
	vector<Mat> splitted;
	Mat vec;
	Mat temp;
	
	if (I.channels() == 3)
	{
		cv::split(I, splitted);
		for (int i = 0; i < 3; i++)
		{
			transpose(splitted[i], splitted[i]);
			temp = splitted[i].reshape(0, 1);
			temp.convertTo(temp,CV_64F);
			vec.push_back(temp);
		}
	}
	else if (I.channels() == 1)
	{
		transpose(I, I);
		I.convertTo(I, CV_64F);
		vec.push_back(I.reshape(0, 1));
	}
	else
	{   
		vec.push_back(Mat::zeros(Size(1, 1), CV_64FC1));
	}
	std::cout << "im2vec size: " << vec.size() << endl;
	return vec;
}

Mat initializationPhase::colordeconv_normalize(Mat data)
{  
	Mat denorm_deconv;
	double epsd = numeric_limits<double>::epsilon();
	data += epsd; data /= 255;
	log(data, denorm_deconv);
	denorm_deconv *= -1;
	return denorm_deconv;
	 
}

Mat initializationPhase::colordeconv_denormalize(Mat data)
{
	Mat denorm_deconv;
	exp(-data, denorm_deconv);
	denorm_deconv *= 255;
	return denorm_deconv;
}

Mat initializationPhase::complement_contrast_smoothen(Mat hemat)
{   
	Mat result, G,h;
	int kernel_size = 4;
	Mat gray_hemat;
	double alpha = 3.0; int beta = 30;
	if (hemat.channels()>1)
		cvtColor(hemat, gray_hemat, CV_BGR2GRAY);
	result = Mat::zeros(gray_hemat.size(), gray_hemat.type());
	cout << type2str(gray_hemat.type()) << endl;
	for (int y = 0; y < gray_hemat.rows; y++)
	{
		for (int x = 0; x < gray_hemat.cols; x++)
		{
				result.at<uchar>(y, x) =
					saturate_cast<uchar>(alpha*(gray_hemat.at<uchar>(y, x)) + beta);
		}
	}
	
	h = 255 - result;
	GaussianBlur(h,G, Size(2*kernel_size+1, 2*kernel_size+1), 1);
	//imshow("pre-processed image", G);
	//waitKey(0);
	return G;
}

Mat initializationPhase::diff_image(Mat smoothened)
{
	Mat result;
	int morph_size = 6;
	int morph_elem = MORPH_ELLIPSE;
	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1));
	morphologyEx(smoothened, result, MORPH_OPEN, element);
	result = abs(smoothened-result);
	return result;
}

Mat initializationPhase::voting_map_const(Mat pp) {
	frangi2d_opts_t opts;
	frangi2d_createopts(&opts);
	Mat vesselness, scale, angles;
	frangi2d(pp,vesselness,scale,angles,opts);
	return vesselness;
}

Mat initializationPhase::matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch)
{
	int old_row, old_col, old_ch;
	old_row = m.size().height;
	old_col = m.size().width;
	old_ch = m.channels();

	Mat m1(1, new_row*new_col*new_ch, m.depth());

	vector <Mat> p(old_ch);
	cv::split(m, p);
	for (int i = 0; i<p.size(); ++i) {
		Mat t(p[i].size().height, p[i].size().width, m1.type());
		t = p[i].t();
		Mat aux = m1.colRange(i*old_row*old_col, (i + 1)*old_row*old_col).rowRange(0, 1);
		t.reshape(0, 1).copyTo(aux);
	}

	vector <Mat> r(new_ch);
	for (int i = 0; i<r.size(); ++i) {
		Mat aux = m1.colRange(i*new_row*new_col, (i + 1)*new_row*new_col).rowRange(0, 1);
		r[i] = aux.reshape(0, new_col);
		r[i] = r[i].t();
	}

	Mat result;
	merge(r, result);
	return result;
}

string initializationPhase::type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

