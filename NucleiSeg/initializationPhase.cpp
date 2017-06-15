#include "initializationPhase.h"

initializationPhase::initializationPhase(Mat im)
{
	input = im;
}

vector<Mat> initializationPhase::colordeconv(Mat I, Mat M, Mat stains)
{   
	
	
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
			M.at<double>(0, 2) = sqrt(1-x1+x2);
		}
		 x1 = pow(M.at<double>(1, 0), 2);
		 x2 = pow(M.at<double>(1, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(1, 2) = 0;
		else
		{
			M.at<double>(1, 2) = sqrt(1 - x1 + x2);
		}

		x1 = pow(M.at<double>(2, 0), 2);
		x2 = pow(M.at<double>(2, 1), 2);
		if ((x1 + x2) > 1)
			M.at<double>(2, 2) = 0;
		else
		{
			M.at<double>(2, 2) = sqrt(1 - x1 + x2);
		}
		M.col(2)/= norm(M.col(2));
	}
	Mat Q = M.inv(DECOMP_LU);
	cout << Q << endl;
	Mat temp1 = im2vec(I);
	temp1.convertTo(temp1,CV_32FC3);
	Mat y_OD = colordeconv_normalize(temp1); 
	y_OD.convertTo(y_OD, CV_64FC1); Q.convertTo(Q, CV_64FC1);

	Mat C = Q*y_OD;
	Mat channel = colordeconv_denormalize(C);
	int m = I.rows; int n = I.cols;
	Mat intensity=Mat::zeros(I.size(),CV_8UC3),temp2;
	vector<Mat> splitCh;
	split(intensity, splitCh);
	for (int i = 0; i < stains.cols; i++)
	{   
		temp2 = channel.row(i);
		temp2.convertTo(temp2, CV_8UC1);
		splitCh[i] = matlab_reshape(temp2, m, n, 1);
	}
	merge(splitCh, intensity);
	vector<Mat> colorStainImages;
	Mat stain_OD, stain_RGB,temp3;
	for (int i = 0; i < stains.cols; i++)
	{
		stain_OD = M.col(i)*C.row(i);
		stain_RGB = colordeconv_denormalize(stain_OD);
		splitCh[0] = matlab_reshape(stain_RGB.row(0), m, n, 1);
		splitCh[1] = matlab_reshape(stain_RGB.row(1), m, n, 1);
		splitCh[2] = matlab_reshape(stain_RGB.row(2), m, n, 1);
		merge(splitCh, temp3);
		colorStainImages.push_back(temp3);
		cout << "iteration: "<< i <<"limit: " << stains.cols <<  endl;
	}
	cout << colorStainImages.size() << endl;
	return colorStainImages;

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
		split(I, splitted);
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
	cout << "im2vec size: " << vec.size() << endl;
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
	exp(data, denorm_deconv);
	denorm_deconv += 255;
	return denorm_deconv;
}

Mat initializationPhase::matlab_reshape(const Mat &m, int new_row, int new_col, int new_ch)
{
	int old_row, old_col, old_ch;
	old_row = m.size().height;
	old_col = m.size().width;
	old_ch = m.channels();

	Mat m1(1, new_row*new_col*new_ch, m.depth());

	vector <Mat> p(old_ch);
	split(m, p);
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