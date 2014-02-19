/*
   emvisi2 makes background subtraction robust to illumination changes.
   Copyright (C) 2008 Julien Pilet, Christoph Strecha, and Pascal Fua.

   This file is part of emvisi2.

   emvisi2 is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   emvisi2 is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with emvisi2.  If not, see <http://www.gnu.org/licenses/>.


   For more information about this code, see our paper "Making Background
   Subtraction Robust to Sudden Illumination Changes".
*/
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "fwncc.h"
#include <opencv2/highgui/highgui.hpp>
#include "emvisi2.h"
#include "growmat.h"

using namespace std;
using namespace cv;

int fillMatrix(cv::Mat ncc, cv::Mat var, cv::Mat visibility, cv::Mat mask, CvGrowMat *mat)
{
	int nb=0;
	assert(visibility.channels()==1);
	//assert(im.channels()==3);
	//assert(im.cols == visibility.cols && im.rows == visibility.rows);
	assert(mask.empty() || (mask.cols == visibility.cols && mask.rows == visibility.rows));
	assert(mask.empty() || (mask.channels()==1));

	for (int y=0; y<ncc.rows; y++) {
		for (int x=0; x<ncc.cols; x++) {
			if (mask.empty() || (mask.at<unsigned char>(y,x)>128)) {
				int n = mat->height;
				mat->resize(n+1,3);
				nb++;
				CV_MAT_ELEM(*mat, float, n, 0) = ncc.at<float>(y, x);
				CV_MAT_ELEM(*mat, float, n, 1) = var.at<float>(y, x);
				CV_MAT_ELEM(*mat, float, n, 2) = visibility.at<unsigned char>(y, x);
			}
		}
	}
	cout << 100.0f*nb / ((float) ncc.cols*ncc.rows) << "% inside image mask.\n";
	return nb;
}

int buildHisto(cv::Mat ncc, cv::Mat var, cv::Mat visibility, cv::Mat mask, NccHisto &f, NccHisto &g)
{
	int nb=0;
	assert(visibility.channels()==1);
	//assert(im.channels()==3);
	//assert(im.cols == visibility.cols && im.rows == visibility.rows);
	assert(mask.empty() || (mask.cols == visibility.cols && mask.rows == visibility.rows));
	assert(mask.empty() || (mask.channels()==1));


	for (int y=0; y<ncc.rows; y++) {
		for (int x=0; x<ncc.cols; x++) {
			if ((mask.empty()) || (mask.at<unsigned char>(y, x)>128)) {
				nb++;
				float c = ncc.at<float>(y,x);
				float v = var.at<float>(y,x);
				if (visibility.at<unsigned char>(y, x) > 128) {
					f.addElem(c,v,1);
				} else {
					g.addElem(c,v,1);
				}
			}
		}
	}


	cout << 100.0f*nb / ((float) ncc.cols*ncc.rows) << "% inside image mask.\n";
	return nb;
}

void save_mat(float *f, int width, int height, const char *filename)
{
	CvMat m;
	cvInitMatHeader(&m, height, width, CV_32FC1, f);
	CvGrowMat::saveMat(&m, filename);
}

cv::Mat loadIm(const char *fn, int c)
{
	Mat im(imread(fn,c));
	if (im.empty()) {
		cerr << fn << ": can't load image.\n";
		exit(-1);
	}
	return im;
}

int main(int argc, char **argv)
{

	if (argc<5) {
		cerr << "usage: " << argv[0] << " { <background> <input image> <ground truth> [ - | <mask> ] }\n";
	}

	CvGrowMat mat(300000, 3, CV_32FC1);
	mat.resize(0,3);

	FNcc fncc;

	NccHisto v,h;

	v.initEmpty();
	h.initEmpty();
	for (int i=1; i<argc; i+=4) {

		Mat model = loadIm(argv[i],0);
		Mat target = loadIm(argv[i+1],0);
		Mat visibility = loadIm(argv[i+2],0);
		Mat mask(imread(argv[i+3], 0));

		fncc.setModel(model, mask);
		fncc.setImage(target);
		Mat ncc(target.size(), CV_32FC1);
		Mat var(target.size(), CV_32FC1);

		fncc.computeNcc(EMVisi2::ncc_size, ncc, var);

		cout << argv[i] << " - " << argv[i+1] << ": ";
		fillMatrix(ncc,var,visibility,mask, &mat);
		buildHisto(ncc,var,visibility,mask, v, h);
	}

	float pf = v.nelem / (v.nelem + h.nelem);
	cout << "PF = " << 100.0f * pf  << "%" <<endl;
	v.normalize(.1);
	h.normalize(.1);
	v.saveHistogram("ncc_proba_v.mat");
	h.saveHistogram("ncc_proba_h.mat");

	/*
	CvGrowMat::saveMat(&mat, "pixels.mat");
	cout << mat.rows << " pixels stored in 'pixels.mat'.\n";
	*/

	FILE *f = fopen ("ncc_proba.cpp", "w");
	if (f) {
		int n =(NccHisto::NTEX+1)*(NccHisto::NCORR+1);
		fprintf(f, "float ncc_proba_v[] = {\n");
		for (int i=0; i<n; i++) 
			fprintf(f, "\t%16e,\n", v.lut[i]);
		fprintf(f, "};\n\nfloat ncc_proba_h[] = {\n");
		for (int i=0; i<n; i++) 
			fprintf(f, "\t%16e,\n", h.lut[i]);
		fprintf(f,"};\n");
		fclose(f);
	}
}

