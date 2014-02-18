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
/*
 * Julien Pilet, Feb 2008
 */
#ifndef EMVISI2_H
#define EMVISI2_H

#include "fwncc.h"
#include "imstat.h"

/* This class is used to represent an histogram of textureness/correlation 
 * on the background or on the foreground.
 */
class NccHisto {
public:
	NccHisto();
	~NccHisto();
	bool loadHistogram(const char *filename = "proba.mat");
	bool saveHistogram(const char *filename);
	void setHistogram(const float *histo);
	void getProba(const IplImage *ncc, IplImage *proba);
	void getProba(const IplImage *ncc, const IplImage *sumstdev, IplImage *proba);

	void initEmpty();
	void addElem(float corr, float var, float w)
	{
		lut[lut_idx(corr, var)] += w;
		nelem+=w;
	}
	void normalize(float bias);
	
	// these contain the number of bins for the NCC histograms.
	// the values 15,15 work for the distributed ones.
	// if you change these values, you have to rebuild the histograms.
	static const int NTEX=15;
	static const int NCORR=15;

	int tex_idx(float var) {
		int i = cvFloor(sqrtf(var));
		if (i>NTEX) return NTEX;
		if (i<0) return 0;
		return i;
	}

	int correl_idx(float ncc) {
		int i = cvFloor((ncc)*(NCORR+1)/1.0f);
		if (i>NCORR) return NCORR;
		if (i<0) return 0;
		return i;
	}

	int lut_idx(float ncc, float var) {
		return correl_idx(ncc)*(NTEX+1) + tex_idx(var);
	}

	float *lut;
	float *deleteme;
public:
	float nelem;
};

class EMVisi2 {
public:
	bool save_images;

	EMVisi2();

	bool init();
        int setModel(const cv::Mat im1, const cv::Mat *mask = 0);
	int setTarget(const IplImage *target);
	void iterate();
	void smooth(float amount, float threshold);
	void run(int nbIter=3, float smooth_amount=2.5, float smooth_threshold=.0001);
	float process_pixel(const float *rgb, const float *frgb, const float dl, const float nccv, const float ncch, float *proba, float *visi_proba);
	void reset_gaussians();

        cv::Mat proba, visi_proba;

	bool recycle;
	float PF;
	static const int ncc_size = 25;

	cv::Mat prod_f, prod_g;

protected:
	FNcc fncc;
	NccHisto ncc_h;
	NccHisto ncc_v;
	cv::Mat im1f;
	cv::Mat mask;

	cv::Mat visible;
	cv::Mat hidden;
	cv::Mat ncc, sum;
	cv::Mat ratio;
	cv::Mat nccproba_v;
	cv::Mat nccproba_h;
	cv::Mat dx,dy,diffusion;
	int iteration;

	cv::Mat _im2;
        cv::Mat im2;

	cv::Mat dL;

#define NB_VISI_GAUSSIANS 2
#define NB_OCCL_GAUSSIANS 2 
#define NB_GAUSSIANS (NB_VISI_GAUSSIANS+NB_OCCL_GAUSSIANS)

	MultiGaussian3<float> visi_g[NB_VISI_GAUSSIANS];
	MultiGaussian3<float> occl_g[NB_OCCL_GAUSSIANS];
	float weights[NB_GAUSSIANS+1];

	float uniform_resp;
};

void scale_save(const char *fn, cv::Mat im, double scale=-1, double shift=-1);

#endif
