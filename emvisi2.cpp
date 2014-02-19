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
 * Packaged on Nov 2008
 */
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "emvisi2.h"
#include "growmat.h"

// To enable graphcut support, 
//  - download Yuri Boykov's implementation.
//    http://www.adastral.ucl.ac.uk/~vladkolm/software/maxflow-v3.0.src.tar.gz
//  - edit the Makefile

using namespace std;
using namespace cv;

extern const float ncc_proba_h[256];
extern const float ncc_proba_v[256];

EMVisi2::EMVisi2() {
	save_images=false;
	exp_table(0);
	recycle = false;
	//PF=.93;
	PF=.5;
	reset_gaussians();
}


void scale_save(const char *fn, cv::Mat im, double scale, double shift)
{
	double sc=scale,sh=shift;
	cv::Mat cvt = cv::Mat(im.size(), CV_8UC1);
	double min= (0 - shift)/scale, max=(255-shift)/scale;
	if ((scale == -1) && (shift == -1)) {
                cv::minMaxLoc(im, &min, &max);
		sc= 255.0/(max-min);
		sh= -min*sc;
	}
        cv::Mat(im).convertTo(cvt, CV_8UC1, sc, sh);
	cout << fn << " scale: " << max << ", " << min << endl;
        cv::imwrite(fn, cvt);
}

static void log_save(const char *fn, cv::Mat im) {
	cout << "(log) ";
	cv::Mat tmp;
        cv::log(cv::Mat(im),tmp);
	scale_save(fn,tmp, -1, -1);
}
static void a_save(const char *fn, cv::Mat im) {
	cout << "(-log(1-x)) ";
	cv::Mat tmp;
        tmp = cv::Scalar(1) - im;
        cv::log(tmp,tmp);
        tmp *= -1;
	scale_save(fn,tmp, -1, -1);
}
void save_proba(const char *fn, cv::Mat im) {
	char str[1024];
	snprintf(str,1024,"log_%s",fn);
	scale_save(fn, im, -1, -1);
	log_save(str,im);
	snprintf(str,1024,"exp_%s",fn);
	a_save(str,im);
}

void EMVisi2::run(int nbIter, float smooth_amount, float smooth_threshold)
{
	if (!recycle) {
		reset_gaussians();
	}
	for (int i=0;i<nbIter;i++) {
		iterate();
	}
	if (smooth_amount>0)
		smooth(smooth_amount, smooth_threshold);
}

void EMVisi2::reset_gaussians() 
{
	for (int i=0; i<NB_VISI_GAUSSIANS;i++) {
		const float max = 90;
		const float min = 0;
		visi_g[i].init_regular( (i+1)*(max-min)/(NB_VISI_GAUSSIANS+1) + min, 30*((max-min)/NB_VISI_GAUSSIANS));
	}
	for (int i=0; i<NB_OCCL_GAUSSIANS;i++) {
		const float max = 255;
		const float min = 0;
		occl_g[i].init_regular( (i+1)*(max-min)/(NB_OCCL_GAUSSIANS+1) + min, 30*((max-min)/NB_OCCL_GAUSSIANS));
	}
	for (int i=0; i<NB_GAUSSIANS+1; i++)
		weights[i] = 1.0f/(NB_GAUSSIANS+1);
}

void EMVisi2::iterate()
{
	char str[256];
	uniform_resp=0;

	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		visi_g[i].sigma_computed=false;
	}

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		occl_g[i].sigma_computed=false;
	}


	float likelihood = 0;
	for (int y=0; y<proba.rows; y++) {
                float *po = proba.ptr<float>(y);
		float *vpo = (!visi_proba.empty() ? visi_proba.ptr<float>(y) : 0);
		const float *nccv = nccproba_v.ptr<const float>(y);
		const float *ncch = nccproba_h.ptr<const float>(y);
		const float *input = im2.ptr<const float>(y);
		const float *r = ratio.ptr<const float>(y);
		const float *dl = dL.ptr<float>(y);
		const unsigned char *m=0;
		if (!mask.empty()) m =  mask.ptr<const unsigned char>(y);
		for (int x=0; x<proba.cols; x++) {
			if ((!m) || m[x]) {
				float l = process_pixel(input+3*x, r+3*x, dl[x], nccv[x], ncch[x], 
						po+x, (vpo?vpo+x:0));
				if (save_images)
					likelihood += log(l);
				assert(!isnan(po[x]));
			} else
				po[x]=0;
		}
	}

	if (save_images)
		cout << "L=" << likelihood << endl;

	float totN = 0;
	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		visi_g[i].compute_sigma();
		totN += visi_g[i].n;
	}

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		occl_g[i].compute_sigma();
		totN += occl_g[i].n;
	}

	totN += uniform_resp;
	weights[NB_GAUSSIANS] = uniform_resp/totN;

	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		weights[i] = visi_g[i].n / totN;
		assert(weights[i]>=0 && weights[i]<=1);
	}

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		weights[NB_VISI_GAUSSIANS+i] = occl_g[i].n / totN;
		assert(weights[NB_VISI_GAUSSIANS+i]>=0 && weights[NB_VISI_GAUSSIANS+i]<=1);
	}


	recycle=true;
	iteration++;

	if (save_images) {
		sprintf(str, "proba%02d.png", iteration );
		scale_save(str, proba, -1, -1);
	}
}

float EMVisi2::process_pixel(const float *rgb, const float *frgb, const float dl, const float nccv, const float ncch, float *proba, float *visi_proba)
{
	// store responsabilities for each gaussian
	float resp[NB_GAUSSIANS+1];
	float sum_resp=0;
	float *w = weights;
	float *r = resp;

	float epsilon = 1e-40;

	// E-step: compute expectation of latent variables
	for (int i=0; i<NB_VISI_GAUSSIANS; i++) {
		*r = *w++ * visi_g[i]._proba(frgb) * dl * nccv;

		assert(!isnan(*r));
		if (*r<0) *r = 0;
		if (*r>(1-epsilon)) *r= 1-epsilon;
		assert(*r >=0);
		assert(*r <=1);
		sum_resp += *r;
		r++;
	}
	float sum_visi_resp = sum_resp;

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) {
		*r = *w++ * occl_g[i]._proba(rgb) * ncch;
		if (*r<epsilon) *r = epsilon;
		assert(!isnan(*r));
		assert(*r >=0);
		assert(*r <1);
		sum_resp += *r;
		r++;
	}
	resp[NB_GAUSSIANS] = *w * ncch *  1.0f/(255*255*255);
	sum_resp += resp[NB_GAUSSIANS];

	float sum_resp_inv = 1.0f/sum_resp;
	for (int i=0; i<NB_GAUSSIANS+1; i++)
		resp[i] *= sum_resp_inv;

	// M-step: update means and covariance matrices
	for (int i=0; i<NB_VISI_GAUSSIANS; i++) 
		visi_g[i].accumulate(frgb, resp[i]);

	for (int i=0; i<NB_OCCL_GAUSSIANS; i++) 
		occl_g[i].accumulate(rgb, resp[NB_VISI_GAUSSIANS+i]);

	uniform_resp += resp[NB_GAUSSIANS];

	*proba = sum_visi_resp * sum_resp_inv;
	if (visi_proba) *visi_proba = sum_visi_resp;

	return sum_resp;
}

bool EMVisi2::init() {
	ncc_h.setHistogram(ncc_proba_h);
	if (!ncc_h.loadHistogram("ncc_proba_h.mat") && !ncc_h.loadHistogram("../ncc_proba_h.mat")) {
		cerr << "can't load ncc_proba_h.mat histogram. Using built in distribution.\n";
	}
	ncc_v.setHistogram(ncc_proba_v);
	if (!ncc_v.loadHistogram("ncc_proba_v.mat") && !ncc_v.loadHistogram("../ncc_proba_v.mat")) {
		cerr << "can't load ncc_proba_v.mat histogram. Using built in distribution.\n";
	}
	return true;
}

int EMVisi2::setModel(const cv::Mat im1, const cv::Mat mask)
{
	if (proba.empty()) {
		dL = cv::Mat(im1.size(), CV_32FC1);
		ncc = cv::Mat(im1.size(), CV_32FC1);
		sum = cv::Mat(im1.size(), CV_32FC1);
		proba = cv::Mat(im1.size(), CV_32FC1);
		visi_proba = cv::Mat(im1.size(), CV_32FC1);
		nccproba_v = cv::Mat(im1.size(), CV_32FC1);
		nccproba_h = cv::Mat(im1.size(), CV_32FC1);
		ratio = cv::Mat(im1.size(), CV_32FC(im1.channels()));
		im1f = cv::Mat(im1.size(), CV_32FC(im1.channels()));
	}

	if (im1.channels() > 1) {
		cv::Mat gray;
                cv::cvtColor(im1, gray, COLOR_RGB2GRAY);
		fncc.setModel(gray, mask);
	} else {
		fncc.setModel(im1, mask);
	}

        im1.convertTo(im1f, im1f.type());

        this->mask = mask;
	return 0;
}

int EMVisi2::setTarget(cv::Mat target)
{
        assert(im1f.size() == target.size());
	iteration=0;

	if (target.depth() != CV_32F) {
                target.convertTo(_im2, CV_MAKETYPE(CV_32F, target.channels()));
		im2 = _im2;
	}

	assert(!ncc.empty());
	cv::Mat green2;
	if (im2.channels()>1) {
		green2 = cv::Mat(target.size(), CV_MAKETYPE(target.depth(), 1));
                cv::cvtColor(target, green2, COLOR_RGB2GRAY);
		fncc.setImage(green2);
		fncc.computeNcc(ncc_size, ncc, sum);
	} else {
		green2 = im2;
		fncc.setImage(green2);
		fncc.computeNcc(ncc_size, ncc, sum);
	}

	if (save_images) {
		scale_save("ncc.png", ncc);
		scale_save("ncc_tex.png", sum);
	}

	#pragma omp parallel sections
	{
	#pragma omp section
	ncc_v.getProba(ncc, sum, nccproba_v);
	#pragma omp section
	ncc_h.getProba(ncc, sum, nccproba_h);
	}
	if (save_images) {
		save_proba("nccproba_v.png", nccproba_v);
		save_proba("nccproba_h.png", nccproba_h);
	}

	{
	
	static float table[256][256];
	static float dtable[256][256];
	static bool table_computed=false;
	if (!table_computed) {
		table_computed=true;
		for (int i=0;i<256;i++) {
			for (int j=0;j<256;j++) {
				if (i==0 && j==0) { 
					table[i][j]=0;
					dtable[i][j]=1e-10;
				} else {
					table[i][j] = (180.0/M_PI)*atan2(i+1,j+1);
					dtable[i][j] = (180.0/M_PI)/((i+1) + (1 + (j+1)*(j+1)/(i+1)/(i+1)));

					// this also works
					/*
					float s = 64;
					table[i][j] = 45*(j+s)/(i+s);
					dtable[i][j] = 45.0/(i+s);
					*/
				}
			}
		}
	}
	int n=im1f.cols*im1f.channels();
	for (int y=0;y<im1f.rows;y++) {
		float *a = im1f.ptr<float>(y);
		float *b = im2.ptr<float>(y, 0);
		float *d = ratio.ptr<float>(y);
		float *dl = dL.ptr<float>(y);
		for (int x=0;x<n; x+=3) {
			int ia[3];
			int ib[3];
			for (int j=0; j<3; j++) {
				ia[j] = cvRound(a[x+j]);
				ib[j] = cvRound(b[x+j]);
				if (ia[j]<0) ia[j]=0;
				if (ib[j]<0) ib[j]=0;
				if (ia[j]>255) ia[j]=255;
				if (ib[j]>255) ib[j]=255;

				d[x+j] = table[ia[j]][ib[j]];
			}
			dl[x/3] = dtable[ia[0]][ib[0]]*dtable[ia[1]][ib[1]]*dtable[ia[2]][ib[2]];
			assert(dl[x/3]>0);
		}
	}
	if (save_images)
		scale_save("dL.png", dL);

	if (save_images) {
		scale_save("ratio.png", ratio);
	}
	}

	return 0;
}

#ifdef WITH_GRAPHCUT

#include <vector>
#include "graph.h"
#include "graph.hpp"
#include "maxflow.hpp"

using namespace std;
typedef Graph<float, float, float> FGraph;

/*!
  Tags connected '0' regions with an id (1-254)
  return total area
*/
static double connected_regions(cv::Mat *mask, vector<CvConnectedComp> &regions)
{
	assert(mask.channels() == 1);
	assert(mask->depth == IPL_DEPTH_8U);

	regions.clear();
	regions.reserve(254);

	int region = 1;
	double area = 0;

	for (int y=0; y<mask.rows; y++) {
		unsigned char *m = &CV_IMAGE_ELEM(mask, unsigned char, y, 0);

		for (int x=0; x<mask.cols; x++) {
			if (m[x]==0) {
				CvConnectedComp conn;

				cvFloodFill(mask, cvPoint(x,y), cvScalarAll(region), cvScalarAll(0), cvScalarAll(0),
						&conn, 8+CV_FLOODFILL_FIXED_RANGE );
				area += conn.area;
				conn.value.val[0] = region;
				regions.push_back(conn);

				region++;
				if (region==255) region=1;
			}
		}
	}
	return area;
}

static void display_err(char *e)
{
	cerr << "graph error: " << e << endl;
}


void EMVisi2::smooth(float amount, float threshold) {
	const cv::Mat wa = proba;

	// Threshold proba image
	cv::Mat gc_mask(cvGetSize(proba), IPL_DEPTH_8U, 1);
	cvSet(gc_mask, cvScalarAll(255));

	// find pixels on which graph-cut should be applied
	for (int y=1; y<proba.height()-1; y++) {
		float *p = (float *) proba.roi_row(y); 
		unsigned char *m = (unsigned char*) gc_mask.roi_row(y);
		unsigned char *im = 0;
		if (mask.is_valid())
			im = mask.roi_row(y);

		for (int x=1;x<proba.width()-1; x++)
			if ((im==0 || im[x]) // within mask and..
					&& (((p[x]>threshold) && (p[x] < (1-threshold))) // not very confident..
					|| ( fabs(p[x-1]-p[x])>.3) || (fabs(p[x-proba.step()]-p[x])>.3)	// transition
					)) {
				m[x]=0;
				/*
				if (x>0) m[x-1]=0;
				if (x<proba.cols-1) m[x+1]=0;
				if (y<proba.rows-1) m[x+mask.step()]=0;
				if (y>0) m[x-mask.step()]=0;
				*/
				m[x-1]=0;
				m[x+1]=0;
				m[x+mask.step()]=0;
				m[x-mask.step()]=0;

				// diag
				m[x+mask.step()+1]=0;
				m[x-mask.step()+1]=0;
				m[x+mask.step()-1]=0;
				m[x-mask.step()-1]=0;
			}
	}

	// segment connected uncertain areas
	vector<CvConnectedComp> regions;
	connected_regions(gc_mask, regions);
	if (save_images) {
		cv::Mat imreg(cvGetSize(gc_mask), IPL_DEPTH_8U, 3);

		CvMat *lut = cvCreateMat(1,256, CV_8UC3);
		CvRNG rng = cvRNG();
		cvRandArr(&rng, lut, CV_RAND_UNI, cvScalarAll(0), cvScalarAll(255));
		unsigned char *c = lut->data.ptr;
		//c[0] = c[1] = c[2] = 0;
		c[255*3] = c[255*3+1] = c[255*3+2] = 0;
		for (int y=0; y<imreg.height(); y++) {
			unsigned char *dst = imreg.roi_row(y);
			unsigned char *src = gc_mask.roi_row(y);
			for (int x=0; x<imreg.width(); x++) 
				for (int i=0; i<3; i++) 
					dst[x*3+i] = c[src[x]*3+i];
		}
		cvReleaseMat(&lut);
		cvSaveImage("regions.png", imreg);
	}

	// allocate the graph. Note: worst case memory scenario.
	int n_nodes= gc_mask.width()*gc_mask.height();
	int n_edges = 2*((wa.cols)*(wa.rows-1) + (wa.cols-1)*wa.rows);
	FGraph *g = new FGraph(n_nodes, n_edges, display_err);
	int *nodesid = new int[n_nodes];

	// try to run graphcut on all regions
	for (unsigned i=0; i<regions.size(); i++) {

		CvConnectedComp &r = regions[i];

		/*
		cout << "Region " << i << ": area=" << r.area << ", " 
			<< r.rect.width << "x" << r.rect.height << endl;
		*/

		g->reset();
		//g->add_node((int)r.area);
		g->add_node(r.rect.width * r.rect.height);
		for (int i=0; i<r.rect.width+1;i++) nodesid[i]=-1;

		int next_node = 0;

		unsigned rval = (unsigned)r.value.val[0];

		for (int y=r.rect.y; y<r.rect.y+r.rect.height; y++) {
			unsigned char *m = (unsigned char*) gc_mask.roi_row(y);
			int *row_id = nodesid + (1+y-r.rect.y)*(r.rect.width+1)+1;
			row_id[-1]=-1;

			const float c = amount;

			float *proba_l = (float *)proba.roi_row(y);
			float *visi_proba_l = (float *)visi_proba.roi_row(y);

			for (int x=r.rect.x; x<r.rect.x+r.rect.width; x++) {
				if (m[x] == rval) {
					// add a new node
					*row_id = next_node;

					// terminal weights
					float wap = proba_l[x];
					float vp = visi_proba_l[x];
					g->add_tweights(next_node, 
							//-logf(PF*wap), -logf((1-PF)*(1-wap)));
							-log(PF*vp), -log((1-PF)*(vp/wap - vp)));

					// fill connectivity edges ..

					// .. up ?
					int up_id = row_id[-(r.rect.width+1)]; 
					if (up_id>=0) {
						// the node exists. Link it.
						g->add_edge(next_node, up_id, c, c);
					}

					// .. left ?
					int left_id = row_id[-1];
					if (left_id >= 0) {
						// the node exists. Link it.
						g->add_edge(next_node, left_id, c, c);
					}

					// .. up+left ?
					int upleft_id = row_id[-(r.rect.width+1)-1];
					if (upleft_id >= 0) {
						// the node exists. Link it.
						g->add_edge(next_node, upleft_id, c, c);
					}

					// .. up+right ?
					int upright_id = row_id[-(r.rect.width+1)+1];
					if (upright_id >= 0) {
						// the node exists. Link it.
						g->add_edge(next_node, upright_id, c, c);
					}

					next_node++;
				} else {
					*row_id = -1;
				}
				row_id++;
			}
		}

		// solve maxflow
		g->maxflow();

		// write result back
		for (int y=r.rect.y; y<r.rect.y+r.rect.height; y++) {
			float *p = (float *)proba.roi_row(y);
			int *row_id = nodesid + (1+y-r.rect.y)*(r.rect.width+1)+1;

			for (int x=r.rect.x; x<r.rect.x+r.rect.width; x++) {
				if (*row_id >= 0) {
					p[x] = (g->what_segment(*row_id) == FGraph::SOURCE ? 0 : 1);
				}
				row_id++;
			}
		}
	}
	delete[] nodesid;
	delete g;
}
#else
void EMVisi2::smooth(float, float) {
}
#endif


NccHisto::NccHisto() : lut(0), deleteme(0) {
}

NccHisto::~NccHisto() {
	if (deleteme) delete[] deleteme;
}

void NccHisto::setHistogram(const float *histo)
{
	int n =(NTEX+1)*(NCORR+1);
	lut = deleteme = new float[n];
	memcpy(lut, histo, sizeof(float)*n);
}

bool NccHisto::loadHistogram(const char *filename)
{
	CvGrowMat *histo = CvGrowMat::loadMat(filename, CV_32FC1);
	if (!histo) return false;
	if (histo->rows != (NCORR+1) || histo->cols!=(NTEX+1)) {
		std::cerr << filename << ": wrong matrix size.\n";
		return false;
	}
	float *_lut = new float[histo->rows*histo->cols];
	lut = _lut;
	if (deleteme) delete[] deleteme;
	deleteme = lut;

	for (int i=0;i<histo->rows;i++)
		for (int j=0;j<histo->cols;j++)
			_lut[i*(NTEX+1)+j] = cvGetReal2D(histo, i, j);

	delete histo;
	return true;
}

bool NccHisto::saveHistogram(const char *filename)
{
	CvMat m;
	cvInitMatHeader(&m, NCORR+1, NTEX+1, CV_32FC1, lut);
	return CvGrowMat::saveMat(&m, filename);
}

void NccHisto::getProba(const cv::Mat ncc, cv::Mat proba)
{
	if (lut==0) loadHistogram();
	assert(lut);
	if (lut==0) return;
	assert(ncc.channels()==3);
	assert(ncc.cols == proba.cols && ncc.rows==proba.rows);
	assert(proba.channels()==1);

	const int w=ncc.cols;
	const int h=ncc.rows;

	for (int y=0; y<h;y++) {
		float *dst = proba.ptr<float>(y);
		const float *src = ncc.ptr<float>(y);
		for (int x=0;x<w;x++) {
			dst[x] = lut[lut_idx(src[x*3], src[x*3+1])];
		}
	}
}

void NccHisto::getProba(const cv::Mat ncc, const cv::Mat sumstdev, cv::Mat proba)
{
	if (lut==0) loadHistogram();
	assert(lut);
	if (lut==0) return;
	assert(ncc.channels()==1);
	assert(sumstdev.channels()==1);
	assert(ncc.cols == proba.cols && ncc.rows==proba.rows);
	assert(proba.channels()==1);

	const int w=ncc.cols;
	const int h=ncc.rows;

	for (int y=0; y<h;y++) {
		float *dst = proba.ptr<float>(y);
		const float *src = ncc.ptr<float>(y);
		const float *sum = sumstdev.ptr<float>(y);
		for (int x=0;x<w;x++) {
			dst[x] = lut[lut_idx(src[x], sum[x])];
		}
	}
}

void NccHisto::initEmpty() {
	int n =(NTEX+1)*(NCORR+1);
	lut = deleteme = new float[n];
	for (int i=0; i<n; i++) lut[i] = 0.0f;
	nelem=0;
}

void NccHisto::normalize(float bias)
{
	int n =(NTEX+1)*(NCORR+1);
	float div = nelem + n*bias;
	for (int i=0; i<n; i++) lut[i] = (lut[i]+bias) / div;
}

