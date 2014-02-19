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
#ifndef IMSTAT_H
#define IMSTAT_H

// for memset/memcpy
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;

//#define DEBUG_MULTIGAUSSIAN3 1
//#define MG3_METHOD_INCREM

static inline float exp_table(float f) 
{
	static float *table=0;
	const unsigned n = 4096;
	const float range=300;
	if (!table) {
		table = new float[n];
		for (unsigned i=0; i<n; i++) {
			table[i] = exp(-(float)(range*i)/(float)n);
		}
	}
	//assert(!isnan(f));
	unsigned idx = (unsigned)(-f*(n/range) + .5f);
	return table[(idx<n) ? idx : n-1];
}


template <class T>
class Accumulator {
public:
	virtual ~Accumulator() {}
	virtual void accumulate(const T *data, float w) =0;
	virtual void clear()=0;
	virtual float proba(T *data)=0;
	virtual int area()=0;

	void stat_mask(const IplImage *src, const IplImage *mask=0)
	{
		const int n=src->nChannels;
// pragma omp parallel for schedule(static)
		for (int y=0;y<src->height;y++) {
			T *s = (T*) &CV_IMAGE_ELEM(src, T, y, 0);
			unsigned char *m = 0;
			if (mask) m=&CV_IMAGE_ELEM(mask, unsigned char, y, 0);
			for (int x=0;x<src->width;x++) {
				if (mask==0 || m[x]) accumulate(s+x*n,1);
			}
		}
	}

	double stat_weight(const IplImage *src, const IplImage *weight, float wmul, float wshift, const IplImage *mask=0)
	{
		const int n=src->nChannels;
		double r=0;
// pragma omp parallel for schedule(static)
		for (int y=0;y<src->height;y++) {
			T *s = (T*) &CV_IMAGE_ELEM(src, T, y, 0);
			unsigned char *m = 0;
			if (mask) m=&CV_IMAGE_ELEM(mask, unsigned char, y, 0);
			float *wptr = &CV_IMAGE_ELEM(weight, float, y, 0);
			for (int x=0;x<src->width;x++) {
				if (mask==0 || m[x]) {
					float w = wptr[x]*wmul + wshift;
					assert(w>=0);
					assert(w<=1);
					if (w>0.0001)
					accumulate(s+x*n, w);
					r+=w;
				}
			}
		}
		return r;
	}


	float proba_image(const IplImage *src, IplImage *dst, const IplImage *mask=0)
	{
		const int n=src->nChannels;
		float sum=0;
// pragma omp parallel for schedule(static)
		for (int y=0;y<src->height;y++) {
			T *s = (T*) &CV_IMAGE_ELEM(src, T, y, 0);
			float *d = &CV_IMAGE_ELEM(dst, float,y,0);
			unsigned char *m=0;
		       	if (mask) m = &CV_IMAGE_ELEM(mask, unsigned char, y, 0);
			for (int x=0;x<src->width;x++) {
				if (mask==0 || m[x]) {
					sum += d[x] = proba(s+x*n);
					assert(d[x]>=0);
					assert(d[x]<=1);
				} else
					d[x]=0;
			}
		}
		return sum;
	}
};

template <class T>
class MultiGaussian3 : public Accumulator<T> {
public:
	float n;
	float sigma[3][3];
	double new_sigma[3][3];
	double new_mean[3];
	double new_n;
	float one_over_sq_det;
	float two_ln_sq_det;
	float mean[3];
	bool sigma_computed;


	MultiGaussian3() { 
		memset(sigma, 0, sizeof(sigma));
		sigma[0][0]=sigma[1][1]=sigma[2][2]=1;
		memset(mean,0,sizeof(mean));
		one_over_sq_det=1;
		two_ln_sq_det=2;

		clear();
       	}

	void clear() {
		n=0;
		sigma_computed=false;

		memset(new_sigma, 0, sizeof(new_sigma));
#ifdef MG3_METHOD_INCREM
		new_sigma[0][0]=new_sigma[1][1]=new_sigma[2][2]=1;
#endif
		memset(new_mean,0, sizeof(new_mean));

		new_n=0;
	}

	virtual void accumulate(const T *x, float w) {

#ifdef MG3_METHOD_INCREM
		float n_pre=new_n;
		new_n+=w;

		float inv_new_n = 1.0f/new_n;
		float d[3];
		float rn = w*inv_new_n;
		float nm1on = n_pre*inv_new_n;

		for (int i=0; i<3; i++)
			d[i] = x[i]-new_mean[i];

		float rnd0 = rn*d[0];
		float rnd1 = rn*d[1];
		new_sigma[0][0] = nm1on*new_sigma[0][0] + rnd0*d[0];
		new_sigma[0][1] = nm1on*new_sigma[0][1] + rnd0*d[1];
		new_sigma[0][2] = nm1on*new_sigma[0][2] + rnd0*d[2];
		new_sigma[1][1] = nm1on*new_sigma[1][1] + rnd1*d[1];
		new_sigma[1][2] = nm1on*new_sigma[1][2] + rnd1*d[2];
		new_sigma[2][2] = nm1on*new_sigma[2][2] + rn*d[2]*d[2];
		
		for (int i=0;i<3;i++) 
			new_mean[i] = nm1on*new_mean[i] + rn * x[i];
#else
		new_n+=w;
		double wx0 = w*x[0];
		double wx1 = w*x[1];
		double wx2 = w*x[2];
		new_mean[0] += wx0;
		new_mean[1] += wx1;
		new_mean[2] += wx2;
		new_sigma[0][0] += wx0*x[0];
		new_sigma[0][1] += wx0*x[1];
		new_sigma[0][2] += wx0*x[2];
		new_sigma[1][1] += wx1*x[1];
		new_sigma[1][2] += wx1*x[2];
		new_sigma[2][2] += wx2*x[2];
#endif
	}

	virtual int area() { return (int)n; }
	virtual float proba(T *a) {
		if (!sigma_computed) {
			if (new_n>1)
				compute_sigma();
		}
		if (n<1) return 1e-20;
		return _proba(a);
	}

	float _proba(const T *a) {

		float d[3];
	       	d[0] = a[0] - mean[0];
	       	d[1] = a[1] - mean[1];
	       	d[2] = a[2] - mean[2];
		/*
		float sv[3];
		for (int i=0;i<3; i++) 
			sv[i] = sigma[i][0]*v[0] + sigma[i][1]*v[1] + sigma[i][2]*v[2];
		float vsv = v[0]*sv[0] + v[1]*sv[1] + v[2]*sv[2];
		*/
		float t1 = d[0]*d[0];
		float t9 = d[1]*d[1];
		float t14 = d[2]*d[2];
		float t15 = d[0]*d[1]*sigma[0][1] +d[0]*d[2]*sigma[0][2] + d[1]*d[2]*sigma[1][2];
		float vsv = t1*sigma[0][0]
			+t9*sigma[1][1]
			+t14*sigma[2][2]
			+t15+t15;

		assert(!isnan(vsv));
		assert(!isnan(one_over_sq_det));
		if (vsv<0) vsv=0;

		// this constant is 1/(2Pi)^(3/2)
		float p = .0634936359342409f*one_over_sq_det*exp_table(-.5f*vsv);

		float epsilon = 1e-30;
		if (p<epsilon) p=epsilon;

		assert(!isnan(p));

		if (p>1-epsilon) {
			//std::cout << "Warning: clamping proba " << p << " to 1.\n";
			p=1-epsilon;
		}
		assert((p>0) && (p<=1));
		return p;
	}

	float dist_to_mean(T *a) {

		float d[3];
	       	d[0] = a[0] - mean[0];
	       	d[1] = a[1] - mean[1];
	       	d[2] = a[2] - mean[2];
		return sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
	}


	float log_proba(T *a) {

		float d[3];
	       	d[0] = a[0] - mean[0];
	       	d[1] = a[1] - mean[1];
	       	d[2] = a[2] - mean[2];

		float sv[3];
		/*
		for (int i=0;i<3; i++) 
			sv[i] = sigma[i][0]*v[0] + sigma[i][1]*v[1] + sigma[i][2]*v[2];
		float vsv = v[0]*sv[0] + v[1]*sv[1] + v[2]*sv[2];
		*/

		float t1 = d[0]*d[0];
		float t9 = d[1]*d[1];
		float t14 = d[2]*d[2];
		float t15 = d[0]*d[1]*sigma[0][1] +d[0]*d[2]*sigma[0][2] + d[1]*d[2]*sigma[1][2];
		float vsv = t1*sigma[0][0]
			+t9*sigma[1][1]
			+t14*sigma[2][2]
			+t15+t15;

		return -(vsv - two_ln_sq_det);
	}



	void compute_sigma_no_reset() {
		//if (new_n<5) return;

		float s[3][3];

#ifndef MG3_METHOD_INCREM

		for (int i=0; i<3; i++) 
			new_mean[i] = new_mean[i]/new_n;

		for (int j=0; j<3; j++)
			for (int i=j; i<3; i++) {
				s[i][j] = s[j][i] = new_sigma[j][i]/new_n - new_mean[i]*new_mean[j];
			}

		for (int j=0; j<3; j++)
			for (int i=0; i<3; i++) 
				new_sigma[j][i] = s[j][i];
#endif
		memset(s,0,sizeof(s));


		for (int i=0; i<3; i++) {
			mean[i] = new_mean[i];
			for (int j=0; j<3; j++) {
				if (i<j)
					s[i][j] = new_sigma[i][j];
				else
					s[i][j] = new_sigma[j][i];
			}
		}

		float det = inverse3x3(s,sigma);

		if (!!isnan(det) || det<1e-5) {
			sigma_computed=false;
			n=0;
			one_over_sq_det= 1;
			memset(sigma,0,sizeof(sigma));
			sigma[0][0]=1;
			sigma[1][1]=1;
			sigma[2][2]=1;
			two_ln_sq_det = 2;
		} else {
			one_over_sq_det = 1/sqrtf(det);
			assert(!isnan(one_over_sq_det));
			two_ln_sq_det = 2*log(sqrtf(det));
			sigma_computed=true;
			n = new_n;
		}

	}

	void reset_new_sigma() {
		new_n=0;
		memset(new_sigma, 0, sizeof(new_sigma));
#ifdef MG3_METHOD_INCREM
		new_sigma[0][0]=new_sigma[1][1]=new_sigma[2][2]=1;
#endif
		memset(new_mean,0, sizeof(new_mean));
	}

	void compute_sigma() {
		compute_sigma_no_reset();
		reset_new_sigma();
	}

	float inverse3x3(const float m[3][3], float dst[3][3])
	{
		float t4 = m[0][0]*m[1][1];
		float t6 = m[0][0]*m[1][2];
		float t8 = m[0][1]*m[1][0];
		float t10 = m[0][2]*m[1][0];
		float t12 = m[0][1]*m[2][0];
		float t14 = m[0][2]*m[2][0];
		float t16 = (t4*m[2][2]-t6*m[2][1]-t8*m[2][2]+t10*m[2][1]+t12*m[1][2]-t14*m
				[1][1]);
		float t17 = 1/t16;
		dst[0][0] = (m[1][1]*m[2][2]-m[1][2]*m[2][1])*t17;
		dst[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1])*t17;
		dst[0][2] = -(-m[0][1]*m[1][2]+m[0][2]*m[1][1])*t17;
		dst[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0])*t17;
		dst[1][1] = (m[0][0]*m[2][2]-t14)*t17;
		dst[1][2] = -(t6-t10)*t17;
		dst[2][0] = -(-m[1][0]*m[2][1]+m[1][1]*m[2][0])*t17;
		dst[2][1] = -(m[0][0]*m[2][1]-t12)*t17;
		dst[2][2] = (t4-t8)*t17;
		return t16;
	}

	void init_regular(float m, float s) 
	{
		mean[0] = mean[1] = mean[2] = m;

		memset(sigma,0,sizeof(sigma));
		sigma[0][0]=sigma[1][1]=sigma[2][2] = 1/s;

		float det = s*s*s;
		one_over_sq_det = 1/sqrtf(det);
		assert(!isnan(one_over_sq_det));
		two_ln_sq_det = 2*log(sqrtf(det));
		sigma_computed=true;
		n = 0;
	}

};

#endif
