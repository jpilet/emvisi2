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
 * Normalized cross-correlation computation using integral images.
 * Julien Pilet, 2008.
 */
#include <pmmintrin.h>
#include "fwncc.h"

//#define FWNCC_MAIN

FNcc::FNcc() {
	width=0;
	integral=0;
	ncc=0;
	dint=0;
	a=b=0;
	mask=0;
	mask_integral=0;
}

FNcc::~FNcc() {
	if (integral) {
		delete[] integral;
		delete[] ncc;
	}
	if (dint) delete[] dint;
	cvReleaseImage(&this->mask);
	if (mask_integral) cvReleaseImage(&mask_integral);
	cvReleaseImage(&b);
}

void FNcc::setModel(const IplImage *b, const IplImage *mask) {

	cvReleaseImage(&this->mask);
	cvReleaseImage(&mask_integral);
	if (mask) {
		this->mask=cvCloneImage(mask);
		mask_integral = cvCreateImage(cvSize(mask->width+1, mask->height+1), IPL_DEPTH_32S, 1);
		cvSetZero(mask_integral);
		for (int y=0; y<mask->height;y++) {
			int *m = &CV_IMAGE_ELEM(mask_integral, int, y+1, 1);
			int *mup = &CV_IMAGE_ELEM(mask_integral, int, y, 1);
			for (int x=0; x<mask->width; x++) {
				m[x] = mup[x]-mup[x-1]+m[x-1] +
					(CV_IMAGE_ELEM(mask, unsigned char, y, x) ? 1:0);
			}
		}
	}

	cvReleaseImage(&this->b);
	this->b=cvCloneImage(b);
	//this->b=b;

	assert(b->nChannels==1);
	assert(b->depth == IPL_DEPTH_8U);

	width = b->width+1;
	height = b->height+1;

	if (integral) delete[] integral;
	integral = new CSum[width*height];
	if (ncc) delete[] ncc;
	ncc = new CSumf[width*height];
	if (dint) delete[] dint;
	dint = new DSum[width*height];

	memset(dint, 0, sizeof(DSum)*width);
	memset(integral, 0, sizeof(CSum)*width);

	CSum *p= integral + width + 1;

	for (int y=0; y<b->height; y++, p+=width) {
		memset(p-1,0, sizeof(CSum));

		unsigned char *line = &CV_IMAGE_ELEM(b, unsigned char, y, 0);
		unsigned char *m=0;
		if (mask) m = &CV_IMAGE_ELEM(mask, unsigned char, y, 0);

		for (int x=0; x<b->width; x++) {
			if (m && m[x]==0) line[x]=0;
			p[x].b = p[x-1].b + line[x];
			p[x].b2 = p[x-1].b2 + line[x]*line[x];
		}
		for (int x=0; x<b->width; x++) {
			p[x].b += p[x-width].b;
			p[x].b2 += p[x-width].b2;
		}
	}
}

void FNcc::setImage(IplImage *a) {

	this->a=a;

	assert(a->width+1 == width && a->height+1 == height);
	assert(a->nChannels==1);
	assert(a->depth==IPL_DEPTH_8U);

	CSum *p= integral + width + 1;

	for (int y=0; y<a->height; y++, p+=width) {

		unsigned char *aline = &CV_IMAGE_ELEM(a, unsigned char, y, 0);
		unsigned char *bline = &CV_IMAGE_ELEM(b, unsigned char, y, 0);
		unsigned char *m=0;
		if (mask) m = &CV_IMAGE_ELEM(mask, unsigned char, y, 0);

		for (int x=0; x<a->width; x++) {
			unsigned char ax = ((m && m[x]==0) ? 0 : aline[x]);
			p[x].a_a2 = _mm_add_pd(
					_mm_sub_pd(p[x-1].a_a2, p[x-width-1].a_a2), 
					_mm_add_pd(p[x-width].a_a2, 
						_mm_set_pd(ax*ax,ax)));
			p[x].ab = p[x-1].ab - p[x-width-1].ab + p[x-width].ab + ax*bline[x];
		}
	}
}
	
static inline float texeval(float a, float b, float n) {
	//return a*b*n*n;
	return (a+b)*n;
}

#define FMAX(a,b) (a>b ? a:b)

void FNcc::computeNcc(int winSize, IplImage *dst, IplImage *sumvar) {
	if (mask) computeNcc_mask(winSize, dst, sumvar);
	else computeNcc_nomask(winSize, dst, sumvar);
}

void FNcc::computeNcc_nomask(int winSize, IplImage *dst, IplImage *sumvar) {

	assert(dst==0 || (dst->width+1 == width && dst->height+1 == height));
	assert(sumvar == 0 || sumvar->depth==IPL_DEPTH_32F);
	const int w2 = winSize/2;

//pragma omp parallel for 
	for (int y=0; y<height-1; y++) {
		CSumf *d = ncc + y*width;
		int yup = MAX(0,y-w2-1);
		int ydown = MIN(height-1,y+w2);
		CSum *up = integral + yup*width;
		CSum *down = integral + ydown*width;

		float *dline = 0;
		if (dst) dline = &CV_IMAGE_ELEM(dst, float, y, 0);
		float *sv=0;
		if (sumvar) sv = &CV_IMAGE_ELEM(sumvar, float, y, 0);

		CSum *upl = up - w2 - 1;
		CSum *upr = up + w2;
		CSum *downl = down -w2 -1;
		CSum *downr = down +w2;

		float sqrtN = 1.0f/float(2*w2+1);
		float N=sqrtN*sqrtN;

		for (int x=0; x<w2+1; x++) {
			//d[x,y] = i[x+w2, y+w2] - i[x-w2-1, y+w2] - i[x+w2,y-w2-1] + i[x-w2-1,y-w2-1]
			
			N = 1.0f/MAX((x+w2)*(ydown-yup),1);
			sqrtN = sqrtf(N);

			CSum sd;
			sd.setaddsub4(up[0], upr[x], downr[x], down[0]);
			CSumf s;

			s = sd;

			float num = s.ab - N*s.a*s.b;
			float vara = s.a2 - N*s.a*s.a;
			float varb = s.b2 - N*s.b*s.b;
			d[x] = s;
			if (sv) {
				vara = sqrtf(FMAX(0,vara));
				varb = sqrtf(FMAX(0,varb));
				if (dline) {
					float f = vara*varb;
					dline[x] = (f>1 ? num/(vara*varb) : 0);
				}
				sv[x] = texeval(vara,varb, sqrtN); //(vara+varb)*sqrtN;
			} else if (dline) {	
				float f = vara*varb;
				if (f<1) dline[x]=0;
				else
					_mm_store_ss(dline+x, _mm_mul_ss(_mm_set_ss(num), _mm_rsqrt_ss(_mm_set_ss(f))));
			}

		}

		N = 1.0f/((float)(ydown-yup)*(w2*2+1));
		sqrtN = sqrtf(N);
		for (int x=w2+1; x<dst->width-w2; x++) {

			//d[x,y] = i[x+w2, y+w2] - i[x-w2-1, y+w2] - i[x+w2,y-w2-1] + i[x-w2-1,y-w2-1]
			CSum sd;
			sd.setaddsub4(upl[x], upr[x], downr[x], downl[x]);
			const CSum s = sd;
			d[x] = s;

			const float num = s.ab - N*s.a*s.b;
			float vara = s.a2 - N*s.a*s.a;
			float varb = s.b2 - N*s.b*s.b;
			if (sv) {
				vara = sqrtf(FMAX(0,vara));
				varb = sqrtf(FMAX(0,varb));
				if (dline) {
					float f = vara*varb;
					dline[x] = (f>1 ? num/(vara*varb) : 0);
				}
				//sv[x] = (vara+varb)*sqrtN;
				sv[x] = texeval(vara,varb, sqrtN); //(vara+varb)*sqrtN;
			} else if (dline) {	
				const float f = vara*varb;
				if (f<1) dline[x]=0;
				else
				_mm_store_ss(dline+x, _mm_mul_ss(_mm_set_ps1(num), _mm_rsqrt_ss(_mm_set_ps1(f))));
			}
		}
		for (int x=dst->width-w2; x<dst->width; x++) {

			//d[x,y] = i[x+w2, y+w2] - i[x-w2-1, y+w2] - i[x+w2,y-w2-1] + i[x-w2-1,y-w2-1]

			N = 1.0f/MAX((ydown-yup)*(dst->width-(x-w2-1)),1);
			sqrtN = sqrtf(N);

			CSum sd;
			sd.setaddsub4(upl[x],up[dst->width], down[dst->width], downl[x]);
			CSum s=sd;
			d[x]=s;

			float num = s.ab - N*s.a*s.b;
			float vara = s.a2 - N*s.a*s.a;
			float varb = s.b2 - N*s.b*s.b;
			if (sv) {
				vara = sqrtf(FMAX(0,vara));
				varb = sqrtf(FMAX(0,varb));
				if (dline) {
					float f = vara*varb;
					dline[x] = (f>1 ? num/(vara*varb) : 0);
				}
				//sv[x] = (vara+varb)*sqrtN;
				sv[x] = texeval(vara,varb, sqrtN); //(vara+varb)*sqrtN;
			} else if (dline) {	
				float f = vara*varb;
				if (f<1) dline[x]=0;
				else
				_mm_store_ss(dline+x, _mm_mul_ss(_mm_set_ps1(num), _mm_rsqrt_ss(_mm_set_ps1(f))));
			}

		}

	}
}

void FNcc::computeNcc_mask(int winSize, IplImage *dst, IplImage *sumvar) {

	assert(dst==0 || (dst->width+1 == width && dst->height+1 == height));
	assert(sumvar == 0 || sumvar->depth==IPL_DEPTH_32F);
	assert(mask->depth==IPL_DEPTH_8U);
	assert(mask->width == (width-1) && mask->height == (height-1));
	assert(mask->nChannels==1);
	assert(mask_integral->depth==IPL_DEPTH_32S && mask_integral->nChannels==1);

	const int w2 = winSize/2;


//#pragma omp parallel for
	for (int y=0; y<height-1; y++) {
		CSumf *d = ncc + y*width;
		int yup = MAX(0,y-w2-1);
		int ydown = MIN(height-1,y+w2);
		CSum *up = integral + yup*width;
		CSum *down = integral + ydown*width;

		float *dline = 0;
		if (dst) dline = &CV_IMAGE_ELEM(dst, float, y, 0);
		float *sv=0;
		if (sumvar) sv = &CV_IMAGE_ELEM(sumvar, float, y, 0);

		CSum *upl = up - w2 - 1;
		CSum *upr = up + w2;
		CSum *downl = down -w2 -1;
		CSum *downr = down +w2;

		float sqrtN = 1.0f/float(2*w2+1);
		float N=sqrtN*sqrtN;

		unsigned char *m = &CV_IMAGE_ELEM(mask, unsigned char, y, 0);

		int *mask_up = &CV_IMAGE_ELEM(mask_integral, int, yup, 0);
		int *mask_down = &CV_IMAGE_ELEM(mask_integral, int, ydown, 0);

		for (int x=0; x<w2+1; x++) {
			if (m[x]==0) {
				if(dline) dline[x]=0;
				if(sv) sv[x]=0;
				continue;
			}
			//d[x,y] = i[x+w2, y+w2] - i[x-w2-1, y+w2] - i[x+w2,y-w2-1] + i[x-w2-1,y-w2-1]
			
			int n = mask_up[0] - mask_down[0] - mask_up[x+w2] + mask_down[x+w2];
			assert(n>0);
			N = 1.0f/n;
			sqrtN = sqrtf(N);

			CSum sd;
			sd.setaddsub4(up[0], upr[x], downr[x], down[0]);
			CSumf s;

			s = sd;

			float num = s.ab - N*s.a*s.b;
			float vara = s.a2 - N*s.a*s.a;
			float varb = s.b2 - N*s.b*s.b;
			d[x] = s;
			if (sv) {
				vara = sqrtf(FMAX(0,vara));
				varb = sqrtf(FMAX(0,varb));
				if (dline) {
					float f = vara*varb;
					dline[x] = (f>1 ? num/(vara*varb) : 0);
				}
				//sv[x] = (vara+varb)*sqrtN;
				sv[x] = texeval(vara,varb, sqrtN); //(vara+varb)*sqrtN;
			} else if (dline) {	
				float f = vara*varb;
				if (f<1) dline[x]=0;
				else
					_mm_store_ss(dline+x, _mm_mul_ss(_mm_set_ss(num), _mm_rsqrt_ss(_mm_set_ss(f))));
			}

		}

		N = 1.0f/((float)(ydown-yup)*(w2*2+1));
		sqrtN = sqrtf(N);
		for (int x=w2+1; x<dst->width-w2; x++) {

			if (m[x]==0) {
				if(dline) dline[x]=0;
				if(sv) sv[x]=0;
				continue;
			}
			//d[x,y] = i[x+w2, y+w2] - i[x-w2-1, y+w2] - i[x+w2,y-w2-1] + i[x-w2-1,y-w2-1]
			
			int n = mask_up[x-w2-1] - mask_down[x-w2-1] - mask_up[x+w2] + mask_down[x+w2];
			assert(n>0);
			N = 1.0f/n;
			sqrtN = sqrtf(N);
			CSum sd;
			sd.setaddsub4(upl[x], upr[x], downr[x], downl[x]);
			const CSum s = sd;
			d[x] = s;

			const float num = s.ab - N*s.a*s.b;
			float vara = s.a2 - N*s.a*s.a;
			float varb = s.b2 - N*s.b*s.b;
			if (sv) {
				vara = sqrtf(FMAX(0,vara));
				varb = sqrtf(FMAX(0,varb));
				if (dline) {
					float f = vara*varb;
					dline[x] = (f>1 ? num/(vara*varb) : 0);
				}
				//sv[x] = (vara+varb)*sqrtN;
				sv[x] = texeval(vara,varb, sqrtN); //(vara+varb)*sqrtN;
			} else if (dline) {	
				const float f = vara*varb;
				if (f<1) dline[x]=0;
				else
				_mm_store_ss(dline+x, _mm_mul_ss(_mm_set_ps1(num), _mm_rsqrt_ss(_mm_set_ps1(f))));
			}
		}
		for (int x=dst->width-w2; x<dst->width; x++) {

			if (m&&(m[x]==0)) {
				if(dline) dline[x]=0;
				if(sv) sv[x]=0;
				continue;
			}
			//d[x,y] = i[x+w2, y+w2] - i[x-w2-1, y+w2] - i[x+w2,y-w2-1] + i[x-w2-1,y-w2-1]

			int n = mask_up[x-w2-1] - mask_down[x-w2-1] - mask_up[dst->width] + mask_down[dst->width];
			assert(n>0);
			N = 1.0f/n;
			sqrtN = sqrtf(N);

			CSum sd;
			sd.setaddsub4(upl[x],up[dst->width], down[dst->width], downl[x]);
			CSum s=sd;
			d[x]=s;

			float num = s.ab - N*s.a*s.b;
			float vara = s.a2 - N*s.a*s.a;
			float varb = s.b2 - N*s.b*s.b;
			if (sv) {
				vara = sqrtf(FMAX(0,vara));
				varb = sqrtf(FMAX(0,varb));
				if (dline) {
					float f = vara*varb;
					dline[x] = (f>1 ? num/(vara*varb) : 0);
				}
				//sv[x] = (vara+varb)*sqrtN;
				sv[x] = texeval(vara,varb, sqrtN); //(vara+varb)*sqrtN;
			} else if (dline) {	
				float f = vara*varb;
				if (f<1) dline[x]=0;
				else
				_mm_store_ss(dline+x, _mm_mul_ss(_mm_set_ps1(num), _mm_rsqrt_ss(_mm_set_ps1(f))));
			}

		}

	}
}

FNccMC::FNccMC() {
	for (int i=0; i<3; i++) {
		a[i] = b[i] = 0;
		tmp_dst[i]=0;
		tmp_sumvar[i]=0;
		tmp_flt1[i]=0;
		tmp_flt2[i]=0;
	}
}

FNccMC::~FNccMC() {
	for (int i=0; i<3; i++) {
		if (a[i]) cvReleaseImage(a+i);
		if (b[i]) cvReleaseImage(b+i);
		if (tmp_dst[i]) cvReleaseImage(tmp_dst+i);
		if (tmp_sumvar[i]) cvReleaseImage(tmp_sumvar+i);
		if (tmp_flt1[i]) cvReleaseImage(tmp_flt1+i);
		if (tmp_flt2[i]) cvReleaseImage(tmp_flt2+i);
	}
}

void FNccMC::setModel(const IplImage *im, const IplImage *mask) 
{
	assert(im->depth == IPL_DEPTH_8U);
	assert(im->nChannels <= 3);
	for (int i=0; i<im->nChannels; i++) {
		if (a[i]) cvReleaseImage(a+i);
		a[i] = cvCreateImage(cvGetSize(im), IPL_DEPTH_8U, 1);
		if (b[i]) cvReleaseImage(b+i);
		b[i] = cvCreateImage(cvGetSize(im), IPL_DEPTH_8U, 1);
		/*
		if (tmp_dst[i]) cvReleaseImage(tmp_dst+i);
		tmp_dst[i] = cvCreateImage(cvGetSize(im), IPL_DEPTH_8U, 1);
		if (tmp_sumvar[i]) cvReleaseImage(tmp_sumvar+i);
		tmp_sumvar[i] = cvCreateImage(cvGetSize(im), IPL_DEPTH_8U, 1);
		*/
	}
	cvSplit(im,b[0],b[1],b[2],0);

#pragma omp parallel for
	for (int i=0; i<im->nChannels; i++) {
		ncc[i].setModel(b[i], mask);
	}
}

void FNccMC::setImage(const IplImage *im)
{
	assert(im->depth == IPL_DEPTH_8U);
	assert(im->nChannels <= 3);

	cvSplit(im, a[0], a[1], a[2],0);

#pragma omp parallel for
	for (int i=0; i<im->nChannels; i++) {
		ncc[i].setImage(a[i]);
	}
}

void FNccMC::computeNcc(int windowSize, IplImage *dst, IplImage *sumvar)
{
#pragma omp parallel for
	for (int i=0; i< dst->nChannels; i++) {
		cvReleaseImage(tmp_flt1+i);
		cvReleaseImage(tmp_flt2+i);
		tmp_flt1[i] = cvCreateImage(cvGetSize(dst), IPL_DEPTH_32F, 1);
		if (sumvar)
			tmp_flt2[i] = cvCreateImage(cvGetSize(dst), IPL_DEPTH_32F, 1);

		ncc[i].computeNcc(windowSize, tmp_flt1[i], (sumvar ? tmp_flt2[i]:0));
	}
	merge(tmp_flt1, dst);
	if (sumvar)
		merge(tmp_flt2, sumvar);
	for (int i=0; i<3; i++) {
		cvReleaseImage(tmp_flt1+i);
		cvReleaseImage(tmp_flt2+i);
	}
}

void FNccMC::merge(IplImage **src, IplImage *dst)
{
	if (dst->nChannels>1) {
		cvMerge(src[0], src[1], src[2], 0, dst);
	} else {
		for (int i=1; i<dst->nChannels; i++)
			cvAdd(src[i], src[0], src[0]);
		cvConvertScale(src[0], dst, 1.0/dst->nChannels);
	}
}

/**** Weighted NCC ****/

FWNcc::FWNcc() {
	integral=0;
}

FWNcc::~FWNcc() {
	if (integral) cvReleaseImage(&integral);
}

void FWNcc::prepare(const IplImage *a, const IplImage *b, const IplImage *w) {

	assert(a->width == b->width);
	assert(a->height == b->height);
	assert(a->nChannels == 1);
	assert(b->nChannels == 1);
	assert(a->depth == IPL_DEPTH_8U);
	assert(b->depth == IPL_DEPTH_8U);

	if (!integral) integral = cvCreateImage(cvSize(NSUMS*(a->width+1),a->height+1), IPL_DEPTH_64F, 1);

	memset(integral->imageData, 0, sizeof(float)*integral->width);

	for (int y=0; y<a->height;y++) {

		unsigned char *la = &CV_IMAGE_ELEM(a, unsigned char, y, 0);
		unsigned char *lb = &CV_IMAGE_ELEM(b, unsigned char, y, 0);
		float *lw = 0;
		if (w) lw = &CV_IMAGE_ELEM(w, float, y, 0);
		integral_type *sum = &CV_IMAGE_ELEM(integral, integral_type, y+1, NSUMS);
		integral_type *upsum = &CV_IMAGE_ELEM(integral, integral_type, y, NSUMS);

		for (int x=-NSUMS; x<0; x++) sum[x]=0;

			for (int x=0; x<a->width; x++) {
				float va = la[x];
				float vb = lb[x];
				float vw = (w?lw[x]:1);
				int x8 = x*NSUMS;

				sum[x8+SUM_A] = sum[x8-NSUMS+SUM_A] + va;
				sum[x8+SUM_B] = sum[x8-NSUMS+SUM_B] + vb;
				sum[x8+SUM_W] = sum[x8-NSUMS+SUM_W] + vw;
				sum[x8+SUM_WA] = sum[x8-NSUMS+SUM_WA] + va*vw;
				sum[x8+SUM_WB] = sum[x8-NSUMS+SUM_WB] + vb*vw;
				sum[x8+SUM_WAB] = sum[x8-NSUMS+SUM_WAB] + va*vb*vw;
				sum[x8+SUM_WA2] = sum[x8-NSUMS+SUM_WA2] + va*va*vw;
				sum[x8+SUM_WB2] = sum[x8-NSUMS+SUM_WB2] + vb*vb*vw;
				sum[x8+SUM_A2] = sum[x8-NSUMS+SUM_A2] + va*va;
				sum[x8+SUM_B2] = sum[x8-NSUMS+SUM_B2] + vb*vb;
				sum[x8+SUM_AB] = sum[x8-NSUMS+SUM_AB] + va*vb;

			}

		int n = a->width*NSUMS;
		for (int x=0; x<n; x++) {
			sum[x] +=  upsum[x];
		}
	}
}

void FWNcc::compute(int winSize, IplImage *dst)
{

	int w = winSize/2;
#pragma omp parallel for
	for (int y=0; y<dst->height; y++) {

		int top = MAX(y-w,0);
		int bot = MIN(y+w, dst->height);
		for (int x = 0; x<w; x++) 
			CV_IMAGE_ELEM(dst, float, y, x) = correl( 0,top, x+w,bot,x,y);
		for (int x = w; x<dst->width-w; x++) 
			CV_IMAGE_ELEM(dst, float, y, x) = correl( x-w,top, x+w,bot,x,y);
		for (int x = dst->width-w; x<dst->width; x++) 
			CV_IMAGE_ELEM(dst, float, y, x) = correl( x-w,top, dst->width,bot,x,y);
	}
}

inline void FWNcc::fetchRect(int x1, int y1, int x2, int y2, float s[NSUMS]) 
{
	integral_type *tl = &CV_IMAGE_ELEM(integral, integral_type, y1, x1*NSUMS);
	integral_type *tr = &CV_IMAGE_ELEM(integral, integral_type, y1, x2*NSUMS);
	integral_type *bl = &CV_IMAGE_ELEM(integral, integral_type, y2, x1*NSUMS);
	integral_type *br = &CV_IMAGE_ELEM(integral, integral_type, y2, x2*NSUMS);
	for (int i=0; i<NSUMS; i++)
		s[i] = (float)(tl[i]-tr[i]-bl[i]+br[i]);
}

static inline float InvSqrt(float x)
{
#if SSE2
	return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
#else
	return 1/sqrtf(x);
#endif
}


void FWNcc::correl(float area, float s[NSUMS], float *cw, float *cx) 
{
	float norm = 1.0f/area;
	float avgA = norm*s[SUM_A];
	float avgB = norm*s[SUM_B];

	if (cw) {
		float num = s[SUM_WAB] - avgB*s[SUM_WA] - avgA*s[SUM_WB] + avgA*avgB*s[SUM_W];
		float twa = (s[SUM_WA2]-2*avgA*s[SUM_WA]+avgA*avgA*s[SUM_W]);
		float twb = (s[SUM_WB2]-2*avgB*s[SUM_WB]+avgB*avgB*s[SUM_W]);

		*cw = num*InvSqrt(twa*twb);
	}

	if (cx) {
		float xab = s[SUM_AB] - s[SUM_WAB];
		float xa = s[SUM_A] - s[SUM_WA];
		float xb = s[SUM_B] - s[SUM_WB];
		float x = area - s[SUM_W];
		float xa2 = s[SUM_A2]-s[SUM_WA2];
		float xb2 = s[SUM_B2]-s[SUM_WB2];

		float txa = (float)(xa2-2*avgA*xa+avgA*avgA*x);
		float txb = (float)(xb2-2*avgB*xb+avgB*avgB*x);
		*cx = (xab - avgB*xa - avgA*xb +avgA*avgB*x)*InvSqrt(txa*txb);
	}
}

float FWNcc::correl(int x1, int y1, int x2, int y2, int centerx, int centery)
{
	float s[NSUMS];
	float cx[NSUMS];
	float se[NSUMS];
	float area = (y2-y1)*(x2-x1);
	fetchRect(x1,y1,x2,y2,s);
	fetchRect(centerx,centery,centerx+1,centery+1,cx);
	for (int i=0;i<NSUMS;i++)
		se[i]=s[i]-cx[i];

	float sw = se[SUM_W];
	float sx = area - se[SUM_W];
	float result;
	if (sw > area/3 && sx > area/3) {
		float cw,cx,cew,cex;
		correl(area, s,&cw,&cx);
		correl(area, se,&cew,&cex);

		float dw = sw*fabs(cw-cew);
		float dx = sx*fabs(cx-cex);
		if (dw<dx)
			result = cw;
		else
			result = cx;
	} else if (sw>sx) {
		correl(area, s, &result, 0);
	} else {
		correl(area, s, 0, &result);
	}
	return MAX(result,0);
}

