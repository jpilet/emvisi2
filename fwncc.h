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
 * Julien Pilet, oct/nov 2007
 */
#ifndef FWNCC_H
#define FWNCC_H

#include <pmmintrin.h>
#include <cv.h>

typedef double integral_type;

class FNcc {
public:
	FNcc();
	~FNcc();


	// these are single channel images only.
	void setModel(const IplImage *b, const IplImage *mask=0);
	void setImage(IplImage *a);

	void computeNcc(int windowSize, IplImage *dst, IplImage *sumvar=0);
	void diffNcc(int windowSize, const IplImage *da, IplImage *diff);

protected:
	struct CSum {
		union { 
			struct {
				double b;
				double b2;
				double a;
				double a2;
				double ab;
			}; 
			struct {
				__m128d b_b2;
				__m128d a_a2;
				__m128d ab_fill;
			};
		};
		void setaddsub4(const CSum &x, const CSum &y, const CSum &z, const CSum &w) {
			b_b2 = _mm_add_pd(_mm_sub_pd(x.b_b2, y.b_b2), _mm_sub_pd(z.b_b2,w.b_b2));
			a_a2 = _mm_add_pd(_mm_sub_pd(x.a_a2, y.a_a2), _mm_sub_pd(z.a_a2,w.a_a2));
			ab = (x.ab - y.ab) + (z.ab - w.ab);

			/*
			b  = x.b  - y.b + z.b - w.b;
			b2 = x.b2 - y.b2 + z.b2 - w.b2;
			a  = x.a  - y.a + z.a - w.a;
			a2 = x.a2 - y.a2 + z.a2 - w.a2;
			ab = x.ab - y.ab + z.ab - w.ab;
			*/
		}

	};

	struct CSumf {
		float b;
		float b2;
		float a;
		float a2;
		float ab;
		CSumf & operator = (const CSum &x) {
			b = (float) x.b;
			b2 = (float) x.b2;
			a = (float) x.a;
			a2 = (float) x.a2;
			ab = (float) x.ab;
			return *this;
		}

	};
	struct DSum {
		double da;
		double ada;
		double bda;
		void setaddsub4(const DSum &x, const DSum &y, const DSum &z, const DSum &w) {
			da = (x.da - y.da) + (z.da - w.da);
			ada = (x.ada - y.ada) + (z.ada - w.ada);
			bda = (x.bda - y.bda) + (z.bda - w.bda);
		}
	};

	void fetchRect(int x1, int y1, int x2, int y2, const float *CSum);
	float correl(int x1, int y1, int x2, int y2, int cx, int cy);
	static void correl(float area, const float *CSum, float *cw, float *cx);

	int width, height;
	CSum *integral;
	CSumf *ncc;
	DSum *dint;
	IplImage *a, *b;
	IplImage *mask;
	IplImage *mask_integral;

	void computeNcc_mask(int windowSize, IplImage *dst, IplImage *sumvar=0);
	void computeNcc_nomask(int windowSize, IplImage *dst, IplImage *sumvar=0);
};


class FNccMC {
public:
	FNccMC();
	~FNccMC();

	// these are multi-channels images.
	void setModel(const IplImage *b, const IplImage *mask=0);
	void setImage(const IplImage *a);

	void computeNcc(int windowSize, IplImage *dst, IplImage *sumvar=0);
	void diffNcc(int windowSize, const IplImage *da, IplImage *diff);

	static const int default_win_size=11;

	void computeNcc(IplImage *dst, IplImage *sumvar=0) {
		computeNcc(default_win_size, dst, sumvar);
	}
	void diffNcc(const IplImage *da, IplImage *diff) {
		diffNcc(default_win_size, da, diff);
	}

private:
	IplImage *a[3];
	IplImage *b[3];
	IplImage *tmp_dst[3];
	IplImage *tmp_sumvar[3];
	IplImage *tmp_flt1[3];
	IplImage *tmp_flt2[3];

	FNcc ncc[3];

	void merge(IplImage **src, IplImage *dst);
};


class FWNcc {
public:
	FWNcc();
	~FWNcc();


	void prepare(const IplImage *a, const IplImage *b, const IplImage *w);

	void compute(int windowSize, IplImage *dst);

protected:

	enum { SUM_A=0, SUM_B, SUM_W, SUM_WA, SUM_WB, SUM_WAB, SUM_WA2, SUM_WB2, SUM_A2, SUM_B2, SUM_AB, NSUMS };
	void fetchRect(int x1, int y1, int x2, int y2, float s[NSUMS]);
	float correl(int x1, int y1, int x2, int y2, int cx, int cy);
	static void correl(float area, float s[NSUMS], float *cw, float *cx);
	IplImage *integral;
};

#endif
