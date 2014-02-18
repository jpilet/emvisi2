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
/* Julien Pilet 2008
 */
#include "growmat.h"
#include <iostream>
#include <fstream>

CvGrowMat::CvGrowMat(int maxlines, int maxcols, int type)
{
	mat = cvCreateMat(maxlines, maxcols, type);
	cvSetZero(mat);
	cvGetSubRect(mat, this, cvRect(0,0,maxcols,maxlines));
}

CvGrowMat::~CvGrowMat()
{
	cvReleaseMat(&mat);
}

void CvGrowMat::resize(int lines, int cols)
{
	if (lines <= mat->rows && cols <= mat->cols) {
		cvGetSubRect(mat, this, cvRect(0,0,cols,lines));
		//this->rows = lines;
		//this->cols = cols;
	} else {
		int nl = (lines > mat->rows ? lines*2 : mat->rows);
		int nc = (cols > mat->cols ? cols*2 : mat->cols);
		CvMat *nm = cvCreateMat(nl, nc, mat->type);
		cvSetZero(nm);
		if (this->rows && this->cols) {
			CvMat sub;
			cvGetSubRect(nm, &sub, cvRect(0,0,this->cols, this->rows));
			cvCopy(this, &sub);
			cvGetSubRect(nm, this, cvRect(0,0,cols, lines));
		} else {
			cvGetSubRect(nm, this, cvRect(0,0,mat->cols, mat->rows));
			this->rows = lines;
			this->cols = cols;
		}
		cvReleaseMat(&mat);
		mat = nm;
	}
}

CvGrowMat *CvGrowMat::loadMat(const char *fn, int type)
{
	std::ifstream f(fn);

	if (!f.good()) return 0;

	CvGrowMat *m = new CvGrowMat(128,3, type);
	m->resize(1,1);

	int nrow=0;
	do {
		char line[4096];
		f.getline(line, 4095);
		line[4095]=0;
		if (!f.good()) break;

		int ncols=1;
		int len = strlen(line);
		char *last=line;
		for (int i=0;i<len+1;i++) {
			if (line[i]==' ' || line[i]=='\t' || line[i]==0) {
				line[i] = 0;
				float val;
				if (sscanf(last, "%f", &val)==1) {
					if (ncols==1) nrow++;
					m->resize(nrow,(ncols > m->cols ? ncols:m->cols));
					cvSet2D(m, nrow-1, ncols-1, cvScalarAll(val));
					ncols++;
				}
			last = line+i+1;
			}
		}
	} while (f.good());

	return m;
}

bool CvGrowMat::saveMat(const CvMat *m, const char *fn)
{
	std::ofstream f(fn);

	if (!f.good()) return false;

	for (int j=0; j<m->rows; j++) {
		for (int i=0; i<m->cols; i++) {
			double v = cvGet2D(m, j, i).val[0];
			f << v;
			if (i+1 < m->cols) f<<'\t';
		}
		f << std::endl;
	}
	f.close();
	return true;
}

