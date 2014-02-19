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
#ifndef GROWMAT_H
#define GROWMAT_H

#include <opencv2/opencv.hpp>

//! A dynamic size version of CvMat.
class CvGrowMat : public CvMat {
public:

	CvGrowMat(int maxlines, int maxcols, int type);
	~CvGrowMat();

	void resize(int lines, int cols);

	//! Load an ascii array of value (tab or space separated)
	static CvGrowMat *loadMat(const char *fn, int type=CV_32FC1);
	static bool saveMat(const CvMat *m, const char *fn);
private:
	CvMat *mat;
};

#endif
