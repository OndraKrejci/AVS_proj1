/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <cstring>

#include "LineMandelCalculator.h"

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int*) _mm_malloc(height * width * sizeof(int), 64);
	lineR = (float*) _mm_malloc(width * sizeof(float), 64);
	lineI = (float*) _mm_malloc(width * sizeof(float), 64);
	defaultLineR = (float*) _mm_malloc(width * sizeof(float), 64);

	#ifdef USE_ZERO
	memset(data, 0, height * width * sizeof(int));
	#else
	std::fill_n(data, height * width, limit);
	#endif
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	_mm_free(lineR);
	_mm_free(lineI);
	_mm_free(defaultLineR);
	data = NULL;
	lineR = NULL;
	lineI = NULL;
	defaultLineR = NULL;
}


int* LineMandelCalculator::calculateMandelbrot(){
	// @TODO implement the calculator & return array of integers
	for(int i = 0; i < width; i++){
		defaultLineR[i] = x_start + i * dx;
	}

	for(int i = 0; i < height; i++){ // radky
		int* const rowData = data + i * width; // zacatek dat pro radek

		// inicializace hodnot pro radek
		const float defaultI = y_start + i * dy;
		std::fill_n(lineI, width, defaultI);
		memcpy(lineR, defaultLineR, width * sizeof(float));

		for(int k = 0; k < limit; k++){ // iterace
			#pragma omp simd
			for(int j = 0; j < width; j++){ // sloupce
				const float r2 = lineR[j] * lineR[j];
				const float i2 = lineI[j] * lineI[j];

				#ifdef USE_ZERO
				rowData[j] += (r2 + i2 < 4.0f) ? 1 : 0;
				#else
				const int val = rowData[j];
				rowData[j] = (r2 + i2 > 4.0f && val == limit) ? k : val;
				#endif

				lineI[j] = 2.0f * lineR[j] * lineI[j] + defaultI;
				lineR[j] = r2 - i2 + defaultLineR[j];
			}
		}
	}
	return data;
}
