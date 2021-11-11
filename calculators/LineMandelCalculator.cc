/**
 * @file LineMandelCalculator.cc
 * @author Ondřej Krejčí <xkrejc69@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 11. 11. 2021
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <cstring>

#include "LineMandelCalculator.h"

// allocate & prefill memory
LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int*) _mm_malloc(height * width * sizeof(int), 64);
	lineR = (float*) _mm_malloc(width * sizeof(float), 64);
	lineI = (float*) _mm_malloc(width * sizeof(float), 64);
	defaultRowR = (float*) _mm_malloc(width * sizeof(float), 64);
	defaultColumnI = (float*) _mm_malloc(height * sizeof(float), 64);

	memset(data, 0, height * width * sizeof(int));
}

// cleanup the memory
LineMandelCalculator::~LineMandelCalculator(){
	_mm_free(data);
	_mm_free(lineR);
	_mm_free(lineI);
	_mm_free(defaultRowR);
	_mm_free(defaultColumnI);
	data = NULL;
	lineR = NULL;
	lineI = NULL;
	defaultRowR = NULL;
	defaultColumnI = NULL;
}

// implement the calculator & return array of integers
int* LineMandelCalculator::calculateMandelbrot(){
	for(int i = 0; i < width; i++){
		defaultRowR[i] = x_start + i * dx;
	}
	for(int i = 0; i < height; i++){
		defaultColumnI[i] = y_start + i * dy;
	}

	for(int i = 0; i < height; i++){ // radky
		int* const rowData = data + i * width; // zacatek dat pro radek

		// inicializace hodnot pro radek
		const float defaultI = defaultColumnI[i];
		std::fill_n(lineI, width, defaultI);
		memcpy(lineR, defaultRowR, width * sizeof(float));

		for(int k = 0; k < limit; k++){ // iterace
			unsigned finished = 0;

			#pragma omp simd reduction(+:finished) simdlen(32)
			for(int j = 0; j < width; j++){ // sloupce
				const float r2 = lineR[j] * lineR[j];
				const float i2 = lineI[j] * lineI[j];

				rowData[j] += (r2 + i2 < 4.0f) ? 1 : (finished++, 0);

				lineI[j] = 2.0f * lineR[j] * lineI[j] + defaultI;
				lineR[j] = r2 - i2 + defaultRowR[j];
			}

			if(finished == width) break;
		}
	}
	return data;
}
