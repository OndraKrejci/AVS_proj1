/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

#define BATCH_SIZE 64

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int*) _mm_malloc(height * width * sizeof(int), 64);
	defaultRowR = (float*) _mm_malloc(width * sizeof(float), 64);
	defaultColumnI = (float*) _mm_malloc(height * sizeof(float), 64);
	batchR = (float*) _mm_malloc(BATCH_SIZE * sizeof(float), 64);
	batchI = (float*) _mm_malloc(BATCH_SIZE * sizeof(float), 64);

	#ifdef USE_ZERO
	memset(data, 0, height * width * sizeof(int));
	#else
	for(int i = 0; i < height * width; i++){
		data[i] = limit;
	}
	#endif
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	_mm_free(defaultRowR);
	_mm_free(defaultColumnI);
	_mm_free(batchR);
	_mm_free(batchI);
	data = NULL;
	defaultColumnI = NULL;
	defaultRowR = NULL;
	batchR = NULL;
	batchI = NULL;
}


int* BatchMandelCalculator::calculateMandelbrot(){
	// @TODO implement the calculator & return array of integers
	for(int i = 0; i < width; i++){
		defaultRowR[i] = x_start + i * dx;
	}
	for(int i = 0; i < height; i++){
		defaultColumnI[i] = y_start + i * dy;
	}

	for(int batch = 0; batch < (width * height / BATCH_SIZE); batch++){ // batches
		for(int i = (batch * BATCH_SIZE); i < (batch * BATCH_SIZE + BATCH_SIZE); i++){
			const int real = i % width;
			const int imag = i / width;
			const int batchIdx = i % BATCH_SIZE;
			batchR[batchIdx] = defaultRowR[real];
			batchI[batchIdx] = defaultColumnI[imag];
		}

		for(int k = 0; k < limit; k++){ // iterace
			#pragma omp simd
			for(int i = (batch * BATCH_SIZE); i < (batch * BATCH_SIZE + BATCH_SIZE); i++){ // batch
				const int real = i % width;
				const int imag = i / width;
				const int batchIdx = i % BATCH_SIZE;
				const float r2 = batchR[batchIdx] * batchR[batchIdx];
				const float i2 = batchI[batchIdx] * batchI[batchIdx];

				#ifdef USE_ZERO
				if (r2 + i2 < 4.0f)
				{
					data[i]++;
				}
				#else
				if (r2 + i2 > 4.0f && data[i] == limit)
				{
					data[i] = k;
				}
				#endif

				batchI[batchIdx] = 2.0f * batchR[batchIdx] * batchI[batchIdx] + defaultColumnI[imag];
				batchR[batchIdx] = r2 - i2 + defaultRowR[real];
			}
		}
	}
	return data;
}