/**
 * @file BatchMandelCalculator.cc
 * @author Ondřej Krejčí <xkrejc69@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 12. 11. 2021
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include <cstring>

#include "BatchMandelCalculator.h"

// allocate & prefill memory
BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator"),
	SIZE(width * height)
{
	data = (int*) _mm_malloc(SIZE * sizeof(int), 64);
	defaultRowR = (float*) _mm_malloc(width * sizeof(float), 64);
	defaultColumnI = (float*) _mm_malloc(height * sizeof(float), 64);
	batchR = (float*) _mm_malloc(BATCH_SIZE * sizeof(float), 64);
	batchI = (float*) _mm_malloc(BATCH_SIZE * sizeof(float), 64);
	batchDefaultR = (float*) _mm_malloc(BATCH_SIZE * sizeof(float), 64);
	batchDefaultI = (float*) _mm_malloc(BATCH_SIZE * sizeof(float), 64);

	memset(data, 0, SIZE * sizeof(int));

	for(int i = 0; i < width; i++){
		defaultRowR[i] = x_start + i * dx;
	}
	for(int i = 0; i < height; i++){
		defaultColumnI[i] = y_start + i * dy;
	}
}

// cleanup the memory
BatchMandelCalculator::~BatchMandelCalculator(){
	_mm_free(data);
	_mm_free(defaultRowR);
	_mm_free(defaultColumnI);
	_mm_free(batchR);
	_mm_free(batchI);
	_mm_free(batchDefaultR);
	_mm_free(batchDefaultI);
	data = NULL;
	defaultRowR = NULL;
	defaultColumnI = NULL;
	batchR = NULL;
	batchI = NULL;
	batchDefaultR = NULL;
	batchDefaultI = NULL;
}

// implement the calculator & return array of integers
int* BatchMandelCalculator::calculateMandelbrot(){
	const int batchCount = (SIZE / BATCH_SIZE);
	for(int batch = 0; batch < batchCount; batch++){ // batches
		mandelbrotIterations(batch * BATCH_SIZE, BATCH_SIZE);
	}

	// dokroceni
	const int batchStartIdx = batchCount * BATCH_SIZE;
	const int end = SIZE - batchStartIdx;
	if(end){
		mandelbrotIterations(batchStartIdx, end);
	}
	
	return data;
}

inline void BatchMandelCalculator::mandelbrotIterations(int batchStartIdx, int end){
	// inicializace dat pro batch
	#pragma omp simd simdlen(32)
	for(int i = 0; i < end; i++){
		const int idx = batchStartIdx + i;
		const int real = idx % width;
		const int imag = idx / width;
		batchR[i] = defaultRowR[real];
		batchI[i] = defaultColumnI[imag];
		batchDefaultR[i] = defaultRowR[real];
		batchDefaultI[i] = defaultColumnI[imag];
	}

	int* const batchData = data + batchStartIdx; // zacatek dat pro batch

	for(int k = 0; k < limit; k++){ // iterace
		unsigned finished = 0;

		#pragma omp simd reduction(+:finished) simdlen(32)
		for(int i = 0; i < end; i++){ // batch
			const float r2 = batchR[i] * batchR[i];
			const float i2 = batchI[i] * batchI[i];

			batchData[i] += (r2 + i2 <= 4.0f) ? 1 : (finished++, 0);

			batchI[i] = 2.0f * batchR[i] * batchI[i] + batchDefaultI[i];
			batchR[i] = r2 - i2 + batchDefaultR[i];
		}

		if(finished == end) break;
	}
}
