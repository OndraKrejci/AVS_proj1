/**
 * @file BatchMandelCalculator.h
 * @author Ondřej Krejčí <xkrejc69@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 11. 11. 2021
 */

#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    // @TODO add all internal parameters
    int *data;
    float* defaultRowR;
    float* defaultColumnI;
    float* batchR;
    float* batchI;
    float* batchDefaultR;
    float* batchDefaultI;

    static constexpr int BATCH_SIZE = 128;
    const int SIZE;

    void mandelbrotIterations(const int batchStartIdx, const int end = BATCH_SIZE);
};

#endif