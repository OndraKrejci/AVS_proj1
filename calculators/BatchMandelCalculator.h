/**
 * @file BatchMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
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

    static constexpr int BATCH_SIZE = 64;
    const int SIZE;

    void mandelbrotIterations(const int batchStartIdx, const int end = BATCH_SIZE);
};

#endif