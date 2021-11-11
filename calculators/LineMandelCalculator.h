/**
 * @file LineMandelCalculator.h
 * @author Ondřej Krejčí <xkrejc69@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 11. 11. 2021
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    // add all internal parameters
    int* data;
    float* lineR;
    float* lineI;
    float* defaultRowR;
    float* defaultColumnI;
};