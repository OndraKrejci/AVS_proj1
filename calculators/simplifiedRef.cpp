

int* data;
int height;
int width;
float x_start;
float y_start;
float dx;
float dy;
int limit;

int* calculateMandelbrot () {
	// implement the calculator & return array of integers
	int *pdata = data;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float x = x_start + j * dx; // current real value
			float y = y_start + i * dy; // current imaginary value

			int value = limit;
			float zReal = x;
			float zImag = y;

			for (int i = 0; i < limit; ++i)
			{
				float r2 = zReal * zReal;
				float i2 = zImag * zImag;

				if (r2 + i2 > 4.0f)
				{
					value = i;
					break;
				}

				zImag = 2.0f * zReal * zImag + y;
				zReal = r2 - i2 + x;
			}

			*(pdata++) = value;
		}
	}
	return data;
}