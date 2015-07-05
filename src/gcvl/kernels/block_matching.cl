str(
__kernel void calculateDisparity(const __global uchar * inputLeft,
                                 const __global uchar * inputRight,
                                 __global uchar * output,
                                 const int width,
								 const int height,
                                 const int dim,
                                 const int radius,
                                 const int maxDisp) {

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int offsetx = x - radius;
    const int offsety = y - radius;

    if(offsetx >= 0 && offsetx + dim < width && offsety >= 0 && offsety + dim < height) {

        unsigned int sum = 0;
        unsigned int bestSum = -1;
        unsigned int bestd = 0;

        for(int d = 0; d < maxDisp; ++d) {
            for(int i = offsety; i < dim + offsety; ++i) {
                for(int j = offsetx; j < dim + offsetx; ++j) {
					if(j - d >= 0)
                    sum += abs((int)inputLeft[i * width + j] - (int)inputRight[i * width + j - d]);
					else
					sum += abs((int)inputLeft[i * width + j]);
                }
            }
            if(sum < bestSum) {
                bestSum = sum;
                bestd = d;
            }
            sum = 0;
        }

        //TODO Normalize here, not in another kernel, waste of time (at least in naive approach)
        output[y * width + x] = bestd;

    }

}
);
