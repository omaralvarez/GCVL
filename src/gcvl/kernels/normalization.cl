str(
__kernel void normalizeMap(__global uchar *input, __global uchar *output, const unsigned int width, const int maxDisp) {

  const int x = get_global_id(0); //rows
  const int y = get_global_id(1); //cols

  output[y * width + x] = (input[y * width + x]/(float)maxDisp)*255;

}
);
