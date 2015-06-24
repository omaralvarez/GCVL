str(
__kernel void test(const __global uchar *img, __global uchar *result, const unsigned int width) {

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  result[y * width + x] = img[y * width + x] + 20;
  result[y * width + x+1] = img[y * width + x+1] + 20;
  result[y * width + x+2] = img[y * width + x+2] + 20;

}
);
