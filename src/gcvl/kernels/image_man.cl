str(
__kernel void test(const __global uchar *img, __global uchar *result, const unsigned int width) {

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  result[y * width + x] = min(img[y * width + x] + 200,255);
  result[y * width + x + 1] = min(img[y * width + x + 1] + 200,255);
  result[y * width + x + 2] = min(img[y * width + x + 2] + 200,255);

}
);
