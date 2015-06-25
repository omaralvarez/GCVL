str(
__kernel void test(const __global uchar *img, __global uchar *result, const unsigned int width) {

  const int x = get_global_id(0);
  const int y = get_global_id(1);

  result[y * width * 3 + x*3] = min(img[y * width * 3 + x*3] + 20,255);
  result[y * width * 3 + x*3 + 1] = min(img[y * width * 3 + x*3+1] + 20,255);
  result[y * width * 3 + x*3 + 2] = min(img[y * width * 3 + x*3+2] + 20,255);

}
);
