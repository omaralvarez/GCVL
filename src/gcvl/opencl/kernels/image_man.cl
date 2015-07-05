str(
__kernel void test(const __global uchar *img, __global uchar *result, const unsigned int width) {

  const int x = get_global_id(0); //rows
  const int y = get_global_id(1); //cols

  result[x * width + y] = min(img[x * width + y] + 155,255);

}
);
