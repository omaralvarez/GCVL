# GCVL [![Build Status](https://travis-ci.org/omaralvarez/GCVL.svg?branch=master)](https://travis-ci.org/omaralvarez/GCVL)
###GPGPU Computer Vision Library
Multiplatform (Windows, Unix and MacOS) C++/OpenCL/CUDA library with several Computer Vision Algorithms.
###Building:
```bash
cmake -G "Unix Makefiles" .
make
```
###Example:
```cpp
#include <gcvl/oclcore.h>
#include <gcvl/oclblockmatching.h>

int main(int argc, char *argv[]) {

  gcvl::opencl::Core core;
  gcvl::opencl::BlockMatching bm(&core, image_left.cols, image_left.rows, image_left.data, image_right.data, output);
  bm.compute();
  
  return 0;
  
}
```
