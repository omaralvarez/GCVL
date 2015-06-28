# GPGPU Computer Vision Library [![Build Status](https://travis-ci.org/omaralvarez/GCVL.svg?branch=master)](https://travis-ci.org/omaralvarez/GCVL)
Multi-platform (Windows, Unix and Mac OS) C++/OpenCL/CUDA library with several Computer Vision algorithms.
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
