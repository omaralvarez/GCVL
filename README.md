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

  std::unique_ptr<unsigned char[]> output;
  
  //argv[1] -> path to left image argv[2] -> path to right image
  gcvl::opencl::Core core;
  gcvl::opencl::BlockMatching bm(&core, argv[1], argv[2], output);
  
  bm.setAggDim(9);
  bm.setMaxDisp(255);
  
  bm.compute();
  
  return 0;
  
}
```
