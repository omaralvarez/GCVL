# GPU Computer Vision Library [![Build Status](https://travis-ci.org/omaralvarez/GCVL.svg?branch=master)](https://travis-ci.org/omaralvarez/GCVL)
Multi-platform (Windows, Unix and Mac OS) C++/OpenCL/CUDA library with several Computer Vision algorithms.
###Building:
```bash
cmake -G "Unix Makefiles" -DBUILD_OPENCL=ON -DBUILD_CUDA=OFF -DBUILD_TESTS=ON .
make
```
###Example:
```cpp
#include <gcvl/blockmatching.h>
#include <gcvl/opencl/oclcore.h>
#include <gcvl/opencl/oclblockmatching.h>

//argv[1] -> path to left   image argv[2] -> path to right image
int main(int argc, char *argv[]) {

  int dim = 5, maxDisp = 16;
  bool norm = true;
  std::unique_ptr<unsigned char[]> output;
  
  gcvl::BlockMatching bmCPU(argv[1], argv[2], output);
  bmCPU.setAggDim(dim);
  bmCPU.setMaxDisp(maxDisp);
  bmCPU.setNormalize(norm);
  bmCPU.compute();
  
  gcvl::opencl::Core core;
  gcvl::opencl::BlockMatching bmOCL(&core, argv[1], argv[2], output);
  bmOCL.setAggDim(dim);
  bmOCL.setMaxDisp(maxDisp);
  bmOCL.setNormalize(norm);
  bmOCL.compute();
  
  return 0;
  
}
```
