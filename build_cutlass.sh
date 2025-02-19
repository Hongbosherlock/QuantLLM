export CUDACXX=/usr/local/cuda/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd cutlass
rm -rf build
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 64

# make cutlass_profiler -j12
