# GPU4GST_code
## Environment

- Linux (Ubuntu 22.04+)
- CUDA 11.8+
- CMake 3.9+

## File Structure

- `include/`: source code
- `CPU_code/`: core codes


## Build & Run

you can build and run the test program.
cmake3 .. -D CMAKE_CXX_COMPILER=/opt/rh/devtoolset-11/root/bin/g++ -D CMAKE_C_COMPILER=/opt/rh/devtoolset-11/root/bin/gcc

```shell
mkdir build
cd build
cmake ..
make
./test
```

