#ifndef CUDA_CAPABILITIES_H
#define CUDA_CAPABILITIES_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace Cuda {
    int showCapabilities() {
        int devicesCount;
        if (cudaSuccess != cudaGetDeviceCount(&devicesCount))
            return std::printf("Cuda::ShowCapabilities failed: cudaGetDeviceCount function failed.");

        for (unsigned i = 0; i < devicesCount; ++i) {
            if (cudaSuccess != cudaSetDevice(i))
                return std::printf("Cuda::ShowCapabilities failed: cudaSetDevice function failed.");

            cudaDeviceProp properties;
            if (cudaSuccess != cudaGetDeviceProperties(&properties, i))
                return std::printf("Cuda::ShowCapabilities failed: cudaGetDeviceProperties function failed.");

            return std::print("Device: %s.\n"
                              "Max threads per block: %d, max threads dimension: (%d, %d, %d), max grid size: (%d, %d, %d). Total global\n"
                              "memory: %zu bytes, total const memory: %zu bytes, memory bus width: %d bits, l2 cache size: %d bytes\n\n",
                              properties.name,
                              properties.maxThreadsPerBlock,
                              properties.maxThreadsDim[0],
                              properties.maxThreadsDim[1],
                              properties.maxThreadsDim[2],
                              properties.maxGridSize[0],
                              properties.maxGridSize[1],
                              properties.maxGridSize[2],
                              properties.totalGlobalMem,
                              properties.totalConstMem,
                              properties.memoryBusWidth,
                              properties.l2CacheSize);
        }
    }
}

#endif //CUDA_CAPABILITIES_H
