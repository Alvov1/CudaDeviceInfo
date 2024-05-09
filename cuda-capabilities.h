#ifndef CUDA_CAPABILITIES_H
#define CUDA_CAPABILITIES_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace Cuda {
    int showCapabilities() {
        int devicesCount;
        if(cudaSuccess != cudaGetDeviceCount( &devicesCount ))
            return std::printf("Cuda::ShowCapabilities failed: cudaGetDeviceCount function failed.");

        for (unsigned i = 0; i < devicesCount; ++i) {
            if(cudaSuccess != cudaSetDevice(i))
                return std::printf("Cuda::ShowCapabilities failed: cudaSetDevice function failed.");

            cudaDeviceProp properties;
            if(cudaSuccess != cudaGetDeviceProperties(&properties, i))
                return std::printf("Cuda::ShowCapabilities failed: cudaGetDeviceProperties function failed.");

            std::cout << "Device: " << properties.name << ".\n" << "Max threads per block: " << properties.maxThreadsPerBlock <<
                ", max threads dimension: (" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " <<
                properties.maxThreadsDim[2] << "), max grid size: (" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] <<
                ", " << properties.maxGridSize[2] << ").\nTotal global memory: " << properties.totalGlobalMem <<
                " bytes, total const memory: " << properties.totalConstMem << " bytes, memory bus width: " << properties.memoryBusWidth <<
                " bits, l2 cache size: " << properties.l2CacheSize << " bytes.\n\n";
        }
    }
}

#endif //CUDA_CAPABILITIES_H
