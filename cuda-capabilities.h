#ifndef CUDA_CAPABILITIES_H
#define CUDA_CAPABILITIES_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace Cuda {
    int showCapabilities() {
        int devicesCount;
        auto code = cudaGetDeviceCount(&devicesCount);
        if (cudaSuccess != code)
            return std::printf("Cuda::ShowCapabilities failed: cudaGetDeviceCount function failed: %s.\n", cudaGetErrorString(code));

        for (unsigned i = 0; i < devicesCount; ++i) {
            code = cudaSetDevice(i);
            if (cudaSuccess != code)
                return std::printf("Cuda::ShowCapabilities failed: cudaSetDevice function failed: %s.\n", cudaGetErrorString(code));

            cudaDeviceProp properties;
            code = cudaGetDeviceProperties(&properties, i);
            if (cudaSuccess != code)
                return std::printf("Cuda::ShowCapabilities failed: cudaGetDeviceProperties function failed: %s.\n", cudaGetErrorString(code));

            std::printf("Device: %s.\n"
                        "Max threads per block: %d, max blocks per multiprocessor %d, multiProcessorCount %d, max threads dimension: (%d, %d, %d),\n"
                        "max grid size: (%d, %d, %d). Total global memory: %zu bytes, total const memory: %zu bytes, memory bus width: %d bits, l2 cache size: %d bytes\n\n",
                        properties.name,
                        properties.maxThreadsPerBlock,
                        properties.maxBlocksPerMultiProcessor,
                        properties.multiProcessorCount,
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

        return devicesCount;
    }
}

#endif //CUDA_CAPABILITIES_H
