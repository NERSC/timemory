#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#include "cupti_profiler.h"

#define PROFILE_ALL_EVENTS_METRICS 1

template <typename T>
__global__ void
kernel(T begin, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        if(i < n)
            *(begin + i) += 1;
    }
}

template <typename T>
__global__ void
kernel2(T begin, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        if(i < n / 2)
            *(begin + i) += 2;
        else if(i >= n / 2 && i < n)
            *(begin + i) += 3;
    }
}

template <typename T>
void
call_kernel(T& arg, cudaStream_t stream = 0)
{
    kernel<<<2, 64, 0, stream>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

template <typename T>
void
call_kernel2(T& arg, cudaStream_t stream = 0)
{
    kernel2<<<64, 2, 0, stream>>>(thrust::raw_pointer_cast(&arg[0]), arg.size() / 2);
}

int
main()
{
    using namespace std;
    // using namespace thrust;

    CUdevice device;

    CUDA_DRIVER_API_CALL(cuInit(0));
    CUDA_DRIVER_API_CALL(cuDeviceGet(&device, 0));

    //#if PROFILE_ALL_EVENTS_METRICS
    // const auto event_names = cupti_counters::available_events(device);
    // const auto metric_names = cupti_counters::available_metrics(device);
    //#else
    vector<string> event_names{
        "active_warps",
        "active_cycles",
    };
    vector<string> metric_names{
        "inst_per_warp",
        "branch_efficiency",
        "warp_execution_efficiency",
        "flop_count_sp",
    };
    //#endif

    constexpr int                N = 100;
    thrust::device_vector<float> data(N, 0);

    // cupti_counters::profiler profiler(vector<string>{}, metric_names);

    // XXX: Disabling all metrics seems to change the values
    // of some events. Not sure if this is correct behavior.
    // cupti_counters::profiler profiler(event_names, vector<string>{});

    cupti_counters::profiler profiler(event_names, metric_names);
    // Get #passes required to compute all metrics and events
    const int passes = profiler.get_passes();
    printf("Passes: %d\n", passes);

    std::vector<cudaStream_t> streams(2);
    for(auto itr : streams)
        cudaStreamCreate(&itr);

    profiler.start();
    for(int i = 0; i < 10; ++i)
    {
        printf("\n\n[%s]> iteration %i...\n", __FUNCTION__, i);
        call_kernel(data, streams.front());
        call_kernel2(data, streams.back());
    }
    profiler.stop();

    for(auto itr : streams)
    {
        cudaStreamSynchronize(itr);
        cudaStreamDestroy(itr);
    }
    cudaDeviceSynchronize();

    auto names = profiler.get_kernel_names();
    printf("\n\n\nKernel Names:\n\n\n");
    for(auto name : names)
    {
        printf("%s\n", name.c_str());
    }
    printf("\n\n\nEvent Trace:\n\n\n");
    profiler.print_event_values(std::cout);
    printf("\n\n\nMetric Trace\n\n\n");
    profiler.print_metric_values(std::cout);
    std::cout << std::flush;

    thrust::host_vector<float> h_data(data);

    /*printf("\n");
    for(int i = 0; i < 10; ++i) {
      printf("%.2lf ", h_data[i]);
    }*/
    printf("\n");
    return 0;
}
