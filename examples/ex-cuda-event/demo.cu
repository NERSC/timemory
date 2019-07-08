#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#include "cupti_profiler.h"

#define PROFILE_ALL_EVENTS_METRICS 1

template <typename T>
__global__ void
kernel(T begin, int size)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < size)
        *(begin + thread_id) += 1;
}

template <typename T>
__global__ void
kernel2(T begin, int size)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < size)
        *(begin + thread_id) += 2;
}

template <typename T>
void
call_kernel(T& arg)
{
    kernel<<<1, 100>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
}

template <typename T>
void
call_kernel2(T& arg)
{
    kernel2<<<1, 50>>>(thrust::raw_pointer_cast(&arg[0]), arg.size());
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
    // const auto event_names = cupti_profiler::available_events(device);
    // const auto metric_names = cupti_profiler::available_metrics(device);
    //#else
    vector<string> event_names{
        "active_warps",
        "active_cycles",
    };
    vector<string> metric_names{
        "inst_per_warp",
        "branch_efficiency",
        "warp_execution_efficiency",
        "warp_nonpred_execution_efficiency",
        "inst_replay_overhead",
    };
    //#endif

    constexpr int                N = 100;
    thrust::device_vector<float> data(N, 0);

    // cupti_profiler::profiler profiler(vector<string>{}, metric_names);

    // XXX: Disabling all metrics seems to change the values
    // of some events. Not sure if this is correct behavior.
    // cupti_profiler::profiler profiler(event_names, vector<string>{});

    cupti_profiler::profiler profiler(event_names, metric_names);
    // Get #passes required to compute all metrics and events
    const int passes = profiler.get_passes();
    printf("Passes: %d\n", passes);

    profiler.start();
    for(int i = 0; i < 50; ++i)
    {
        call_kernel(data);
        cudaDeviceSynchronize();
        call_kernel2(data);
        cudaDeviceSynchronize();
    }
    profiler.stop();

    printf("\n\n\nEvent Trace:\n\n\n");
    profiler.print_event_values(std::cout);
    printf("\n\n\nMetric Trace\n\n\n");
    profiler.print_metric_values(std::cout);
    std::cout << std::flush;
    auto names = profiler.get_kernel_names();
    for(auto name : names)
    {
        printf("%s\n", name.c_str());
    }

    thrust::host_vector<float> h_data(data);

    /*printf("\n");
    for(int i = 0; i < 10; ++i) {
      printf("%.2lf ", h_data[i]);
    }*/
    printf("\n");
    return 0;
}
