// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "timemory/ert.hpp"
#include "timemory/timemory.hpp"

#include <cstdint>
#include <set>

//--------------------------------------------------------------------------------------//
// make the namespace usage a little clearer
//
namespace ert {
using namespace tim::ert;
}
namespace units {
using namespace tim::units;
}
namespace device {
using namespace tim::device;
}

using settings = tim::settings;

//--------------------------------------------------------------------------------------//
// timemory has a backends that will not call MPI_Init, cudaDeviceCount, etc.
// when that library/package is not available
//
namespace dmp {
using namespace tim::dmp;
}
namespace cuda {
using namespace tim::cuda;
}

//--------------------------------------------------------------------------------------//
// some short-hand aliases
//
using counter_type   = tim::component::wall_clock;
using fp16_t         = tim::cuda::fp16_t;
using ert_data_t     = ert::exec_data<counter_type>;
using ert_data_ptr_t = std::shared_ptr<ert_data_t>;
using init_list_t    = std::set<uint64_t>;

//--------------------------------------------------------------------------------------//
//  this will invoke ERT with the specified settings
//
template <typename Tp, typename DeviceT>
void
run_ert(ert_data_ptr_t, int64_t num_threads, int64_t min_size, int64_t max_data,
        int64_t num_streams = 0, int64_t block_size = 0, int64_t num_gpus = 0);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    settings::verbose() = 0;
    tim::timemory_init(argc, argv);
    dmp::initialize(argc, argv);
    tim::enable_signal_detection();

    auto data  = ert_data_ptr_t(new ert_data_t());
    auto nproc = dmp::size();

    // ert::execute<device::cpu, counter_type, float, double>(data);

    auto cpu_min_size = 64;
    auto cpu_max_data = 2 * ert::cache_size::get_max();

    init_list_t cpu_num_threads;

    if(argc > 1) cpu_min_size = atol(argv[1]);
    if(argc > 2) cpu_max_data = tim::from_string<long>(argv[2]);

    auto default_thread_init_list = init_list_t({ 1, 2 });

    if(argc > 3)
    {
        default_thread_init_list.clear();
        for(int i = 3; i < argc; ++i) default_thread_init_list.insert(atoi(argv[i]));
    }

    for(auto itr : default_thread_init_list)
    {
        auto entry = itr / nproc;
        if(entry > 0) cpu_num_threads.insert(entry);
    }

    TIMEMORY_BLANK_AUTO_TIMER("run_ert");

    for(int i = 0; i < tim::get_env<int>("NUM_ITER", 1); ++i)
    {
#if !defined(USE_CUDA)

        // execute the single-precision ERT calculations
        for(auto nthread : cpu_num_threads)
            run_ert<float, device::cpu>(data, nthread, cpu_min_size, cpu_max_data);

        // execute the double-precision ERT calculations
        for(auto nthread : cpu_num_threads)
            run_ert<double, device::cpu>(data, nthread, cpu_min_size, cpu_max_data);

#else

        auto gpu_min_size = 1 * units::megabyte;
        auto gpu_max_data = 500 * units::megabyte;

        // determine how many GPUs to execute on (or on CPU if zero)
        int num_gpus = cuda::device_count();
        if(num_gpus > 0 && argc > 1) num_gpus = std::min<int>(num_gpus, atoi(argv[1]));
        init_list_t gpu_num_streams = { 1 };
        init_list_t gpu_block_sizes = { 32, 128, 256, 512, 1024 };

        if(num_gpus < 1)
        {
            // execute the single-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                run_ert<float, device::cpu>(data, nthread, cpu_min_size, cpu_max_data);

            // execute the double-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                run_ert<double, device::cpu>(data, nthread, cpu_min_size, cpu_max_data);
        } else  // num_gpus >= 1
        {
#    if defined(TIMEMORY_USE_CUDA_HALF)
            // execute the half-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!tim::get_env("ERT_GPU_FP16", false)) continue;
                        run_ert<fp16_t, device::gpu>(data, nthread, gpu_min_size,
                                                     gpu_max_data, nstream, block,
                                                     num_gpus);
                    }
#    endif
            // execute the single-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!tim::get_env("ERT_GPU_FP32", true)) continue;
                        run_ert<float, device::gpu>(data, nthread, gpu_min_size,
                                                    gpu_max_data, nstream, block,
                                                    num_gpus);
                    }

            // execute the double-precision ERT calculations
            for(auto nthread : cpu_num_threads)
                for(auto nstream : gpu_num_streams)
                    for(auto block : gpu_block_sizes)
                    {
                        if(!tim::get_env("ERT_GPU_FP64", true)) continue;
                        run_ert<double, device::gpu>(data, nthread, gpu_min_size,
                                                     gpu_max_data, nstream, block,
                                                     num_gpus);
                    }
        }
#endif
    }

    std::string fname = "ert_results";

    printf("\n");
    ert::serialize(fname, *data);

    tim::timemory_finalize();
    return 0;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename DeviceT>
void
run_ert(ert_data_ptr_t data, int64_t num_threads, int64_t min_size, int64_t max_data,
        int64_t num_streams, int64_t block_size, int64_t num_gpus)
{
    // create a label for this test
    auto dtype = tim::demangle(typeid(Tp).name());
    auto htype = DeviceT::name();
    auto label = TIMEMORY_JOIN("_", __FUNCTION__, dtype, htype, num_threads, "threads",
                               min_size, "min-ws", max_data, "max-size");

    if(std::is_same<DeviceT, device::gpu>::value)
    {
        label = TIMEMORY_JOIN("_", label, num_gpus, "gpus", num_streams, "streams",
                              block_size, "thr-per-blk");
    }

    printf("\n[ert-example]> Executing %s...\n", label.c_str());

    using ert_executor_type = ert::executor<DeviceT, Tp, counter_type>;
    using ert_config_type   = typename ert_executor_type::configuration_type;
    using ert_counter_type  = ert::counter<DeviceT, Tp, counter_type>;

    //
    // simple modifications to override method number of threads, number of streams,
    // block size, minimum working set size, and max data size
    //
    ert_config_type::get_num_threads()      = [=]() { return num_threads; };
    ert_config_type::get_num_streams()      = [=]() { return num_streams; };
    ert_config_type::get_block_size()       = [=]() { return block_size; };
    ert_config_type::get_min_working_size() = [=]() { return min_size; };
    ert_config_type::get_max_data_size()    = [=]() { return max_data; };

    //
    // create a callback function that sets the device based on the thread-id
    //
    auto set_counter_device = [=](uint64_t tid, ert_counter_type&) {
        if(num_gpus > 0) cuda::set_device(tid % num_gpus);
    };

    //
    // create a configuration object -- this handles a lot of the setup
    //
    ert_config_type config;
    //
    // start generic timemory timer
    //
    TIMEMORY_BLANK_CALIPER(config, tim::auto_timer, label);
    TIMEMORY_CALIPER_APPLY(config, report_at_exit, true);
    //
    // "construct" an ert::executor that executes the configuration and inserts
    // into the data object.
    //
    // NOTE: the ert::executor has callbacks that allows one to customize the
    //       the ERT execution
    //
    ert_executor_type(config, data, set_counter_device);
    if(data && (settings::verbose() > 0 || settings::debug()))
        std::cout << "\n" << *(data) << std::endl;
    printf("\n");
}

//======================================================================================//
