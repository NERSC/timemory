//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file extern/init.hpp
 * \headerfile extern/init.hpp "timemory/extern/init.hpp"
 * Provides extern initialization
 *
 */

#pragma once

#if defined(TIMEMORY_EXTERN_INIT) && !defined(TIMEMORY_BUILD_EXTERN_INIT)
#    include "timemory/components/types.hpp"
#    include "timemory/utility/storage.hpp"

namespace tim
{
//======================================================================================//
// declares:
//      extern template get_storage_singleton<TYPE>();
//
#    if defined(TIMEMORY_USE_CALIPER)
TIMEMORY_DECLARE_EXTERN_INIT(caliper)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(cpu_clock)
#    if defined(TIMEMORY_USE_PAPI)
TIMEMORY_DECLARE_EXTERN_INIT(cpu_roofline_dp_flops)
TIMEMORY_DECLARE_EXTERN_INIT(cpu_roofline_flops)
TIMEMORY_DECLARE_EXTERN_INIT(cpu_roofline_sp_flops)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(cpu_util)
#    if defined(TIMEMORY_USE_CUDA)
TIMEMORY_DECLARE_EXTERN_INIT(cuda_event)
TIMEMORY_DECLARE_EXTERN_INIT(cuda_profiler)
#    endif
#    if defined(TIMEMORY_USE_CUPTI)
TIMEMORY_DECLARE_EXTERN_INIT(cupti_activity)
TIMEMORY_DECLARE_EXTERN_INIT(cupti_counters)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(data_rss)
TIMEMORY_DECLARE_EXTERN_INIT(gperf_cpu_profiler)
TIMEMORY_DECLARE_EXTERN_INIT(gperf_heap_profiler)
#    if defined(TIMEMORY_USE_CUPTI)
// TIMEMORY_DECLARE_EXTERN_INIT(gpu_roofline_flops)
TIMEMORY_DECLARE_EXTERN_INIT(gpu_roofline_dp_flops)
// TIMEMORY_DECLARE_EXTERN_INIT(gpu_roofline_hp_flops)
TIMEMORY_DECLARE_EXTERN_INIT(gpu_roofline_sp_flops)
#    endif
#    if defined(TIMEMORY_USE_LIKWID)
TIMEMORY_DECLARE_EXTERN_INIT(likwid_perfmon)
TIMEMORY_DECLARE_EXTERN_INIT(likwid_nvmon)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(monotonic_clock)
TIMEMORY_DECLARE_EXTERN_INIT(monotonic_raw_clock)
TIMEMORY_DECLARE_EXTERN_INIT(num_io_in)
TIMEMORY_DECLARE_EXTERN_INIT(num_io_out)
TIMEMORY_DECLARE_EXTERN_INIT(num_major_page_faults)
TIMEMORY_DECLARE_EXTERN_INIT(num_minor_page_faults)
TIMEMORY_DECLARE_EXTERN_INIT(num_msg_recv)
TIMEMORY_DECLARE_EXTERN_INIT(num_msg_sent)
TIMEMORY_DECLARE_EXTERN_INIT(num_signals)
TIMEMORY_DECLARE_EXTERN_INIT(num_swap)
#    if defined(TIMEMORY_USE_NVTX)
TIMEMORY_DECLARE_EXTERN_INIT(nvtx_marker)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(page_rss)
#    if defined(TIMEMORY_USE_PAPI)
TIMEMORY_DECLARE_EXTERN_INIT(papi_array_t)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(peak_rss)
TIMEMORY_DECLARE_EXTERN_INIT(priority_context_switch)
TIMEMORY_DECLARE_EXTERN_INIT(process_cpu_clock)
TIMEMORY_DECLARE_EXTERN_INIT(process_cpu_util)
TIMEMORY_DECLARE_EXTERN_INIT(read_bytes)
TIMEMORY_DECLARE_EXTERN_INIT(wall_clock)
TIMEMORY_DECLARE_EXTERN_INIT(stack_rss)
TIMEMORY_DECLARE_EXTERN_INIT(system_clock)
#    if defined(TIMEMORY_USE_TAU)
TIMEMORY_DECLARE_EXTERN_INIT(tau_marker)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(thread_cpu_clock)
TIMEMORY_DECLARE_EXTERN_INIT(thread_cpu_util)
TIMEMORY_DECLARE_EXTERN_INIT(trip_count)
TIMEMORY_DECLARE_EXTERN_INIT(user_tuple_bundle)
TIMEMORY_DECLARE_EXTERN_INIT(user_list_bundle)
TIMEMORY_DECLARE_EXTERN_INIT(user_clock)
TIMEMORY_DECLARE_EXTERN_INIT(virtual_memory)
#    if defined(TIMEMORY_USE_VTUNE)
TIMEMORY_DECLARE_EXTERN_INIT(vtune_event)
TIMEMORY_DECLARE_EXTERN_INIT(vtune_frame)
#    endif
TIMEMORY_DECLARE_EXTERN_INIT(voluntary_context_switch)
TIMEMORY_DECLARE_EXTERN_INIT(written_bytes)

}  // namespace tim

//--------------------------------------------------------------------------------------//

#endif  // defined(TIMEMORY_EXTERN_INIT) && !defined(TIMEMORY_BUILD_EXTERN_INIT)

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_MPI)

#    include "timemory/backends/mpi.hpp"
#    include "timemory/manager.hpp"

extern "C" int
MPI_Finalize();

extern "C" int
MPI_Init(int* argc, char*** argv);

extern "C" int
MPI_Init_thread(int* argc, char*** argv, int required, int* provided);

#    if !defined(TIMEMORY_EXTERN_INIT)
inline ::tim::manager*
timemory_mpi_manager_master_instance()
{
    using manager_t     = tim::manager;
    static auto& _pinst = tim::get_shared_ptr_pair<manager_t>();
    return _pinst.first.get();
}

extern "C"
{
    int MPI_Init(int* argc, char*** argv)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init!\n", __FUNCTION__, __FILE__,
                   __LINE__);
        }
#        if defined(TIMEMORY_USE_TAU)
        Tau_init(*argc, *argv);
#        endif
        auto        ret      = PMPI_Init(argc, argv);
        static auto _manager = timemory_mpi_manager_master_instance();
        tim::consume_parameters(_manager);
        ::tim::timemory_init(*argc, *argv);
        return ret;
    }

    int MPI_Init_thread(int* argc, char*** argv, int req, int* prov)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init_thread!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
#        if defined(TIMEMORY_USE_TAU)
        Tau_init(*argc, *argv);
#        endif
        auto        ret      = PMPI_Init_thread(argc, argv, req, prov);
        static auto _manager = timemory_mpi_manager_master_instance();
        tim::consume_parameters(_manager);
        ::tim::timemory_init(*argc, *argv);
        return ret;
    }

    int MPI_Finalize()
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Finalize!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
        auto manager = timemory_mpi_manager_master_instance();
        if(manager)
            manager->finalize();
        ::tim::dmp::is_finalized() = true;
        return PMPI_Finalize();
    }
}  // extern "C"

#    endif  // !defined(TIMEMORY_EXTERN_INIT)
#endif      // defined(TIMEMORY_USE_MPI)
