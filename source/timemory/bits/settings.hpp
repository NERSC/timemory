// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#pragma once

#include "timemory/backends/mpi.hpp"     // only depends on below
#include "timemory/utility/macros.hpp"   // macro definitions w/ no internal deps
#include "timemory/utility/utility.hpp"  // generic functions w/ no internal deps

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

#if defined(TIMEMORY_EXTERN_INIT)

#    define TIMEMORY_STATIC_ACCESSOR(TYPE, FUNC, INIT) extern TYPE& FUNC();

#    define TIMEMORY_ENV_STATIC_ACCESSOR(TYPE, FUNC, ENV_VAR, INIT) extern TYPE& FUNC();

#else

#    define TIMEMORY_STATIC_ACCESSOR(TYPE, FUNC, INIT)                                   \
        inline TYPE& FUNC()                                                              \
        {                                                                                \
            static TYPE instance = INIT;                                                 \
            return instance;                                                             \
        }

#    define TIMEMORY_ENV_STATIC_ACCESSOR(TYPE, FUNC, ENV_VAR, INIT)                      \
        inline TYPE& FUNC()                                                              \
        {                                                                                \
            static TYPE instance = get_env<TYPE>(ENV_VAR, INIT);                         \
            return instance;                                                             \
        }

#endif

namespace tim
{
namespace settings
{
//--------------------------------------------------------------------------------------//

using string_t = std::string;

//======================================================================================//
//
//                  GENERAL SETTINGS THAT APPLY TO MULTIPLE COMPONENTS
//
//======================================================================================//

// logical settings
TIMEMORY_STATIC_ACCESSOR(bool, suppress_parsing, false)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, enabled, "TIMEMORY_ENABLED", TIMEMORY_DEFAULT_ENABLED)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, auto_output, "TIMEMORY_AUTO_OUTPUT", true)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, cout_output, "TIMEMORY_COUT_OUTPUT", true)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, file_output, "TIMEMORY_FILE_OUTPUT", true)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, text_output, "TIMEMORY_TEXT_OUTPUT", true)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, json_output, "TIMEMORY_JSON_OUTPUT", false)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, dart_output, "TIMEMORY_DART_OUTPUT", false)

// general settings
TIMEMORY_ENV_STATIC_ACCESSOR(int, verbose, "TIMEMORY_VERBOSE", 0)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, debug, "TIMEMORY_DEBUG", false)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, banner, "TIMEMORY_BANNER", true)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, flat_profile, "TIMEMORY_FLAT_PROFILE", false)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, collapse_threads, "TIMEMORY_COLLAPSE_THREADS", true)
TIMEMORY_ENV_STATIC_ACCESSOR(uint16_t, max_depth, "TIMEMORY_MAX_DEPTH",
                             std::numeric_limits<uint16_t>::max())

// general formatting
TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, precision, "TIMEMORY_PRECISION", -1)
TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, width, "TIMEMORY_WIDTH", -1)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, scientific, "TIMEMORY_SCIENTIFIC", false)

// timing formatting
TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, timing_precision, "TIMEMORY_TIMING_PRECISION", -1)
TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, timing_width, "TIMEMORY_TIMING_WIDTH", -1)
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, timing_units, "TIMEMORY_TIMING_UNITS", "")
TIMEMORY_ENV_STATIC_ACCESSOR(bool, timing_scientific, "TIMEMORY_TIMING_SCIENTIFIC", false)

// memory formatting
TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, memory_precision, "TIMEMORY_MEMORY_PRECISION", -1)
TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, memory_width, "TIMEMORY_MEMORY_WIDTH", -1)
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, memory_units, "TIMEMORY_MEMORY_UNITS", "")
TIMEMORY_ENV_STATIC_ACCESSOR(bool, memory_scientific, "TIMEMORY_MEMORY_SCIENTIFIC", false)

// output control
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, output_path, "TIMEMORY_OUTPUT_PATH",
                             "timemory-output/")  // folder
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, output_prefix, "TIMEMORY_OUTPUT_PREFIX",
                             "")  // file prefix

// dart control
/// only echo this measurement type
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, dart_type, "TIMEMORY_DART_TYPE", "")
/// only echo this many measurement
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, dart_count, "TIMEMORY_DART_COUNT", 0)

//======================================================================================//
//
//                          COMPONENTS SPECIFIC SETTINGS
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
//      PAPI
//--------------------------------------------------------------------------------------//

/// allow multiplexing
TIMEMORY_ENV_STATIC_ACCESSOR(bool, papi_multiplexing, "TIMEMORY_PAPI_MULTIPLEXING", true)

/// errors with PAPI will throw
TIMEMORY_ENV_STATIC_ACCESSOR(bool, papi_fail_on_error, "TIMEMORY_PAPI_FAIL_ON_ERROR",
                             false)

/// PAPI hardware counters
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, papi_events, "TIMEMORY_PAPI_EVENTS", "")

//--------------------------------------------------------------------------------------//
//      CUDA / CUPTI
//--------------------------------------------------------------------------------------//

/// batch size for create cudaEvent_t in cuda_event components
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, cuda_event_batch_size,
                             "TIMEMORY_CUDA_EVENT_BATCH_SIZE", 5)

/// Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)
TIMEMORY_ENV_STATIC_ACCESSOR(bool, nvtx_marker_device_sync,
                             "TIMEMORY_NVTX_MARKER_DEVICE_SYNC", true)

/// default group of kinds tracked via CUpti Activity API
TIMEMORY_ENV_STATIC_ACCESSOR(int32_t, cupti_activity_level,
                             "TIMEMORY_CUPTI_ACTIVITY_LEVEL", 1)

/// specific activity kinds
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cupti_activity_kinds,
                             "TIMEMORY_CUPTI_ACTIVITY_KINDS", "")

/// CUPTI events
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cupti_events, "TIMEMORY_CUPTI_EVENTS", "")

/// CUPTI metrics
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cupti_metrics, "TIMEMORY_CUPTI_METRICS", "")

/// Device to use CUPTI on
TIMEMORY_ENV_STATIC_ACCESSOR(int, cupti_device, "TIMEMORY_CUPTI_DEVICE", 0)

//--------------------------------------------------------------------------------------//
//      ROOFLINE
//--------------------------------------------------------------------------------------//

/// roofline mode for roofline components
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, roofline_mode, "TIMEMORY_ROOFLINE_MODE", "op")

/// set the roofline mode when running ERT on CPU
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cpu_roofline_mode, "TIMEMORY_ROOFLINE_MODE_CPU",
                             roofline_mode())

/// set the roofline mode when running ERT on GPU
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, gpu_roofline_mode, "TIMEMORY_ROOFLINE_MODE_GPU",
                             roofline_mode())

/// custom hw counters to add to the cpu roofline
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cpu_roofline_events,
                             "TIMEMORY_ROOFLINE_EVENTS_CPU", "")

/// custom hw counters to add to the gpu roofline
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, gpu_roofline_events,
                             "TIMEMORY_ROOFLINE_EVENTS_GPU", "")

/// roofline labels/descriptions/output-files encode the list of data types
TIMEMORY_ENV_STATIC_ACCESSOR(bool, roofline_type_labels, "TIMEMORY_ROOFLINE_TYPE_LABELS",
                             false)

/// set the roofline mode when running ERT on CPU
TIMEMORY_ENV_STATIC_ACCESSOR(bool, roofline_type_labels_cpu,
                             "TIMEMORY_ROOFLINE_TYPE_LABELS_CPU", roofline_type_labels())

/// set the roofline mode when running ERT on GPU
TIMEMORY_ENV_STATIC_ACCESSOR(bool, roofline_type_labels_gpu,
                             "TIMEMORY_ROOFLINE_TYPE_LABELS_GPU", roofline_type_labels())

//--------------------------------------------------------------------------------------//
//      ERT
//--------------------------------------------------------------------------------------//

/// set the number of threads when running ERT (0 == default-specific)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_num_threads, "TIMEMORY_ERT_NUM_THREADS", 0)

/// set the number of threads when running ERT on CPU
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_num_threads_cpu,
                             "TIMEMORY_ERT_NUM_THREADS_CPU",
                             std::thread::hardware_concurrency())

/// set the number of threads when running ERT on GPU
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_num_threads_gpu,
                             "TIMEMORY_ERT_NUM_THREADS_GPU", 1)

TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_num_streams, "TIMEMORY_ERT_NUM_STREAMS", 1)

/// set the grid size (number of blocks) for ERT on GPU (0 == auto-compute)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_grid_size, "TIMEMORY_ERT_GRID_SIZE", 0)

/// set the block size (number of threads per block) for ERT on GPU
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_block_size, "TIMEMORY_ERT_BLOCK_SIZE", 1024)

/// set the alignment (in bits) when running ERT on CPU (0 == 8 * sizeof(T))
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_alignment, "TIMEMORY_ERT_ALIGNMENT", 0)

/// set the minimum working size when running ERT (0 == default specific)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_min_working_size,
                             "TIMEMORY_ERT_MIN_WORKING_SIZE", 0)

/// set the minimum working size when running ERT on CPU
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_min_working_size_cpu,
                             "TIMEMORY_ERT_MIN_WORKING_SIZE_CPU", 64)

/// set the minimum working size when running ERT on CPU (default is 10 MB)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_min_working_size_gpu,
                             "TIMEMORY_ERT_MIN_WORKING_SIZE_GPU", 10 * 1000 * 1000)

/// set the max data size when running ERT on CPU (0 == device-specific)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_max_data_size, "TIMEMORY_ERT_MAX_DATA_SIZE", 0)

/// set the max data size when running ERT on CPU (0 == 2 * max-cache-size)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_max_data_size_cpu,
                             "TIMEMORY_ERT_MAX_DATA_SIZE_CPU", 0)

/// set the max data size when running ERT on GPU (default is 500 MB)
TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_max_data_size_gpu,
                             "TIMEMORY_ERT_MAX_DATA_SIZE_GPU", 500 * 1000 * 1000)

//--------------------------------------------------------------------------------------//
//      Signals (more specific signals checked in timemory/details/settings.hpp
//--------------------------------------------------------------------------------------//

/// allow signal handling to be activated
TIMEMORY_ENV_STATIC_ACCESSOR(bool, allow_signal_handler, "TIMEMORY_ALLOW_SIGNAL_HANDLER",
                             true)

/// enable signals in timemory_init
TIMEMORY_ENV_STATIC_ACCESSOR(bool, enable_signal_handler,
                             "TIMEMORY_ENABLE_SIGNAL_HANDLER", false)

/// enable all signals
TIMEMORY_ENV_STATIC_ACCESSOR(bool, enable_all_signals, "TIMEMORY_ENABLE_ALL_SIGNALS",
                             false)

/// disable all signals
TIMEMORY_ENV_STATIC_ACCESSOR(bool, disable_all_signals, "TIMEMORY_DISABLE_ALL_SIGNALS",
                             false)

//--------------------------------------------------------------------------------------//
//     Number of nodes
//--------------------------------------------------------------------------------------//

TIMEMORY_ENV_STATIC_ACCESSOR(int32_t, node_count, "TIMEMORY_NODE_COUNT", 0)

//--------------------------------------------------------------------------------------//
//     For auto_* types
//--------------------------------------------------------------------------------------//

/// default setting for auto_{list,tuple,hybrid} "report_at_exit" member variable
TIMEMORY_ENV_STATIC_ACCESSOR(bool, destructor_report, "TIMEMORY_DESTRUCTOR_REPORT", false)

//--------------------------------------------------------------------------------------//
//     For plotting
//--------------------------------------------------------------------------------------//

/// default setting for python invocation when plotting from C++ code
TIMEMORY_ENV_STATIC_ACCESSOR(string_t, python_exe, "TIMEMORY_PYTHON_EXE", "python")

//--------------------------------------------------------------------------------------//

inline string_t
tolower(string_t str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}

//--------------------------------------------------------------------------------------//

inline string_t
toupper(string_t str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}

//--------------------------------------------------------------------------------------//

inline string_t
get_output_prefix()
{
    auto dir = output_path();
    auto ret = makedir(dir);
    return (ret == 0) ? path_t(dir + string_t("/") + output_prefix())
                      : path_t(string_t("./") + output_prefix());
}

//--------------------------------------------------------------------------------------//

inline string_t
compose_output_filename(const string_t& _tag, string_t _ext, bool _mpi_init = false,
                        const int32_t* _mpi_rank = nullptr)
{
    int32_t _rank = 0;
    if(_mpi_rank)
        _rank = *_mpi_rank;
    else
    {
        // fallback if not specified
        if(mpi::is_initialized())
        {
            _mpi_init = true;
            _rank     = mpi::rank();
        }
    }

    auto _prefix = get_output_prefix();
    auto _rank_suffix =
        (!_mpi_init) ? string_t("") : (string_t("_") + std::to_string(_rank));
    if(_ext.find('.') != 0)
        _ext = string_t(".") + _ext;
    auto plast = _prefix.length() - 1;
    if(_prefix.length() > 0 && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "_";
    auto fpath = path_t(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != string_t::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}

//--------------------------------------------------------------------------------------//

}  // namespace settings
}  // namespace tim

#if !defined(TIMEMORY_ERROR_FUNCTION_MACRO)
#    if defined(__PRETTY_FUNCTION__)
#        define TIMEMORY_ERROR_FUNCTION_MACRO __PRETTY_FUNCTION__
#    else
#        define TIMEMORY_ERROR_FUNCTION_MACRO __FUNCTION__
#    endif
#endif
