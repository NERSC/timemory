//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

/** \file timemory/settings.hpp
 * \headerfile timemory/settings.hpp "timemory/settings.hpp"
 * Handles TiMemory settings, parses environment
 *
 */

#pragma once

//======================================================================================//

#include "timemory/units.hpp"
#include "timemory/utility/environment.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

#if defined(_UNIX)
#    include <unistd.h>
extern "C"
{
    extern char** environ;
}
#endif

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

#define TIMEMORY_ENV_STATIC_ACCESSOR_CALLBACK(TYPE, FUNC, ENV_VAR, INIT)                 \
    static auto _generate = []() {                                                       \
        auto _parse = []() { FUNC() = tim::get_env(ENV_VAR, FUNC()); };                  \
        get_parse_callbacks().push_back(_parse);                                         \
        return get_env<TYPE>(ENV_VAR, INIT);                                             \
    };

#if defined(TIMEMORY_EXTERN_INIT)

#    define TIMEMORY_STATIC_ACCESSOR(TYPE, FUNC, INIT) static TYPE& FUNC();
#    define TIMEMORY_ENV_STATIC_ACCESSOR(TYPE, FUNC, ENV_VAR, INIT) static TYPE& FUNC();

#else

#    define TIMEMORY_STATIC_ACCESSOR(TYPE, FUNC, INIT)                                   \
        static TYPE& FUNC()                                                              \
        {                                                                                \
            static TYPE* instance = new TYPE(INIT);                                      \
            return *instance;                                                            \
        }

#    define TIMEMORY_ENV_STATIC_ACCESSOR(TYPE, FUNC, ENV_VAR, INIT)                      \
        static TYPE& FUNC()                                                              \
        {                                                                                \
            TIMEMORY_ENV_STATIC_ACCESSOR_CALLBACK(TYPE, FUNC, ENV_VAR, INIT)             \
            static TYPE instance = _generate();                                          \
            return instance;                                                             \
        }

#endif

namespace tim
{
//--------------------------------------------------------------------------------------//

inline std::string
get_local_datetime(const char* dt_format)
{
    std::stringstream ss;
    std::time_t       t = std::time(nullptr);
    char              mbstr[100];
    if(std::strftime(mbstr, sizeof(mbstr), dt_format, std::localtime(&t)))
    {
        ss << mbstr;
    }
    return ss.str();
}

//--------------------------------------------------------------------------------------//

tim_api struct settings
{
    using string_t    = std::string;
    using strvector_t = std::vector<std::string>;

    //==================================================================================//
    //
    //                  GENERAL SETTINGS THAT APPLY TO MULTIPLE COMPONENTS
    //
    //==================================================================================//

    // logical settings
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, suppress_parsing, "TIMEMORY_SUPPRESS_PARSING",
                                 false)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, enabled, "TIMEMORY_ENABLED",
                                 TIMEMORY_DEFAULT_ENABLED)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, auto_output, "TIMEMORY_AUTO_OUTPUT", true)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, cout_output, "TIMEMORY_COUT_OUTPUT", true)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, file_output, "TIMEMORY_FILE_OUTPUT", true)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, text_output, "TIMEMORY_TEXT_OUTPUT", true)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, json_output, "TIMEMORY_JSON_OUTPUT", false)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, dart_output, "TIMEMORY_DART_OUTPUT", false)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, time_output, "TIMEMORY_TIME_OUTPUT", false)

    // general settings
    TIMEMORY_ENV_STATIC_ACCESSOR(int, verbose, "TIMEMORY_VERBOSE", 0)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, debug, "TIMEMORY_DEBUG", false)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, banner, "TIMEMORY_BANNER", true)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, flat_profile, "TIMEMORY_FLAT_PROFILE", false)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, collapse_threads, "TIMEMORY_COLLAPSE_THREADS",
                                 true)
    TIMEMORY_ENV_STATIC_ACCESSOR(uint16_t, max_depth, "TIMEMORY_MAX_DEPTH",
                                 std::numeric_limits<uint16_t>::max())
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, time_format, "TIMEMORY_TIME_FORMAT",
                                 "%F_%I.%M_%p")

    // general formatting
    TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, precision, "TIMEMORY_PRECISION", -1)
    TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, width, "TIMEMORY_WIDTH", -1)
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, scientific, "TIMEMORY_SCIENTIFIC", false)

    // timing formatting
    TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, timing_precision, "TIMEMORY_TIMING_PRECISION",
                                 -1)
    TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, timing_width, "TIMEMORY_TIMING_WIDTH", -1)
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, timing_units, "TIMEMORY_TIMING_UNITS", "")
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, timing_scientific, "TIMEMORY_TIMING_SCIENTIFIC",
                                 false)

    // memory formatting
    TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, memory_precision, "TIMEMORY_MEMORY_PRECISION",
                                 -1)
    TIMEMORY_ENV_STATIC_ACCESSOR(int16_t, memory_width, "TIMEMORY_MEMORY_WIDTH", -1)
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, memory_units, "TIMEMORY_MEMORY_UNITS", "")
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, memory_scientific, "TIMEMORY_MEMORY_SCIENTIFIC",
                                 false)

    // output control
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, output_path, "TIMEMORY_OUTPUT_PATH",
                                 "timemory-output/")  // folder
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, output_prefix, "TIMEMORY_OUTPUT_PREFIX",
                                 "")  // file prefix

    // dart control
    /// only echo this measurement type
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, dart_type, "TIMEMORY_DART_TYPE", "")
    /// only echo this many dart tags
    TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, dart_count, "TIMEMORY_DART_COUNT", 1)
    /// echo the category, not the identifier
    TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, dart_label, "TIMEMORY_DART_LABEL", true)

    /// enable thread affinity
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, cpu_affinity, "TIMEMORY_CPU_AFFINITY", false)

    //==================================================================================//
    //
    //                          COMPONENTS SPECIFIC SETTINGS
    //
    //==================================================================================//

    //----------------------------------------------------------------------------------//
    //      MPI
    //----------------------------------------------------------------------------------//

    /// timemory will try to call MPI_Init or MPI_Init_thread during certain
    /// timemory_init()
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, mpi_init, "TIMEMORY_MPI_INIT", true)

    /// timemory will try to call MPI_Finalize during timemory_finalize()
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, mpi_finalize, "TIMEMORY_MPI_FINALIZE", true)

    /// use MPI_Init and MPI_Init_thread
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, mpi_thread, "TIMEMORY_MPI_THREAD", true)

    /// use MPI_Init_thread type
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, mpi_thread_type, "TIMEMORY_MPI_THREAD_TYPE",
                                 "")

    /// output MPI data per rank
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, mpi_output_per_rank,
                                 "TIMEMORY_MPI_OUTPUT_PER_RANK", false)

    /// output MPI data per node
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, mpi_output_per_node,
                                 "TIMEMORY_MPI_OUTPUT_PER_NODE", false)

    //----------------------------------------------------------------------------------//
    //      UPC++
    //----------------------------------------------------------------------------------//

    /// timemory will try to call upcxx::init during certain timemory_init()
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, upcxx_init, "TIMEMORY_UPCXX_INIT", true)

    /// timemory will try to call upcxx::finalize during timemory_finalize()
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, upcxx_finalize, "TIMEMORY_UPCXX_FINALIZE", true)

    //----------------------------------------------------------------------------------//
    //      PAPI
    //----------------------------------------------------------------------------------//

    /// allow multiplexing
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, papi_multiplexing, "TIMEMORY_PAPI_MULTIPLEXING",
                                 true)

    /// errors with PAPI will throw
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, papi_fail_on_error, "TIMEMORY_PAPI_FAIL_ON_ERROR",
                                 false)

    /// errors with PAPI will be suppressed
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, papi_quiet, "TIMEMORY_PAPI_QUIET", false)

    /// PAPI hardware counters
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, papi_events, "TIMEMORY_PAPI_EVENTS", "")

    //----------------------------------------------------------------------------------//
    //      CUDA / CUPTI
    //----------------------------------------------------------------------------------//

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

    //----------------------------------------------------------------------------------//
    //      ROOFLINE
    //----------------------------------------------------------------------------------//

    /// roofline mode for roofline components
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, roofline_mode, "TIMEMORY_ROOFLINE_MODE", "op")

    /// set the roofline mode when running ERT on CPU
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cpu_roofline_mode,
                                 "TIMEMORY_ROOFLINE_MODE_CPU", roofline_mode())

    /// set the roofline mode when running ERT on GPU
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, gpu_roofline_mode,
                                 "TIMEMORY_ROOFLINE_MODE_GPU", roofline_mode())

    /// custom hw counters to add to the cpu roofline
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, cpu_roofline_events,
                                 "TIMEMORY_ROOFLINE_EVENTS_CPU", "")

    /// custom hw counters to add to the gpu roofline
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, gpu_roofline_events,
                                 "TIMEMORY_ROOFLINE_EVENTS_GPU", "")

    /// roofline labels/descriptions/output-files encode the list of data types
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, roofline_type_labels,
                                 "TIMEMORY_ROOFLINE_TYPE_LABELS", false)

    /// set the roofline mode when running ERT on CPU
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, roofline_type_labels_cpu,
                                 "TIMEMORY_ROOFLINE_TYPE_LABELS_CPU",
                                 roofline_type_labels())

    /// set the roofline mode when running ERT on GPU
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, roofline_type_labels_gpu,
                                 "TIMEMORY_ROOFLINE_TYPE_LABELS_GPU",
                                 roofline_type_labels())

    /// include the instruction roofline
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, instruction_roofline,
                                 "TIMEMORY_INSTRUCTION_ROOFLINE", false)

    //----------------------------------------------------------------------------------//
    //      ERT
    //----------------------------------------------------------------------------------//

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
    TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_block_size, "TIMEMORY_ERT_BLOCK_SIZE",
                                 1024)

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
    TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_max_data_size,
                                 "TIMEMORY_ERT_MAX_DATA_SIZE", 0)

    /// set the max data size when running ERT on CPU (0 == 2 * max-cache-size)
    TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_max_data_size_cpu,
                                 "TIMEMORY_ERT_MAX_DATA_SIZE_CPU", 0)

    /// set the max data size when running ERT on GPU (default is 500 MB)
    TIMEMORY_ENV_STATIC_ACCESSOR(uint64_t, ert_max_data_size_gpu,
                                 "TIMEMORY_ERT_MAX_DATA_SIZE_GPU", 500 * 1000 * 1000)

    /// set the ops to skip at runtime
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, ert_skip_ops, "TIMEMORY_ERT_SKIP_OPS", "")

    //----------------------------------------------------------------------------------//
    //      Signals (more specific signals checked in timemory/details/settings.hpp
    //----------------------------------------------------------------------------------//

    /// allow signal handling to be activated
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, allow_signal_handler,
                                 "TIMEMORY_ALLOW_SIGNAL_HANDLER", true)

    /// enable signals in timemory_init
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, enable_signal_handler,
                                 "TIMEMORY_ENABLE_SIGNAL_HANDLER", false)

    /// enable all signals
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, enable_all_signals, "TIMEMORY_ENABLE_ALL_SIGNALS",
                                 false)

    /// disable all signals
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, disable_all_signals,
                                 "TIMEMORY_DISABLE_ALL_SIGNALS", false)

    //----------------------------------------------------------------------------------//
    //     Number of nodes
    //----------------------------------------------------------------------------------//

    TIMEMORY_ENV_STATIC_ACCESSOR(int32_t, node_count, "TIMEMORY_NODE_COUNT", 0)

    //----------------------------------------------------------------------------------//
    //     For auto_* types
    //----------------------------------------------------------------------------------//

    /// default setting for auto_{list,tuple,hybrid} "report_at_exit" member variable
    TIMEMORY_ENV_STATIC_ACCESSOR(bool, destructor_report, "TIMEMORY_DESTRUCTOR_REPORT",
                                 false)

    //----------------------------------------------------------------------------------//
    //     For plotting
    //----------------------------------------------------------------------------------//

    /// default setting for python invocation when plotting from C++ code
    TIMEMORY_ENV_STATIC_ACCESSOR(string_t, python_exe, "TIMEMORY_PYTHON_EXE", "python")

    //----------------------------------------------------------------------------------//
    //     Command line
    //----------------------------------------------------------------------------------//

    TIMEMORY_STATIC_ACCESSOR(strvector_t, command_line, strvector_t())

    //----------------------------------------------------------------------------------//
    //     Command line
    //----------------------------------------------------------------------------------//

    TIMEMORY_STATIC_ACCESSOR(strvector_t, environment, get_environment())

    //----------------------------------------------------------------------------------//

    static string_t tolower(string_t str)
    {
        for(auto& itr : str)
            itr = ::tolower(itr);
        return str;
    }

    //----------------------------------------------------------------------------------//

    static string_t toupper(string_t str)
    {
        for(auto& itr : str)
            itr = ::toupper(itr);
        return str;
    }

    //----------------------------------------------------------------------------------//

    static string_t get_output_prefix()
    {
        auto dir = output_path();
        if(time_output())
        {
            if(dir.length() > 0 && dir[dir.length() - 1] != '/')
                dir += "/";
            // ensure that all output files use same local datetime
            static auto _local_datetime = get_local_datetime(time_format().c_str());
            dir += _local_datetime;
        }
        auto ret = makedir(dir);
        return (ret == 0) ? filepath::osrepr(dir + string_t("/") + output_prefix())
                          : filepath::osrepr(string_t("./") + output_prefix());
    }

    //----------------------------------------------------------------------------------//

    static void store_command_line(int argc, char** argv)
    {
        auto& _cmdline = command_line();
        _cmdline.clear();
        for(int i = 0; i < argc; ++i)
            _cmdline.push_back(std::string(argv[i]));
    }

    //----------------------------------------------------------------------------------//

    static string_t compose_output_filename(const string_t& _tag, string_t _ext,
                                            bool          _mpi_init = false,
                                            const int32_t _mpi_rank = -1)
    {
        auto _prefix      = get_output_prefix();
        auto _rank_suffix = (_mpi_init && _mpi_rank >= 0)
                                ? (string_t("_") + std::to_string(_mpi_rank))
                                : string_t("");
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

    //----------------------------------------------------------------------------------//

    /// initialize the storage of the specified types
    static void initialize_storage();

    template <typename _Tuple>
    static void initialize_storage();

    //----------------------------------------------------------------------------------//

    static void parse();

    //----------------------------------------------------------------------------------//

    template <typename Archive>
    static void serialize_settings(Archive& ar)
    {
#define _TRY_CATCH_NVP(ENV_VAR, FUNC)                                                    \
    try                                                                                  \
    {                                                                                    \
        auto& _VAL = FUNC();                                                             \
        ar(cereal::make_nvp(ENV_VAR, _VAL));                                             \
    } catch(...)                                                                         \
    {}

        _TRY_CATCH_NVP("TIMEMORY_SUPPRESS_PARSING", suppress_parsing)
        _TRY_CATCH_NVP("TIMEMORY_ENABLED", enabled)
        _TRY_CATCH_NVP("TIMEMORY_AUTO_OUTPUT", auto_output)
        _TRY_CATCH_NVP("TIMEMORY_COUT_OUTPUT", cout_output)
        _TRY_CATCH_NVP("TIMEMORY_FILE_OUTPUT", file_output)
        _TRY_CATCH_NVP("TIMEMORY_TEXT_OUTPUT", text_output)
        _TRY_CATCH_NVP("TIMEMORY_JSON_OUTPUT", json_output)
        _TRY_CATCH_NVP("TIMEMORY_DART_OUTPUT", dart_output)
        _TRY_CATCH_NVP("TIMEMORY_TIME_OUTPUT", time_output)
        _TRY_CATCH_NVP("TIMEMORY_VERBOSE", verbose)
        _TRY_CATCH_NVP("TIMEMORY_DEBUG", debug)
        _TRY_CATCH_NVP("TIMEMORY_BANNER", banner)
        _TRY_CATCH_NVP("TIMEMORY_FLAT_PROFILE", flat_profile)
        _TRY_CATCH_NVP("TIMEMORY_COLLAPSE_THREADS", collapse_threads)
        _TRY_CATCH_NVP("TIMEMORY_MAX_DEPTH", max_depth)
        _TRY_CATCH_NVP("TIMEMORY_TIME_FORMAT", time_format)
        _TRY_CATCH_NVP("TIMEMORY_PRECISION", precision)
        _TRY_CATCH_NVP("TIMEMORY_WIDTH", width)
        _TRY_CATCH_NVP("TIMEMORY_SCIENTIFIC", scientific)
        _TRY_CATCH_NVP("TIMEMORY_TIMING_PRECISION", timing_precision)
        _TRY_CATCH_NVP("TIMEMORY_TIMING_WIDTH", timing_width)
        _TRY_CATCH_NVP("TIMEMORY_TIMING_UNITS", timing_units)
        _TRY_CATCH_NVP("TIMEMORY_TIMING_SCIENTIFIC", timing_scientific)
        _TRY_CATCH_NVP("TIMEMORY_MEMORY_PRECISION", memory_precision)
        _TRY_CATCH_NVP("TIMEMORY_MEMORY_WIDTH", memory_width)
        _TRY_CATCH_NVP("TIMEMORY_MEMORY_UNITS", memory_units)
        _TRY_CATCH_NVP("TIMEMORY_MEMORY_SCIENTIFIC", memory_scientific)
        _TRY_CATCH_NVP("TIMEMORY_MPI_INIT", mpi_init)
        _TRY_CATCH_NVP("TIMEMORY_MPI_FINALIZE", mpi_finalize)
        _TRY_CATCH_NVP("TIMEMORY_MPI_THREAD", mpi_thread)
        _TRY_CATCH_NVP("TIMEMORY_MPI_THREAD_TYPE", mpi_thread_type)
        _TRY_CATCH_NVP("TIMEMORY_MPI_OUTPUT_PER_RANK", mpi_output_per_rank)
        _TRY_CATCH_NVP("TIMEMORY_MPI_OUTPUT_PER_NODE", mpi_output_per_node)
        _TRY_CATCH_NVP("TIMEMORY_OUTPUT_PATH", output_path)
        _TRY_CATCH_NVP("TIMEMORY_OUTPUT_PREFIX", output_prefix)
        _TRY_CATCH_NVP("TIMEMORY_DART_TYPE", dart_type)
        _TRY_CATCH_NVP("TIMEMORY_DART_COUNT", dart_count)
        _TRY_CATCH_NVP("TIMEMORY_DART_LABEL", dart_label)
        _TRY_CATCH_NVP("TIMEMORY_CPU_AFFINITY", cpu_affinity)
        _TRY_CATCH_NVP("TIMEMORY_PAPI_MULTIPLEXING", papi_multiplexing)
        _TRY_CATCH_NVP("TIMEMORY_PAPI_FAIL_ON_ERROR", papi_fail_on_error)
        _TRY_CATCH_NVP("TIMEMORY_PAPI_QUIET", papi_quiet)
        _TRY_CATCH_NVP("TIMEMORY_PAPI_EVENTS", papi_events)
        _TRY_CATCH_NVP("TIMEMORY_CUDA_EVENT_BATCH_SIZE", cuda_event_batch_size)
        _TRY_CATCH_NVP("TIMEMORY_NVTX_MARKER_DEVICE_SYNC", nvtx_marker_device_sync)
        _TRY_CATCH_NVP("TIMEMORY_CUPTI_ACTIVITY_LEVEL", cupti_activity_level)
        _TRY_CATCH_NVP("TIMEMORY_CUPTI_ACTIVITY_KINDS", cupti_activity_kinds)
        _TRY_CATCH_NVP("TIMEMORY_CUPTI_EVENTS", cupti_events)
        _TRY_CATCH_NVP("TIMEMORY_CUPTI_METRICS", cupti_metrics)
        _TRY_CATCH_NVP("TIMEMORY_CUPTI_DEVICE", cupti_device)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_MODE", roofline_mode)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_MODE_CPU", cpu_roofline_mode)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_MODE_GPU", gpu_roofline_mode)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_EVENTS_CPU", cpu_roofline_events)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_EVENTS_GPU", gpu_roofline_events)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_TYPE_LABELS", roofline_type_labels)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_TYPE_LABELS_CPU", roofline_type_labels_cpu)
        _TRY_CATCH_NVP("TIMEMORY_ROOFLINE_TYPE_LABELS_GPU", roofline_type_labels_gpu)
        _TRY_CATCH_NVP("TIMEMORY_INSTRUCTION_ROOFLINE", instruction_roofline)
        _TRY_CATCH_NVP("TIMEMORY_ERT_NUM_THREADS", ert_num_threads)
        _TRY_CATCH_NVP("TIMEMORY_ERT_NUM_THREADS_CPU", ert_num_threads_cpu)
        _TRY_CATCH_NVP("TIMEMORY_ERT_NUM_THREADS_GPU", ert_num_threads_gpu)
        _TRY_CATCH_NVP("TIMEMORY_ERT_NUM_STREAMS", ert_num_streams)
        _TRY_CATCH_NVP("TIMEMORY_ERT_GRID_SIZE", ert_grid_size)
        _TRY_CATCH_NVP("TIMEMORY_ERT_BLOCK_SIZE", ert_block_size)
        _TRY_CATCH_NVP("TIMEMORY_ERT_ALIGNMENT", ert_alignment)
        _TRY_CATCH_NVP("TIMEMORY_ERT_MIN_WORKING_SIZE", ert_min_working_size)
        _TRY_CATCH_NVP("TIMEMORY_ERT_MIN_WORKING_SIZE_CPU", ert_min_working_size_cpu)
        _TRY_CATCH_NVP("TIMEMORY_ERT_MIN_WORKING_SIZE_GPU", ert_min_working_size_gpu)
        _TRY_CATCH_NVP("TIMEMORY_ERT_MAX_DATA_SIZE", ert_max_data_size)
        _TRY_CATCH_NVP("TIMEMORY_ERT_MAX_DATA_SIZE_CPU", ert_max_data_size_cpu)
        _TRY_CATCH_NVP("TIMEMORY_ERT_MAX_DATA_SIZE_GPU", ert_max_data_size_gpu)
        _TRY_CATCH_NVP("TIMEMORY_ERT_SKIP_OPS", ert_skip_ops)
        _TRY_CATCH_NVP("TIMEMORY_ALLOW_SIGNAL_HANDLER", allow_signal_handler)
        _TRY_CATCH_NVP("TIMEMORY_ENABLE_SIGNAL_HANDLER", enable_signal_handler)
        _TRY_CATCH_NVP("TIMEMORY_ENABLE_ALL_SIGNALS", enable_all_signals)
        _TRY_CATCH_NVP("TIMEMORY_DISABLE_ALL_SIGNALS", disable_all_signals)
        _TRY_CATCH_NVP("TIMEMORY_NODE_COUNT", node_count)
        _TRY_CATCH_NVP("TIMEMORY_DESTRUCTOR_REPORT", destructor_report)
        _TRY_CATCH_NVP("TIMEMORY_PYTHON_EXE", python_exe)
        _TRY_CATCH_NVP("TIMEMORY_COMMAND_LINE", command_line)
        _TRY_CATCH_NVP("TIMEMORY_ENVIRONMENT", environment)
        _TRY_CATCH_NVP("TIMEMORY_UPCXX_INIT", upcxx_init)
        _TRY_CATCH_NVP("TIMEMORY_UPCXX_FINALIZE", upcxx_finalize)
#undef _TRY_CATCH_NVP
    }

    //----------------------------------------------------------------------------------//

    static tim_api std::vector<std::string> get_environment()
    {
#if defined(_UNIX)
        std::vector<std::string> _environ;
        if(environ != nullptr)
        {
            int idx = 0;
            while(environ[idx] != nullptr)
                _environ.push_back(environ[idx++]);
        }
        return _environ;
#else
        return std::vector<std::string>();
#endif
    }

    //----------------------------------------------------------------------------------//

};  // struct settings

}  // namespace tim

#if !defined(TIMEMORY_ERROR_FUNCTION_MACRO)
#    if defined(__PRETTY_FUNCTION__)
#        define TIMEMORY_ERROR_FUNCTION_MACRO __PRETTY_FUNCTION__
#    else
#        define TIMEMORY_ERROR_FUNCTION_MACRO __FUNCTION__
#    endif
#endif

//--------------------------------------------------------------------------------------//
// function to parse the environment for settings
//
// Nearly all variables will parse env when first access but this allows provides a
// way to reparse the environment so that default settings (possibly from previous
// invocation) can be overwritten
//
inline void
tim::settings::parse()
{
    if(suppress_parsing())
        return;

    for(auto& itr : get_parse_callbacks())
        itr();
}

//--------------------------------------------------------------------------------------//
