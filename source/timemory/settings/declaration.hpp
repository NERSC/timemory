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

/**
 * \file timemory/settings/declaration.hpp
 * \brief The declaration for the types for settings without definitions
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/compat/macros.h"
#include "timemory/environment/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/utility.hpp"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#if defined(_UNIX)
//
#    include <ctime>
#    include <unistd.h>
//
extern "C"
{
    extern char** environ;
}
#endif

namespace tim
{
class manager;
//
//--------------------------------------------------------------------------------------//
//
//                              settings
//
//--------------------------------------------------------------------------------------//
//
struct TIMEMORY_SETTINGS_DLL settings
{
    friend class manager;
    using string_t    = std::string;
    using strvector_t = std::vector<std::string>;
    using strmap_t    = std::map<std::string, std::string>;

    template <typename Tag = api::native_tag>
    static std::shared_ptr<settings> shared_instance() TIMEMORY_VISIBILITY("default");

    template <typename Tag = api::native_tag>
    static settings* instance() TIMEMORY_VISIBILITY("default");

    settings()  = default;
    ~settings() = default;

    settings(const settings&) = default;
    settings(settings&&)      = default;

    settings& operator=(const settings&) = default;
    settings& operator=(settings&&) = default;

    //==================================================================================//
    //
    //                  GENERAL SETTINGS THAT APPLY TO MULTIPLE COMPONENTS
    //
    //==================================================================================//

    // logical settings
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, suppress_parsing, "TIMEMORY_SUPPRESS_PARSING",
                                    "Disable parsing environment", false)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, enabled, "TIMEMORY_ENABLED",
                                    "Activation state of timemory",
                                    TIMEMORY_DEFAULT_ENABLED)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, auto_output, "TIMEMORY_AUTO_OUTPUT",
                                    "Generate output at application termination", true)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, cout_output, "TIMEMORY_COUT_OUTPUT",
                                    "Write output to stdout", true)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, file_output, "TIMEMORY_FILE_OUTPUT",
                                    "Write output to files", true)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, text_output, "TIMEMORY_TEXT_OUTPUT",
                                    "Write text output files", true)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, json_output, "TIMEMORY_JSON_OUTPUT",
                                    "Write json output files", true)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, dart_output, "TIMEMORY_DART_OUTPUT",
                                    "Write dart measurements for CDash", false)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, time_output, "TIMEMORY_TIME_OUTPUT",
        "Output data to subfolder w/ a timestamp (see also: TIMEMORY_TIME_FORMAT)", false)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, plot_output, "TIMEMORY_PLOT_OUTPUT",
                                    "Generate plot outputs from json outputs",
                                    TIMEMORY_DEFAULT_PLOTTING)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, diff_output, "TIMEMORY_DIFF_OUTPUT",
        "Generate a difference output vs. a pre-existing output (see also: "
        "TIMEMORY_INPUT_PATH and TIMEMORY_INPUT_PREFIX)",
        false)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, flamegraph_output, "TIMEMORY_FLAMEGRAPH_OUTPUT",
        "Write a json output for flamegraph visualization (use chrome://tracing)", true)

    // general settings
    TIMEMORY_MEMBER_STATIC_ACCESSOR(int, verbose, "TIMEMORY_VERBOSE", "Verbosity level",
                                    0)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, debug, "TIMEMORY_DEBUG", "Enable debug output",
                                    false)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, banner, "TIMEMORY_BANNER",
                                    "Notify about tim::manager creation and destruction",
                                    (get_env<bool>("TIMEMORY_LIBRARY_CTOR", true)))
    TIMEMORY_MEMBER_STATIC_REFERENCE(
        bool, flat_profile, "TIMEMORY_FLAT_PROFILE",
        "Set the label hierarchy mode to default to flat",
        ([]() -> bool& { return scope::get_fields()[scope::flat::value]; }),
        ([](bool v) { scope::get_fields()[scope::flat::value] = v; }))
    TIMEMORY_MEMBER_STATIC_REFERENCE(
        bool, timeline_profile, "TIMEMORY_TIMELINE_PROFILE",
        "Set the label hierarchy mode to default to timeline",
        ([]() -> bool& { return scope::get_fields()[scope::timeline::value]; }),
        ([](bool v) { scope::get_fields()[scope::timeline::value] = v; }))
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, collapse_threads, "TIMEMORY_COLLAPSE_THREADS",
                                    "Enable/disable combining thread-specific data", true)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(uint16_t, max_depth, "TIMEMORY_MAX_DEPTH",
                                    "Set the maximum depth of label hierarchy reporting",
                                    std::numeric_limits<uint16_t>::max())
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, time_format, "TIMEMORY_TIME_FORMAT",
        "Customize the folder generation when TIMEMORY_TIME_OUTPUT is enabled (see also: "
        "strftime)",
        "%F_%I.%M_%p")

    // general formatting
    TIMEMORY_MEMBER_STATIC_ACCESSOR(int16_t, precision, "TIMEMORY_PRECISION",
                                    "Set the global output precision for components", -1)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(int16_t, width, "TIMEMORY_WIDTH",
                                    "Set the global output width for components", -1)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(int32_t, max_width, "TIMEMORY_MAX_WIDTH",
                                    "Set the maximum width for component label outputs",
                                    120)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, scientific, "TIMEMORY_SCIENTIFIC",
        "Set the global numerical reporting to scientific format", false)

    // timing formatting
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        int16_t, timing_precision, "TIMEMORY_TIMING_PRECISION",
        "Set the precision for components with 'is_timing_category' type-trait", -1)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        int16_t, timing_width, "TIMEMORY_TIMING_WIDTH",
        "Set the output width for components with 'is_timing_category' type-trait", -1)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, timing_units, "TIMEMORY_TIMING_UNITS",
        "Set the units for components with 'uses_timing_units' type-trait", "")
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, timing_scientific, "TIMEMORY_TIMING_SCIENTIFIC",
                                    "Set the numerical reporting format for components "
                                    "with 'is_timing_category' type-trait",
                                    false)

    // memory formatting
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        int16_t, memory_precision, "TIMEMORY_MEMORY_PRECISION",
        "Set the precision for components with 'is_memory_category' type-trait", -1)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        int16_t, memory_width, "TIMEMORY_MEMORY_WIDTH",
        "Set the output width for components with 'is_memory_category' type-trait", -1)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, memory_units, "TIMEMORY_MEMORY_UNITS",
        "Set the units for components with 'uses_memory_units' type-trait", "")
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, memory_scientific, "TIMEMORY_MEMORY_SCIENTIFIC",
                                    "Set the numerical reporting format for components "
                                    "with 'is_memory_category' type-trait",
                                    false)

    // output control
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, output_path, "TIMEMORY_OUTPUT_PATH",
                                    "Explicitly specify the output folder for results",
                                    "timemory-output")  // folder
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, output_prefix, "TIMEMORY_OUTPUT_PREFIX",
                                    "Explicitly specify a prefix for all output files",
                                    "")  // file prefix

    // input control (for computing differences)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, input_path, "TIMEMORY_INPUT_PATH",
                                    "Explicitly specify the input folder for difference "
                                    "comparisons (see also: TIMEMORY_DIFF_OUTPUT)",
                                    "")  // folder
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, input_prefix, "TIMEMORY_INPUT_PREFIX",
        "Explicitly specify the prefix for input files used in difference comparisons "
        "(see also: TIMEMORY_DIFF_OUTPUT)",
        "")  // file prefix
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, input_extensions, "TIMEMORY_INPUT_EXTENSIONS",
        "File extensions used when searching for input files used in difference "
        "comparisons (see also: TIMEMORY_DIFF_OUTPUT)",
        "json,xml")  // extensions

    // dart control
    /// only echo this measurement type
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, dart_type, "TIMEMORY_DART_TYPE",
        "Only echo this measurement type (see also: TIMEMORY_DART_OUTPUT)", "")
    /// only echo this many dart tags
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, dart_count, "TIMEMORY_DART_COUNT",
        "Only echo this number of dart tags (see also: TIMEMORY_DART_OUTPUT)", 1)
    /// echo the category, not the identifier
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, dart_label, "TIMEMORY_DART_LABEL",
        "Echo the category instead of the label (see also: TIMEMORY_DART_OUTPUT)", true)

    /// enable thread affinity
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, cpu_affinity, "TIMEMORY_CPU_AFFINITY",
                                    "Enable pinning threads to CPUs (Linux-only)", false)
    /// target pid
    TIMEMORY_MEMBER_STATIC_REFERENCE(
        process::id_t, target_pid, "TIMEMORY_TARGET_PID",
        "Process ID for the components which require this",
        ([]() -> process::id_t& { return process::get_target_id(); }),
        ([](process::id_t v) { process::get_target_id() = v; }))
    /// configure component storage stack clearing
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, stack_clearing, "TIMEMORY_STACK_CLEARING",
        "Enable/disable stopping any markers still running during finalization", true)

    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, add_secondary, "TIMEMORY_ADD_SECONDARY",
        "Enable/disable components adding secondary (child) entries", true)

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
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, mpi_init, "TIMEMORY_MPI_INIT",
        "Enable/disable timemory calling MPI_Init / MPI_Init_thread during certain "
        "timemory_init(...) invocations",
        false)

    /// timemory will try to call MPI_Finalize during timemory_finalize()
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, mpi_finalize, "TIMEMORY_MPI_FINALIZE",
                                    "Enable/disable timemory calling MPI_Finalize during "
                                    "timemory_finalize(...) invocations",
                                    false)

    /// use MPI_Init and MPI_Init_thread
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, mpi_thread, "TIMEMORY_MPI_THREAD",
        "Call MPI_Init_thread instead of MPI_Init (see also: TIMEMORY_MPI_INIT)", true)

    /// use MPI_Init_thread type
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, mpi_thread_type, "TIMEMORY_MPI_THREAD_TYPE",
                                    "MPI_Init_thread mode: 'single', 'serialized', "
                                    "'funneled', or 'multiple' (see also: "
                                    "TIMEMORY_MPI_INIT and TIMEMORY_MPI_THREAD)",
                                    "")

    /// output MPI data per rank
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, mpi_output_per_rank,
                                    "TIMEMORY_MPI_OUTPUT_PER_RANK",
                                    "Generate MPI output per-rank (skip aggregation)",
                                    false)

    /// output MPI data per node
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, mpi_output_per_node,
                                    "TIMEMORY_MPI_OUTPUT_PER_NODE",
                                    "Aggregate MPI output per-node", false)

    //----------------------------------------------------------------------------------//
    //      UPC++
    //----------------------------------------------------------------------------------//

    /// timemory will try to call upcxx::init during certain timemory_init()
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, upcxx_init, "TIMEMORY_UPCXX_INIT",
        "Enable/disable timemory calling upcxx::init() during certain "
        "timemory_init(...) invocations",
        true)

    /// timemory will try to call upcxx::finalize during timemory_finalize()
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, upcxx_finalize, "TIMEMORY_UPCXX_FINALIZE",
        "Enable/disable timemory calling upcxx::finalize() during "
        "timemory_finalize()",
        true)

    //----------------------------------------------------------------------------------//
    //      PROCESS
    //----------------------------------------------------------------------------------//

    //----------------------------------------------------------------------------------//
    //      PAPI
    //----------------------------------------------------------------------------------//

    /// allow multiplexing
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, papi_multiplexing, "TIMEMORY_PAPI_MULTIPLEXING",
                                    "Enable multiplexing when using PAPI", true)

    /// errors with PAPI will throw
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, papi_fail_on_error,
                                    "TIMEMORY_PAPI_FAIL_ON_ERROR",
                                    "Configure PAPI errors to trigger a runtime error",
                                    false)

    /// errors with PAPI will be suppressed
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, papi_quiet, "TIMEMORY_PAPI_QUIET",
        "Configure suppression of reporting PAPI errors/warnings", false)

    /// PAPI hardware counters
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, papi_events, "TIMEMORY_PAPI_EVENTS",
        "PAPI presets and events to collect (see also: papi_avail)", "")

    /// PAPI attach to pid
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, papi_attach, "TIMEMORY_PAPI_ATTACH",
        "Configure PAPI to attach to another process (see also: TIMEMORY_TARGET_PID)",
        false)

    /// PAPI overflow threshold
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        int, papi_overflow, "TIMEMORY_PAPI_OVERFLOW",
        "Value at which PAPI hw counters trigger an overflow callback", 0)

    //----------------------------------------------------------------------------------//
    //      CUDA / CUPTI
    //----------------------------------------------------------------------------------//

    /// batch size for create cudaEvent_t in cuda_event components
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, cuda_event_batch_size, "TIMEMORY_CUDA_EVENT_BATCH_SIZE",
        "Batch size for create cudaEvent_t in cuda_event components", 5)

    /// Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, nvtx_marker_device_sync, "TIMEMORY_NVTX_MARKER_DEVICE_SYNC",
        "Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)", true)

    /// default group of kinds tracked via CUpti Activity API
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        int32_t, cupti_activity_level, "TIMEMORY_CUPTI_ACTIVITY_LEVEL",
        "Default group of kinds tracked via CUpti Activity API", 1)

    /// specific activity kinds
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, cupti_activity_kinds,
                                    "TIMEMORY_CUPTI_ACTIVITY_KINDS",
                                    "Specific cupti activity kinds to track", "")

    /// CUPTI events
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, cupti_events, "TIMEMORY_CUPTI_EVENTS",
        "Hardware counter event types to collect on NVIDIA GPUs", "")

    /// CUPTI metrics
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, cupti_metrics, "TIMEMORY_CUPTI_METRICS",
        "Hardware counter metric types to collect on NVIDIA GPUs", "")

    /// Device to use CUPTI on
    TIMEMORY_MEMBER_STATIC_ACCESSOR(int, cupti_device, "TIMEMORY_CUPTI_DEVICE",
                                    "Target device for CUPTI hw counter collection", 0)

    //----------------------------------------------------------------------------------//
    //      ROOFLINE
    //----------------------------------------------------------------------------------//

    /// roofline mode for roofline components
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, roofline_mode, "TIMEMORY_ROOFLINE_MODE",
        "Configure the roofline collection mode. Options: 'op' 'ai'.", "op")

    /// set the roofline mode when running ERT on CPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, cpu_roofline_mode, "TIMEMORY_ROOFLINE_MODE_CPU",
        "Configure the roofline collection mode for CPU specifically. Options: 'op' "
        "'ai'",
        m__roofline_mode)

    /// set the roofline mode when running ERT on GPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, gpu_roofline_mode, "TIMEMORY_ROOFLINE_MODE_GPU",
        "Configure the roofline collection mode for GPU specifically. Options: 'op' "
        "'ai'.",
        m__roofline_mode)

    /// custom hw counters to add to the cpu roofline
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, cpu_roofline_events, "TIMEMORY_ROOFLINE_EVENTS_CPU",
        "Configure custom hw counters to add to the cpu roofline", "")

    /// custom hw counters to add to the gpu roofline
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, gpu_roofline_events, "TIMEMORY_ROOFLINE_EVENTS_GPU",
        "Configure custom hw counters to add to the gpu roofline", "")

    /// roofline labels/descriptions/output-files encode the list of data types
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, roofline_type_labels,
                                    "TIMEMORY_ROOFLINE_TYPE_LABELS",
                                    "Configure roofline labels/descriptions/output-files "
                                    "encode the list of data types",
                                    false)

    /// set the roofline mode when running ERT on CPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, roofline_type_labels_cpu,
                                    "TIMEMORY_ROOFLINE_TYPE_LABELS_CPU",
                                    "Configure labels, etc. for the roofline components "
                                    "for CPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)",
                                    m__roofline_type_labels)

    /// set the roofline mode when running ERT on GPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, roofline_type_labels_gpu,
                                    "TIMEMORY_ROOFLINE_TYPE_LABELS_GPU",
                                    "Configure labels, etc. for the roofline components "
                                    "for GPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)",
                                    m__roofline_type_labels)

    /// include the instruction roofline
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, instruction_roofline,
                                    "TIMEMORY_INSTRUCTION_ROOFLINE",
                                    "Configure the roofline to include the hw counters "
                                    "required for generating an instruction roofline",
                                    false)

    //----------------------------------------------------------------------------------//
    //      ERT
    //----------------------------------------------------------------------------------//

    /// set the number of threads when running ERT (0 == default-specific)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(uint64_t, ert_num_threads, "TIMEMORY_ERT_NUM_THREADS",
                                    "Number of threads to use when running ERT", 0)

    /// set the number of threads when running ERT on CPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(uint64_t, ert_num_threads_cpu,
                                    "TIMEMORY_ERT_NUM_THREADS_CPU",
                                    "Number of threads to use when running ERT on CPU",
                                    std::thread::hardware_concurrency())

    /// set the number of threads when running ERT on GPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_num_threads_gpu, "TIMEMORY_ERT_NUM_THREADS_GPU",
        "Number of threads which launch kernels when running ERT on the GPU", 1)

    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_num_streams, "TIMEMORY_ERT_NUM_STREAMS",
        "Number of streams to use when launching kernels in ERT on the GPU", 1)

    /// set the grid size (number of blocks) for ERT on GPU (0 == auto-compute)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_grid_size, "TIMEMORY_ERT_GRID_SIZE",
        "Configure the grid size (number of blocks) for ERT on GPU (0 == auto-compute)",
        0)

    /// set the block size (number of threads per block) for ERT on GPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_block_size, "TIMEMORY_ERT_BLOCK_SIZE",
        "Configure the block size (number of threads per block) for ERT on GPU", 1024)

    /// set the alignment (in bits) when running ERT on CPU (0 == 8 * sizeof(T))
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_alignment, "TIMEMORY_ERT_ALIGNMENT",
        "Configure the alignment (in bits) when running ERT on CPU (0 == 8 * sizeof(T))",
        0)

    /// set the minimum working size when running ERT (0 == device specific)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_min_working_size, "TIMEMORY_ERT_MIN_WORKING_SIZE",
        "Configure the minimum working size when running ERT (0 == device specific)", 0)

    /// set the minimum working size when running ERT on CPU
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_min_working_size_cpu, "TIMEMORY_ERT_MIN_WORKING_SIZE_CPU",
        "Configure the minimum working size when running ERT on CPU", 64)

    /// set the minimum working size when running ERT on GPU (default is 10 MB)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        uint64_t, ert_min_working_size_gpu, "TIMEMORY_ERT_MIN_WORKING_SIZE_GPU",
        "Configure the minimum working size when running ERT on GPU", 10 * 1000 * 1000)

    /// set the max data size when running ERT on CPU (0 == device-specific)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(uint64_t, ert_max_data_size,
                                    "TIMEMORY_ERT_MAX_DATA_SIZE",
                                    "Configure the max data size when running ERT on CPU",
                                    0)

    /// set the max data size when running ERT on CPU (0 == 2 * max-cache-size)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(uint64_t, ert_max_data_size_cpu,
                                    "TIMEMORY_ERT_MAX_DATA_SIZE_CPU",
                                    "Configure the max data size when running ERT on CPU",
                                    0)

    /// set the max data size when running ERT on GPU (default is 500 MB)
    TIMEMORY_MEMBER_STATIC_ACCESSOR(uint64_t, ert_max_data_size_gpu,
                                    "TIMEMORY_ERT_MAX_DATA_SIZE_GPU",
                                    "Configure the max data size when running ERT on GPU",
                                    500 * 1000 * 1000)

    /// set the ops to skip at runtime
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        string_t, ert_skip_ops, "TIMEMORY_ERT_SKIP_OPS",
        "Skip these number of ops (i.e. ERT_FLOPS) when were set at compile time", "")

    //----------------------------------------------------------------------------------//
    //      Craypat
    //----------------------------------------------------------------------------------//

    /// set the craypat categories to collect
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, craypat_categories, "TIMEMORY_CRAYPAT",
                                    "Configure the CrayPAT categories to collect",
                                    get_env<std::string>("PAT_RT_PERFCTR", ""))

    //----------------------------------------------------------------------------------//
    //      Signals
    //--------------------------------------------/settings/declaration.hpp-------------------------//

    /// allow signal handling to be activated
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, allow_signal_handler,
                                    "TIMEMORY_ALLOW_SIGNAL_HANDLER",
                                    "Allow signal handling to be activated", true)

    /// enable signals in timemory_init
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, enable_signal_handler,
                                    "TIMEMORY_ENABLE_SIGNAL_HANDLER",
                                    "Enable signals in timemory_init", false)

    /// enable all signals
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, enable_all_signals,
                                    "TIMEMORY_ENABLE_ALL_SIGNALS",
                                    "Enable catching all signals", false)

    /// disable all signals
    TIMEMORY_MEMBER_STATIC_ACCESSOR(bool, disable_all_signals,
                                    "TIMEMORY_DISABLE_ALL_SIGNALS",
                                    "Disable catching any signals", false)

    //----------------------------------------------------------------------------------//
    //     Number of nodes
    //----------------------------------------------------------------------------------//

    TIMEMORY_MEMBER_STATIC_ACCESSOR(int32_t, node_count, "TIMEMORY_NODE_COUNT",
                                    "Total number of nodes used in application", 0)

    //----------------------------------------------------------------------------------//
    //     For auto_* types
    //----------------------------------------------------------------------------------//

    /// default setting for auto_{list,tuple,hybrid} "report_at_exit" member variable
    TIMEMORY_MEMBER_STATIC_ACCESSOR(
        bool, destructor_report, "TIMEMORY_DESTRUCTOR_REPORT",
        "Configure default setting for auto_{list,tuple,hybrid} to write to stdout during"
        " destruction of the bundle",
        false)

    //----------------------------------------------------------------------------------//
    //     For plotting
    //----------------------------------------------------------------------------------//

    /// default setting for python invocation when plotting from C++ code
    TIMEMORY_MEMBER_STATIC_ACCESSOR(string_t, python_exe, "TIMEMORY_PYTHON_EXE",
                                    "Configure the python executable to use",
                                    TIMEMORY_PYTHON_PLOTTER)

    //----------------------------------------------------------------------------------//
    //     Command line
    //----------------------------------------------------------------------------------//

    TIMEMORY_STATIC_ACCESSOR(strvector_t, command_line, strvector_t{})

    //----------------------------------------------------------------------------------//

public:
    static strvector_t get_environment()
    {
#if defined(_UNIX)
        strvector_t _environ;
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

    TIMEMORY_STATIC_ACCESSOR(strvector_t, environment, get_environment())

public:
    static string_t tolower(string_t str);
    static string_t toupper(string_t str);
    static string_t get_input_prefix();
    static string_t get_output_prefix(bool fake = false);
    static void     store_command_line(int argc, char** argv);
    static string_t compose_output_filename(const string_t& _tag, string_t _ext,
                                            bool          _mpi_init = false,
                                            const int32_t _mpi_rank = -1,
                                            bool          fake      = false,
                                            std::string   _explicit = "");
    static string_t compose_input_filename(const string_t& _tag, string_t _ext,
                                           bool          _mpi_init = false,
                                           const int32_t _mpi_rank = -1,
                                           std::string   _explicit = "");

    static void parse();

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int);

    template <typename Archive>
    static void serialize_settings(Archive& ar)
    {
        if(settings::instance())
            ar(cereal::make_nvp("settings", *settings::instance()));
    }

    template <typename Archive>
    static void serialize_settings(Archive& ar, settings& _obj)
    {
        ar(cereal::make_nvp("settings", _obj));
    }

    template <size_t Idx = 0>
    static int64_t indent_width(int64_t _w = settings::width());

    template <typename Tp, size_t Idx = 0>
    static int64_t indent_width(int64_t _w = indent_width<Idx>());

    template <typename Tp>
    static size_t data_width(int64_t _idx, int64_t _w);
};
//
//----------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>
settings::shared_instance()
{
    static std::shared_ptr<settings> _instance = std::make_shared<settings>();
    return _instance;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tag>
settings*
settings::instance()
{
    static auto _instance = shared_instance<Tag>();
    return _instance.get();
}
//
//----------------------------------------------------------------------------------//
//
template <size_t Idx>
int64_t
settings::indent_width(int64_t _w)
{
    static std::atomic<int64_t> _instance(_w);
    _instance.store(std::max<int64_t>(_instance.load(), _w));
    return _instance.load();
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp, size_t Idx>
int64_t
settings::indent_width(int64_t _w)
{
    static std::atomic<int64_t> _instance(_w);
    _instance.store(std::max<int64_t>(_instance.load(), _w));
    return _instance.load();
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
size_t
settings::data_width(int64_t _idx, int64_t _w)
{
    static std::vector<int64_t> _instance;
    static std::recursive_mutex _mtx;
    auto_lock_t                 lk(_mtx);
    if(_idx >= (int64_t) _instance.size())
        _instance.resize(_idx + 1, _w);
    _instance[_idx] = std::max<int64_t>(_instance[_idx], _w);
    return _instance[_idx];
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::serialize(Archive& ar, const unsigned int)
{
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_SUPPRESS_PARSING", suppress_parsing)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ENABLED", enabled)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_AUTO_OUTPUT", auto_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_COUT_OUTPUT", cout_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_FILE_OUTPUT", file_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TEXT_OUTPUT", text_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_JSON_OUTPUT", json_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_OUTPUT", dart_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIME_OUTPUT", time_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PLOT_OUTPUT", plot_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DIFF_OUTPUT", diff_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_FLAMEGRAPH_OUTPUT", flamegraph_output)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_VERBOSE", verbose)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DEBUG", debug)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_BANNER", banner)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_FLAT_PROFILE", flat_profile)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIMELINE_PROFILE", timeline_profile)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_COLLAPSE_THREADS", collapse_threads)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MAX_DEPTH", max_depth)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIME_FORMAT", time_format)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PRECISION", precision)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_WIDTH", width)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MAX_WIDTH", max_width)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_SCIENTIFIC", scientific)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIMING_PRECISION", timing_precision)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIMING_WIDTH", timing_width)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIMING_UNITS", timing_units)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TIMING_SCIENTIFIC", timing_scientific)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MEMORY_PRECISION", memory_precision)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MEMORY_WIDTH", memory_width)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MEMORY_UNITS", memory_units)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MEMORY_SCIENTIFIC", memory_scientific)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_INIT", mpi_init)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_FINALIZE", mpi_finalize)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_THREAD", mpi_thread)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_THREAD_TYPE", mpi_thread_type)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_OUTPUT_PER_RANK", mpi_output_per_rank)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_OUTPUT_PER_NODE", mpi_output_per_node)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_OUTPUT_PATH", output_path)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_OUTPUT_PREFIX", output_prefix)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INPUT_PATH", input_path)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INPUT_PREFIX", input_prefix)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INPUT_EXTENSIONS", input_extensions)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_TYPE", dart_type)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_COUNT", dart_count)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_LABEL", dart_label)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CPU_AFFINITY", cpu_affinity)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TARGET_PID", target_pid)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_STACK_CLEARING", stack_clearing)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ADD_SECONDARY", add_secondary)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PAPI_MULTIPLEXING", papi_multiplexing)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PAPI_FAIL_ON_ERROR", papi_fail_on_error)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PAPI_QUIET", papi_quiet)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PAPI_EVENTS", papi_events)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PAPI_ATTACH", papi_attach)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PAPI_OVERFLOW", papi_overflow)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CUDA_EVENT_BATCH_SIZE",
                                    cuda_event_batch_size)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_NVTX_MARKER_DEVICE_SYNC",
                                    nvtx_marker_device_sync)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CUPTI_ACTIVITY_LEVEL", cupti_activity_level)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CUPTI_ACTIVITY_KINDS", cupti_activity_kinds)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CUPTI_EVENTS", cupti_events)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CUPTI_METRICS", cupti_metrics)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CUPTI_DEVICE", cupti_device)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_MODE", roofline_mode)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_MODE_CPU", cpu_roofline_mode)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_MODE_GPU", gpu_roofline_mode)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_EVENTS_CPU", cpu_roofline_events)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_EVENTS_GPU", gpu_roofline_events)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_TYPE_LABELS", roofline_type_labels)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_TYPE_LABELS_CPU",
                                    roofline_type_labels_cpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ROOFLINE_TYPE_LABELS_GPU",
                                    roofline_type_labels_gpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INSTRUCTION_ROOFLINE", instruction_roofline)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_NUM_THREADS", ert_num_threads)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_NUM_THREADS_CPU", ert_num_threads_cpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_NUM_THREADS_GPU", ert_num_threads_gpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_NUM_STREAMS", ert_num_streams)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_GRID_SIZE", ert_grid_size)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_BLOCK_SIZE", ert_block_size)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_ALIGNMENT", ert_alignment)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_MIN_WORKING_SIZE", ert_min_working_size)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_MIN_WORKING_SIZE_CPU",
                                    ert_min_working_size_cpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_MIN_WORKING_SIZE_GPU",
                                    ert_min_working_size_gpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_MAX_DATA_SIZE", ert_max_data_size)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_MAX_DATA_SIZE_CPU",
                                    ert_max_data_size_cpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_MAX_DATA_SIZE_GPU",
                                    ert_max_data_size_gpu)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ERT_SKIP_OPS", ert_skip_ops)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ALLOW_SIGNAL_HANDLER", allow_signal_handler)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ENABLE_SIGNAL_HANDLER",
                                    enable_signal_handler)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ENABLE_ALL_SIGNALS", enable_all_signals)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DISABLE_ALL_SIGNALS", disable_all_signals)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_NODE_COUNT", node_count)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DESTRUCTOR_REPORT", destructor_report)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PYTHON_EXE", python_exe)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_COMMAND_LINE", command_line)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ENVIRONMENT", environment)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_UPCXX_INIT", upcxx_init)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_UPCXX_FINALIZE", upcxx_finalize)
}
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim
