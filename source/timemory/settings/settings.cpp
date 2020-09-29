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

#include "timemory/settings/settings.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/bits/signals.hpp"
#include "timemory/utility/declaration.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/macros.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::serialize_settings(Archive& ar)
{
    if(settings::instance())
        ar(cereal::make_nvp("settings", *settings::instance()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::serialize_settings(Archive& ar, settings& _obj)
{
    ar(cereal::make_nvp("settings", _obj));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t&
settings::command_line()
{
    return instance()->get_environment();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t&
settings::environment()
{
    return instance()->get_environment();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t
settings::get_global_environment()
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
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
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
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::tolower(std::string str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::toupper(std::string str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_global_input_prefix()
{
    static auto& _dir    = input_path();
    static auto& _prefix = input_prefix();

    return filepath::osrepr(_dir + std::string("/") + _prefix);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_global_output_prefix(bool fake)
{
    static auto& _dir         = output_path();
    static auto& _prefix      = output_prefix();
    static auto& _time_output = time_output();
    static auto& _time_format = time_format();

    if(_time_output)
    {
        // ensure that all output files use same local datetime
        static auto _local_datetime = get_local_datetime(_time_format.c_str());
        if(_dir.find(_local_datetime) == std::string::npos)
        {
            if(_dir.length() > 0 && _dir[_dir.length() - 1] != '/')
                _dir += "/";
            _dir += _local_datetime;
        }
    }

    if(!fake && (debug() || verbose() > 2))
        PRINT_HERE("creating output directory: '%s'", _dir.c_str());

    if(fake)
        tim::consume_parameters(get_global_input_prefix());

    auto ret = (fake) ? 0 : makedir(_dir);
    return (ret == 0) ? filepath::osrepr(_dir + std::string("/") + _prefix)
                      : filepath::osrepr(std::string("./") + _prefix);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::store_command_line(int argc, char** argv)
{
    auto& _cmdline = command_line();
    _cmdline.clear();
    for(int i = 0; i < argc; ++i)
        _cmdline.push_back(std::string(argv[i]));
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::compose_output_filename(const std::string& _tag, std::string _ext,
                                  bool _mpi_init, const int32_t _mpi_rank, bool fake,
                                  std::string _explicit)
{
    auto _prefix = (_explicit.length() > 0) ? _explicit : get_global_output_prefix(fake);

    // if just caching this static variable return
    if(fake)
        return "";

    auto only_ascii = [](char c) { return !isascii(c); };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    if(_explicit.length() > 0)
    {
        auto ret = makedir(_prefix);
        if(ret != 0)
            _prefix = filepath::osrepr(std::string("./"));
    }

    auto _rank_suffix = (_mpi_init && _mpi_rank >= 0)
                            ? (std::string("_") + std::to_string(_mpi_rank))
                            : std::string("");
    if(_ext.find('.') != 0)
        _ext = std::string(".") + _ext;
    auto plast = _prefix.length() - 1;
    if(_prefix.length() > 0 && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "_";
    auto fpath = path_t(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != std::string::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::compose_input_filename(const std::string& _tag, std::string _ext,
                                 bool _mpi_init, const int32_t _mpi_rank,
                                 std::string _explicit)
{
    if(settings::input_path().empty())
        settings::input_path() = settings::output_path();

    if(settings::input_prefix().empty())
        settings::input_prefix() = settings::output_prefix();

    auto _prefix = (_explicit.length() > 0) ? _explicit : get_global_input_prefix();

    auto only_ascii = [](char c) { return !isascii(c); };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    if(_explicit.length() > 0)
        _prefix = filepath::osrepr(std::string("./"));

    auto _rank_suffix = (_mpi_init && _mpi_rank >= 0)
                            ? (std::string("_") + std::to_string(_mpi_rank))
                            : std::string("");
    if(_ext.find('.') != 0)
        _ext = std::string(".") + _ext;
    auto plast = _prefix.length() - 1;
    if(_prefix.length() > 0 && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "_";
    auto fpath = path_t(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != std::string::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::parse(std::shared_ptr<settings> _settings)
{
    if(_settings)
        parse(_settings.get());
}
//
//----------------------------------------------------------------------------------//
//
// function to parse the environment for settings
//
// Nearly all variables will parse env when first access but this allows provides a
// way to reparse the environment so that default settings (possibly from previous
// invocation) can be overwritten
//
TIMEMORY_SETTINGS_INLINE
void
settings::parse(settings* _settings)
{
    if(!_settings)
        return;

    if(_settings->get_suppress_parsing())
        return;

    for(auto& itr : *_settings)
    {
        if(_settings->get_debug() && _settings->get_verbose() > 0)
            std::cerr << "Executing parse callback for: " << itr.first << std::endl;
        itr.second->parse();
    }
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::settings(const settings& rhs)
: m_data(data_type{})
, m_command_line(rhs.m_command_line)
, m_environment(rhs.m_environment)
{
    for(auto& itr : rhs.m_data)
        m_data.insert({ itr.first, itr.second->clone() });
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings&
settings::operator=(const settings& rhs)
{
    if(this == &rhs)
        return *this;

    for(auto& itr : rhs.m_data)
        m_data[itr.first] = itr.second->clone();
    m_command_line = rhs.m_command_line;
    m_environment  = rhs.m_environment;
    return *this;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize()
{
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, config_file, "TIMEMORY_CONFIG", "Configuration file for timemory",
        TIMEMORY_JOIN("/", get_env<string_t>("HOME"), "timemory.cfg"),
        strvector_t({ "-C", "--timemory-config" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, enabled, "TIMEMORY_ENABLED", "Activation state of timemory",
        TIMEMORY_DEFAULT_ENABLED, strvector_t({ "--timemory-enabled" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(int, verbose, "TIMEMORY_VERBOSE", "Verbosity level",
                                      0, strvector_t({ "--timemory-verbose" }), 1)

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, debug, "TIMEMORY_DEBUG",
                                      "Enable debug output", false,
                                      strvector_t({ "--timemory-debug" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, auto_output, "TIMEMORY_AUTO_OUTPUT",
                                      "Generate output at application termination", true,
                                      strvector_t({ "--timemory-auto-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, cout_output, "TIMEMORY_COUT_OUTPUT",
                                      "Write output to stdout", true,
                                      strvector_t({ "--timemory-cout-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, file_output, "TIMEMORY_FILE_OUTPUT",
                                      "Write output to files", true,
                                      strvector_t({ "--timemory-file-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, text_output, "TIMEMORY_TEXT_OUTPUT",
                                      "Write text output files", true,
                                      strvector_t({ "--timemory-text-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, json_output, "TIMEMORY_JSON_OUTPUT",
                                      "Write json output files", true,
                                      strvector_t({ "--timemory-json-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, dart_output, "TIMEMORY_DART_OUTPUT",
                                      "Write dart measurements for CDash", false,
                                      strvector_t({ "--timemory-dart-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, time_output, "TIMEMORY_TIME_OUTPUT",
        "Output data to subfolder w/ a timestamp (see also: TIMEMORY_TIME_FORMAT)", false,
        strvector_t({ "--timemory-time-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, plot_output, "TIMEMORY_PLOT_OUTPUT",
                                      "Generate plot outputs from json outputs",
                                      TIMEMORY_DEFAULT_PLOTTING,
                                      strvector_t({ "--timemory-plot-output" }))
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, diff_output, "TIMEMORY_DIFF_OUTPUT",
        "Generate a difference output vs. a pre-existing output (see also: "
        "TIMEMORY_INPUT_PATH and TIMEMORY_INPUT_PREFIX)",
        false, strvector_t({ "--timemory-diff-output" }))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, flamegraph_output, "TIMEMORY_FLAMEGRAPH_OUTPUT",
        "Write a json output for flamegraph visualization (use chrome://tracing)", true,
        strvector_t({ "--timemory-flamegraph-output" }))

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, banner, "TIMEMORY_BANNER",
                                  "Notify about tim::manager creation and destruction",
                                  (get_env<bool>("TIMEMORY_LIBRARY_CTOR", false)))

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, flat_profile, "TIMEMORY_FLAT_PROFILE",
        "Set the label hierarchy mode to default to flat",
        scope::get_fields()[scope::flat::value],
        strvector_t({ "--timemory-flat-profile" }))

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, timeline_profile, "TIMEMORY_TIMELINE_PROFILE",
        "Set the label hierarchy mode to default to timeline",
        scope::get_fields()[scope::timeline::value],
        strvector_t({ "--timemory-timeline-profile" }))

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, collapse_threads, "TIMEMORY_COLLAPSE_THREADS",
                                  "Enable/disable combining thread-specific data", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, collapse_processes, "TIMEMORY_COLLAPSE_PROCESSES",
                                  "Enable/disable combining process-specific data", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(uint16_t, max_depth, "TIMEMORY_MAX_DEPTH",
                                  "Set the maximum depth of label hierarchy reporting",
                                  std::numeric_limits<uint16_t>::max())

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, time_format, "TIMEMORY_TIME_FORMAT",
        "Customize the folder generation when TIMEMORY_TIME_OUTPUT is enabled (see also: "
        "strftime)",
        "%F_%I.%M_%p")

    TIMEMORY_SETTINGS_MEMBER_IMPL(int16_t, precision, "TIMEMORY_PRECISION",
                                  "Set the global output precision for components", -1)
    TIMEMORY_SETTINGS_MEMBER_IMPL(int16_t, width, "TIMEMORY_WIDTH",
                                  "Set the global output width for components", -1)
    TIMEMORY_SETTINGS_MEMBER_IMPL(int32_t, max_width, "TIMEMORY_MAX_WIDTH",
                                  "Set the maximum width for component label outputs",
                                  120)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, scientific, "TIMEMORY_SCIENTIFIC",
        "Set the global numerical reporting to scientific format", false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, timing_precision, "TIMEMORY_TIMING_PRECISION",
        "Set the precision for components with 'is_timing_category' type-trait", -1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, timing_width, "TIMEMORY_TIMING_WIDTH",
        "Set the output width for components with 'is_timing_category' type-trait", -1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, timing_units, "TIMEMORY_TIMING_UNITS",
        "Set the units for components with 'uses_timing_units' type-trait", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, timing_scientific, "TIMEMORY_TIMING_SCIENTIFIC",
                                  "Set the numerical reporting format for components "
                                  "with 'is_timing_category' type-trait",
                                  false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, memory_precision, "TIMEMORY_MEMORY_PRECISION",
        "Set the precision for components with 'is_memory_category' type-trait", -1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, memory_width, "TIMEMORY_MEMORY_WIDTH",
        "Set the output width for components with 'is_memory_category' type-trait", -1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, memory_units, "TIMEMORY_MEMORY_UNITS",
        "Set the units for components with 'uses_memory_units' type-trait", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, memory_scientific, "TIMEMORY_MEMORY_SCIENTIFIC",
                                  "Set the numerical reporting format for components "
                                  "with 'is_memory_category' type-trait",
                                  false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(string_t, output_path, "TIMEMORY_OUTPUT_PATH",
                                  "Explicitly specify the output folder for results",
                                  "timemory-output")  // folder

    TIMEMORY_SETTINGS_MEMBER_IMPL(string_t, output_prefix, "TIMEMORY_OUTPUT_PREFIX",
                                  "Explicitly specify a prefix for all output files",
                                  "")  // file prefix

    TIMEMORY_SETTINGS_MEMBER_IMPL(string_t, input_path, "TIMEMORY_INPUT_PATH",
                                  "Explicitly specify the input folder for difference "
                                  "comparisons (see also: TIMEMORY_DIFF_OUTPUT)",
                                  "")  // folder

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, input_prefix, "TIMEMORY_INPUT_PREFIX",
        "Explicitly specify the prefix for input files used in difference comparisons "
        "(see also: TIMEMORY_DIFF_OUTPUT)",
        "")  // file prefix

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, input_extensions, "TIMEMORY_INPUT_EXTENSIONS",
        "File extensions used when searching for input files used in difference "
        "comparisons (see also: TIMEMORY_DIFF_OUTPUT)",
        "json,xml")  // extensions

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, dart_type, "TIMEMORY_DART_TYPE",
        "Only echo this measurement type (see also: TIMEMORY_DART_OUTPUT)", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, dart_count, "TIMEMORY_DART_COUNT",
        "Only echo this number of dart tags (see also: TIMEMORY_DART_OUTPUT)", 1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, dart_label, "TIMEMORY_DART_LABEL",
        "Echo the category instead of the label (see also: TIMEMORY_DART_OUTPUT)", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        size_t, max_thread_bookmarks, "TIMEMORY_MAX_THREAD_BOOKMARKS",
        "Maximum number of times a worker thread bookmarks the call-graph location w.r.t."
        " the master thread. Higher values tend to increase the finalization merge time",
        50)

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, cpu_affinity, "TIMEMORY_CPU_AFFINITY",
                                  "Enable pinning threads to CPUs (Linux-only)", false)

    TIMEMORY_SETTINGS_REFERENCE_IMPL(process::id_t, target_pid, "TIMEMORY_TARGET_PID",
                                     "Process ID for the components which require this",
                                     process::get_target_id());

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, stack_clearing, "TIMEMORY_STACK_CLEARING",
        "Enable/disable stopping any markers still running during finalization", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, add_secondary, "TIMEMORY_ADD_SECONDARY",
        "Enable/disable components adding secondary (child) entries", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(size_t, throttle_count, "TIMEMORY_THROTTLE_COUNT",
                                  "Minimum number of laps before throttling", 10000)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        size_t, throttle_value, "TIMEMORY_THROTTLE_VALUE",
        "Average call time in nanoseconds when # laps > throttle_count that triggers "
        "throttling",
        10000)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, global_components, "TIMEMORY_GLOBAL_COMPONENTS",
        "A specification of components which is used by multiple variadic bundlers and "
        "user_bundles as the fall-back set of components if their specific variable is "
        "not set. E.g. user_mpip_bundle will use this if TIMEMORY_MPIP_COMPONENTS is not "
        "specified",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, tuple_components, "TIMEMORY_TUPLE_COMPONENTS",
        "A specification of components which will be added to component_tuple structures "
        "containing the 'user_tuple_bundle'. These components will automatically be "
        "activated",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, list_components, "TIMEMORY_LIST_COMPONENTS",
        "A specification of components which will be added to component_list structures "
        "containing the 'user_list_bundle'. These components will only be activated if "
        "the 'user_list_bundle' is activated.",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, ompt_components, "TIMEMORY_OMPT_COMPONENTS",
        "A specification of components which will be added "
        "to structures containing the 'user_ompt_bundle'. Priority: TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, mpip_components, "TIMEMORY_MPIP_COMPONENTS",
        "A specification of components which will be added "
        "to structures containing the 'user_mpip_bundle'. Priority: TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, ncclp_components, "TIMEMORY_NCCLP_COMPONENTS",
        "A specification of components which will be added "
        "to structures containing the 'user_ncclp_bundle'. Priority: MPIP_COMPONENTS -> "
        "TRACE_COMPONENTS -> PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, trace_components, "TIMEMORY_TRACE_COMPONENTS",
        "A specification of components which will be used by the interfaces which are "
        "designed for full profiling. These components will be subjected to throttling. "
        "Priority: COMPONENTS -> GLOBAL_COMPONENTS",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, profiler_components, "TIMEMORY_PROFILER_COMPONENTS",
        "A specification of components which will be used by the interfaces which are "
        "designed for full python profiling. This specification will be overridden by a "
        "trace_components specification. Priority: COMPONENTS -> GLOBAL_COMPONENTS",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, components, "TIMEMORY_COMPONENTS",
        "A specification of components which is used by the library interface. This "
        "falls back to TIMEMORY_GLOBAL_COMPONENTS.",
        "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, mpi_init, "TIMEMORY_MPI_INIT",
        "Enable/disable timemory calling MPI_Init / MPI_Init_thread during certain "
        "timemory_init(...) invocations",
        false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, mpi_finalize, "TIMEMORY_MPI_FINALIZE",
                                  "Enable/disable timemory calling MPI_Finalize during "
                                  "timemory_finalize(...) invocations",
                                  false)

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, mpi_thread, "TIMEMORY_MPI_THREAD",
        "Call MPI_Init_thread instead of MPI_Init (see also: TIMEMORY_MPI_INIT)",
        mpi::use_mpi_thread())

    TIMEMORY_SETTINGS_REFERENCE_IMPL(string_t, mpi_thread_type,
                                     "TIMEMORY_MPI_THREAD_TYPE",
                                     "MPI_Init_thread mode: 'single', 'serialized', "
                                     "'funneled', or 'multiple' (see also: "
                                     "TIMEMORY_MPI_INIT and TIMEMORY_MPI_THREAD)",
                                     mpi::use_mpi_thread_type())

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, upcxx_init, "TIMEMORY_UPCXX_INIT",
        "Enable/disable timemory calling upcxx::init() during certain "
        "timemory_init(...) invocations",
        false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, upcxx_finalize, "TIMEMORY_UPCXX_FINALIZE",
        "Enable/disable timemory calling upcxx::finalize() during "
        "timemory_finalize()",
        false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, papi_multiplexing, "TIMEMORY_PAPI_MULTIPLEXING",
                                  "Enable multiplexing when using PAPI", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, papi_fail_on_error, "TIMEMORY_PAPI_FAIL_ON_ERROR",
                                  "Configure PAPI errors to trigger a runtime error",
                                  false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, papi_quiet, "TIMEMORY_PAPI_QUIET",
        "Configure suppression of reporting PAPI errors/warnings", false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, papi_events, "TIMEMORY_PAPI_EVENTS",
        "PAPI presets and events to collect (see also: papi_avail)", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, papi_attach, "TIMEMORY_PAPI_ATTACH",
        "Configure PAPI to attach to another process (see also: TIMEMORY_TARGET_PID)",
        false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int, papi_overflow, "TIMEMORY_PAPI_OVERFLOW",
        "Value at which PAPI hw counters trigger an overflow callback", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, cuda_event_batch_size, "TIMEMORY_CUDA_EVENT_BATCH_SIZE",
        "Batch size for create cudaEvent_t in cuda_event components", 5)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, nvtx_marker_device_sync, "TIMEMORY_NVTX_MARKER_DEVICE_SYNC",
        "Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)", true)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int32_t, cupti_activity_level, "TIMEMORY_CUPTI_ACTIVITY_LEVEL",
        "Default group of kinds tracked via CUpti Activity API", 1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(string_t, cupti_activity_kinds,
                                  "TIMEMORY_CUPTI_ACTIVITY_KINDS",
                                  "Specific cupti activity kinds to track", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, cupti_events, "TIMEMORY_CUPTI_EVENTS",
        "Hardware counter event types to collect on NVIDIA GPUs", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, cupti_metrics, "TIMEMORY_CUPTI_METRICS",
        "Hardware counter metric types to collect on NVIDIA GPUs", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(int, cupti_device, "TIMEMORY_CUPTI_DEVICE",
                                  "Target device for CUPTI hw counter collection", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, roofline_mode, "TIMEMORY_ROOFLINE_MODE",
        "Configure the roofline collection mode. Options: 'op' 'ai'.", "op")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, cpu_roofline_mode, "TIMEMORY_ROOFLINE_MODE_CPU",
        "Configure the roofline collection mode for CPU specifically. Options: 'op' "
        "'ai'",
        get<string_t>("roofline_mode"))

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, gpu_roofline_mode, "TIMEMORY_ROOFLINE_MODE_GPU",
        "Configure the roofline collection mode for GPU specifically. Options: 'op' "
        "'ai'.",
        get<string_t>("roofline_mode"))

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, cpu_roofline_events, "TIMEMORY_ROOFLINE_EVENTS_CPU",
        "Configure custom hw counters to add to the cpu roofline", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, gpu_roofline_events, "TIMEMORY_ROOFLINE_EVENTS_GPU",
        "Configure custom hw counters to add to the gpu roofline", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, roofline_type_labels,
                                  "TIMEMORY_ROOFLINE_TYPE_LABELS",
                                  "Configure roofline labels/descriptions/output-files "
                                  "encode the list of data types",
                                  false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels_cpu, "TIMEMORY_ROOFLINE_TYPE_LABELS_CPU",
        "Configure labels, etc. for the roofline components "
        "for CPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)",
        static_cast<tsettings<bool>*>(m_data["TIMEMORY_ROOFLINE_TYPE_LABELS"].get())
            ->get())

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels_gpu, "TIMEMORY_ROOFLINE_TYPE_LABELS_GPU",
        "Configure labels, etc. for the roofline components "
        "for GPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)",
        static_cast<tsettings<bool>*>(m_data["TIMEMORY_ROOFLINE_TYPE_LABELS"].get())
            ->get())

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, instruction_roofline,
                                  "TIMEMORY_INSTRUCTION_ROOFLINE",
                                  "Configure the roofline to include the hw counters "
                                  "required for generating an instruction roofline",
                                  false)

    TIMEMORY_SETTINGS_MEMBER_IMPL(uint64_t, ert_num_threads, "TIMEMORY_ERT_NUM_THREADS",
                                  "Number of threads to use when running ERT", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(uint64_t, ert_num_threads_cpu,
                                  "TIMEMORY_ERT_NUM_THREADS_CPU",
                                  "Number of threads to use when running ERT on CPU",
                                  std::thread::hardware_concurrency())

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_threads_gpu, "TIMEMORY_ERT_NUM_THREADS_GPU",
        "Number of threads which launch kernels when running ERT on the GPU", 1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_streams, "TIMEMORY_ERT_NUM_STREAMS",
        "Number of streams to use when launching kernels in ERT on the GPU", 1)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_grid_size, "TIMEMORY_ERT_GRID_SIZE",
        "Configure the grid size (number of blocks) for ERT on GPU (0 == auto-compute)",
        0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_block_size, "TIMEMORY_ERT_BLOCK_SIZE",
        "Configure the block size (number of threads per block) for ERT on GPU", 1024)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_alignment, "TIMEMORY_ERT_ALIGNMENT",
        "Configure the alignment (in bits) when running ERT on CPU (0 == 8 * sizeof(T))",
        0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size, "TIMEMORY_ERT_MIN_WORKING_SIZE",
        "Configure the minimum working size when running ERT (0 == device specific)", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size_cpu, "TIMEMORY_ERT_MIN_WORKING_SIZE_CPU",
        "Configure the minimum working size when running ERT on CPU", 64)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size_gpu, "TIMEMORY_ERT_MIN_WORKING_SIZE_GPU",
        "Configure the minimum working size when running ERT on GPU", 10 * 1000 * 1000)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size, "TIMEMORY_ERT_MAX_DATA_SIZE",
        "Configure the max data size when running ERT on CPU", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size_cpu, "TIMEMORY_ERT_MAX_DATA_SIZE_CPU",
        "Configure the max data size when running ERT on CPU", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size_gpu, "TIMEMORY_ERT_MAX_DATA_SIZE_GPU",
        "Configure the max data size when running ERT on GPU", 500 * 1000 * 1000)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, ert_skip_ops, "TIMEMORY_ERT_SKIP_OPS",
        "Skip these number of ops (i.e. ERT_FLOPS) when were set at compile time", "")

    TIMEMORY_SETTINGS_MEMBER_IMPL(string_t, craypat_categories, "TIMEMORY_CRAYPAT",
                                  "Configure the CrayPAT categories to collect",
                                  get_env<std::string>("PAT_RT_PERFCTR", ""))

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, enable_signal_handler,
                                  "TIMEMORY_ENABLE_SIGNAL_HANDLER",
                                  "Enable signals in timemory_init", false)

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, allow_signal_handler, "TIMEMORY_ALLOW_SIGNAL_HANDLER",
        "Allow signal handling to be activated", signal_settings::allow())

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, enable_all_signals, "TIMEMORY_ENABLE_ALL_SIGNALS",
        "Enable catching all signals", signal_settings::enable_all())

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, disable_all_signals, "TIMEMORY_DISABLE_ALL_SIGNALS",
        "Disable catching any signals", signal_settings::disable_all())

    TIMEMORY_SETTINGS_MEMBER_IMPL(int32_t, node_count, "TIMEMORY_NODE_COUNT",
                                  "Total number of nodes used in application", 0)

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, destructor_report, "TIMEMORY_DESTRUCTOR_REPORT",
        "Configure default setting for auto_{list,tuple,hybrid} to write to stdout during"
        " destruction of the bundle",
        false)

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(string_t, python_exe, "TIMEMORY_PYTHON_EXE",
                                      "Configure the python executable to use",
                                      TIMEMORY_PYTHON_PLOTTER,
                                      strvector_t({ "--timemory-python-exe" }))

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, suppress_parsing, "TIMEMORY_SUPPRESS_PARSING",
                                  "Disable parsing environment", false)
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::read(const string_t& inp)
{
    std::ifstream ifs(inp.c_str());
    if(!ifs)
        throw std::runtime_error(
            TIMEMORY_JOIN(" ", "Error reading configuration file:", inp));
    if(inp.find(".json") != std::string::npos)
    {
        using policy_type = policy::input_archive<cereal::JSONInputArchive, TIMEMORY_API>;
        auto ia           = policy_type::get(ifs);
        ia->setNextName("timemory");
        ia->startNode();
        {
            try
            {
                ia->setNextName("metadata");
                ia->startNode();
                // settings
                (*ia)(cereal::make_nvp("settings", *this));
                ia->finishNode();
            } catch(...)
            {
                // settings
                (*ia)(cereal::make_nvp("settings", *this));
            }
        }
        ia->finishNode();
        return true;
    }
#if defined(TIMEMORY_USE_XML_ARCHIVE) || defined(TIMEMORY_XML_SUPPORT)
    else if(inp.find(".xml") != std::string::npos)
    {
        using policy_type = policy::input_archive<cereal::XMLInputArchive, TIMEMORY_API>;
        auto ia           = policy_type::get(ifs);
        ia->setNextName("timemory");
        ia->startNode();
        {
            try
            {
                ia->setNextName("metadata");
                ia->startNode();
                // settings
                (*ia)(cereal::make_nvp("settings", *this));
                ia->finishNode();
            } catch(...)
            {
                // settings
                (*ia)(cereal::make_nvp("settings", *this));
            }
        }
        ia->finishNode();
        return true;
    }
#endif
    else
    {
        std::string line = "";

        auto is_comment = [](const std::string& s) {
            if(s.empty())
                return true;
            for(const auto& itr : s)
            {
                if(std::isprint(itr))
                    return (itr == '#') ? true : false;
            }
            // if there were no printable characters, treat as comment
            return true;
        };

        int expected = 0;
        int valid    = 0;
        while(ifs)
        {
            std::getline(ifs, line);
            if(!ifs)
                continue;
            if(is_comment(line))
                continue;
            ++expected;
            // tokenize the string
            auto delim = tim::delimit(line, "\n=,; ");
            if(delim.size() > 0)
            {
                string_t key = delim.front();
                string_t val = "";
                // combine into another string separated by commas
                for(size_t i = 1; i < delim.size(); ++i)
                    val += "," + delim.at(i);
                // if there was any fields, remove the leading comma
                if(val.length() > 0)
                    val = val.substr(1);
                for(auto itr : *this)
                {
                    if(itr.second->matches(key))
                    {
                        ++valid;
                        itr.second->parse(val);
                    }
                }
            }
        }
        return (expected == valid);
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

#include "timemory/settings/extern.hpp"
