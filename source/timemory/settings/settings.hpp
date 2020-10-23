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
#include "timemory/macros/compiler.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/tsettings.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/settings/vsettings.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

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
//
class manager;
//
//--------------------------------------------------------------------------------------//
//
//                              settings
//
//--------------------------------------------------------------------------------------//
//
struct settings
{
    // this is the list of the current and potentially used data types
    using data_type_list_t =
        tim::type_list<bool, string_t, int16_t, int32_t, int64_t, uint16_t, uint32_t,
                       uint64_t, size_t, float, double>;
    friend class manager;
    using strvector_t    = std::vector<std::string>;
    using strmap_t       = std::map<std::string, std::string>;
    using value_type     = std::shared_ptr<vsettings>;
    using data_type      = std::unordered_map<std::string, value_type>;
    using iterator       = typename data_type::iterator;
    using const_iterator = typename data_type::const_iterator;
    using pointer_t      = std::shared_ptr<settings>;

    template <typename Tp, typename Vp>
    using tsetting_pointer_t = std::shared_ptr<tsettings<Tp, Vp>>;

    template <typename Tag = api::native_tag>
    static pointer_t shared_instance() TIMEMORY_VISIBILITY("default");

    template <typename Tag = api::native_tag>
    static settings* instance() TIMEMORY_VISIBILITY("default");

    settings() { initialize(); }
    ~settings() = default;

    settings(const settings&);
    settings(settings&&) noexcept = default;

    settings& operator=(const settings&);
    settings& operator=(settings&&) noexcept = default;

    void initialize();

    //==================================================================================//
    //
    //                  GENERAL SETTINGS THAT APPLY TO MULTIPLE COMPONENTS
    //
    //==================================================================================//

    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, config_file, "TIMEMORY_CONFIG_FILE")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, suppress_parsing, "TIMEMORY_SUPPRESS_PARSING")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, suppress_config, "TIMEMORY_SUPPRESS_CONFIG")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, enabled, "TIMEMORY_ENABLED")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, auto_output, "TIMEMORY_AUTO_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, cout_output, "TIMEMORY_COUT_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, file_output, "TIMEMORY_FILE_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, text_output, "TIMEMORY_TEXT_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, json_output, "TIMEMORY_JSON_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, dart_output, "TIMEMORY_DART_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, time_output, "TIMEMORY_TIME_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, plot_output, "TIMEMORY_PLOT_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, diff_output, "TIMEMORY_DIFF_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, flamegraph_output, "TIMEMORY_FLAMEGRAPH_OUTPUT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, ctest_notes, "TIMEMORY_CTEST_NOTES")
    TIMEMORY_SETTINGS_MEMBER_DECL(int, verbose, "TIMEMORY_VERBOSE")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, debug, "TIMEMORY_DEBUG")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, banner, "TIMEMORY_BANNER")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, collapse_threads, "TIMEMORY_COLLAPSE_THREADS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, collapse_processes, "TIMEMORY_COLLAPSE_PROCESSES")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint16_t, max_depth, "TIMEMORY_MAX_DEPTH")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, time_format, "TIMEMORY_TIME_FORMAT")
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, precision, "TIMEMORY_PRECISION")
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, width, "TIMEMORY_WIDTH")
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, max_width, "TIMEMORY_MAX_WIDTH")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, scientific, "TIMEMORY_SCIENTIFIC")
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, timing_precision, "TIMEMORY_TIMING_PRECISION")
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, timing_width, "TIMEMORY_TIMING_WIDTH")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, timing_units, "TIMEMORY_TIMING_UNITS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, timing_scientific, "TIMEMORY_TIMING_SCIENTIFIC")
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, memory_precision, "TIMEMORY_MEMORY_PRECISION")
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, memory_width, "TIMEMORY_MEMORY_WIDTH")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, memory_units, "TIMEMORY_MEMORY_UNITS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, memory_scientific, "TIMEMORY_MEMORY_SCIENTIFIC")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, output_path, "TIMEMORY_OUTPUT_PATH")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, output_prefix, "TIMEMORY_OUTPUT_PREFIX")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, input_path, "TIMEMORY_INPUT_PATH")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, input_prefix, "TIMEMORY_INPUT_PREFIX")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, input_extensions, "TIMEMORY_INPUT_EXTENSIONS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, dart_type, "TIMEMORY_DART_TYPE")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, dart_count, "TIMEMORY_DART_COUNT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, dart_label, "TIMEMORY_DART_LABEL")
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, max_thread_bookmarks,
                                  "TIMEMORY_MAX_THREAD_BOOKMARKS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, cpu_affinity, "TIMEMORY_CPU_AFFINITY")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, stack_clearing, "TIMEMORY_STACK_CLEARING")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, add_secondary, "TIMEMORY_ADD_SECONDARY")
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, throttle_count, "TIMEMORY_THROTTLE_COUNT")
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, throttle_value, "TIMEMORY_THROTTLE_VALUE")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, global_components,
                                  "TIMEMORY_GLOBAL_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, tuple_components, "TIMEMORY_TUPLE_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, list_components, "TIMEMORY_LIST_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, ompt_components, "TIMEMORY_OMPT_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, mpip_components, "TIMEMORY_MPIP_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, ncclp_components, "TIMEMORY_NCCLP_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, trace_components, "TIMEMORY_TRACE_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, profiler_components,
                                  "TIMEMORY_PROFILER_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, kokkos_components,
                                  "TIMEMORY_KOKKOS_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, components, "TIMEMORY_COMPONENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_init, "TIMEMORY_MPI_INIT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_finalize, "TIMEMORY_MPI_FINALIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_thread, "TIMEMORY_MPI_THREAD")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, mpi_thread_type, "TIMEMORY_MPI_THREAD_TYPE")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, upcxx_init, "TIMEMORY_UPCXX_INIT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, upcxx_finalize, "TIMEMORY_UPCXX_FINALIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_multiplexing, "TIMEMORY_PAPI_MULTIPLEXING")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_fail_on_error, "TIMEMORY_PAPI_FAIL_ON_ERROR")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_quiet, "TIMEMORY_PAPI_QUIET")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, papi_events, "TIMEMORY_PAPI_EVENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_attach, "TIMEMORY_PAPI_ATTACH")
    TIMEMORY_SETTINGS_MEMBER_DECL(int, papi_overflow, "TIMEMORY_PAPI_OVERFLOW")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, cuda_event_batch_size,
                                  "TIMEMORY_CUDA_EVENT_BATCH_SIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, nvtx_marker_device_sync,
                                  "TIMEMORY_NVTX_MARKER_DEVICE_SYNC")
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, cupti_activity_level,
                                  "TIMEMORY_CUPTI_ACTIVITY_LEVEL")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cupti_activity_kinds,
                                  "TIMEMORY_CUPTI_ACTIVITY_KINDS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cupti_events, "TIMEMORY_CUPTI_EVENTS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cupti_metrics, "TIMEMORY_CUPTI_METRICS")
    TIMEMORY_SETTINGS_MEMBER_DECL(int, cupti_device, "TIMEMORY_CUPTI_DEVICE")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, roofline_mode, "TIMEMORY_ROOFLINE_MODE")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cpu_roofline_mode,
                                  "TIMEMORY_ROOFLINE_MODE_CPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, gpu_roofline_mode,
                                  "TIMEMORY_ROOFLINE_MODE_GPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cpu_roofline_events,
                                  "TIMEMORY_ROOFLINE_EVENTS_CPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, gpu_roofline_events,
                                  "TIMEMORY_ROOFLINE_EVENTS_GPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels,
                                  "TIMEMORY_ROOFLINE_TYPE_LABELS")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels_cpu,
                                  "TIMEMORY_ROOFLINE_TYPE_LABELS_CPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels_gpu,
                                  "TIMEMORY_ROOFLINE_TYPE_LABELS_GPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, instruction_roofline,
                                  "TIMEMORY_INSTRUCTION_ROOFLINE")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads, "TIMEMORY_ERT_NUM_THREADS")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads_cpu,
                                  "TIMEMORY_ERT_NUM_THREADS_CPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads_gpu,
                                  "TIMEMORY_ERT_NUM_THREADS_GPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_streams, "TIMEMORY_ERT_NUM_STREAMS")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_grid_size, "TIMEMORY_ERT_GRID_SIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_block_size, "TIMEMORY_ERT_BLOCK_SIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_alignment, "TIMEMORY_ERT_ALIGNMENT")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size,
                                  "TIMEMORY_ERT_MIN_WORKING_SIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size_cpu,
                                  "TIMEMORY_ERT_MIN_WORKING_SIZE_CPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size_gpu,
                                  "TIMEMORY_ERT_MIN_WORKING_SIZE_GPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size,
                                  "TIMEMORY_ERT_MAX_DATA_SIZE")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size_cpu,
                                  "TIMEMORY_ERT_MAX_DATA_SIZE_CPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size_gpu,
                                  "TIMEMORY_ERT_MAX_DATA_SIZE_GPU")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, ert_skip_ops, "TIMEMORY_ERT_SKIP_OPS")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, craypat_categories, "TIMEMORY_CRAYPAT")
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, node_count, "TIMEMORY_NODE_COUNT")
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, destructor_report, "TIMEMORY_DESTRUCTOR_REPORT")
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, python_exe, "TIMEMORY_PYTHON_EXE")
    // stream
    TIMEMORY_SETTINGS_MEMBER_DECL(int64_t, separator_frequency, "TIMEMORY_SEPARATOR_FREQ")
    // signals
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, enable_signal_handler,
                                  "TIMEMORY_ENABLE_SIGNAL_HANDLER")
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, allow_signal_handler,
                                     "TIMEMORY_ALLOW_SIGNAL_HANDLER")
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, enable_all_signals,
                                     "TIMEMORY_ENABLE_ALL_SIGNALS")
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, disable_all_signals,
                                     "TIMEMORY_DISABLE_ALL_SIGNALS")
    // miscellaneous ref
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, flat_profile, "TIMEMORY_FLAT_PROFILE")
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, timeline_profile, "TIMEMORY_TIMELINE_PROFILE")
    TIMEMORY_SETTINGS_REFERENCE_DECL(process::id_t, target_pid, "TIMEMORY_TARGET_PID")

    static strvector_t& command_line() TIMEMORY_VISIBILITY("default");
    static strvector_t& environment() TIMEMORY_VISIBILITY("default");
    strvector_t&        get_command_line() { return m_command_line; }
    strvector_t&        get_environment() { return m_environment; }

public:
    static strvector_t get_global_environment() TIMEMORY_VISIBILITY("default");
    static string_t    tolower(string_t str) TIMEMORY_VISIBILITY("default");
    static string_t    toupper(string_t str) TIMEMORY_VISIBILITY("default");
    static string_t    get_global_input_prefix() TIMEMORY_VISIBILITY("default");
    static string_t    get_global_output_prefix(bool fake = false)
        TIMEMORY_VISIBILITY("default");
    static void store_command_line(int argc, char** argv) TIMEMORY_VISIBILITY("default");
    static string_t compose_output_filename(const string_t& _tag, string_t _ext,
                                            bool          _mpi_init = false,
                                            const int32_t _mpi_rank = -1,
                                            bool fake = false, std::string _explicit = "")
        TIMEMORY_VISIBILITY("default");
    static string_t compose_input_filename(const string_t& _tag, string_t _ext,
                                           bool          _mpi_init = false,
                                           const int32_t _mpi_rank = -1,
                                           std::string   _explicit = "")
        TIMEMORY_VISIBILITY("default");

    static void parse(settings* = instance<api::native_tag>())
        TIMEMORY_VISIBILITY("default");

    static void parse(std::shared_ptr<settings>) TIMEMORY_VISIBILITY("default");

public:
    template <typename Archive>
    void deprecated_serialize(Archive& ar, const unsigned int);

    template <typename Archive>
    void load(Archive& ar, const unsigned int);

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const;

    template <typename Archive>
    static void serialize_settings(Archive&);

    template <typename Archive>
    static void serialize_settings(Archive&, settings&);

    /// read a configuration file
    bool read(const string_t&);
    bool read(std::istream&, string_t = "");

public:
    template <size_t Idx = 0>
    static int64_t indent_width(int64_t _w = settings::width())
        TIMEMORY_VISIBILITY("default");

    template <typename Tp, size_t Idx = 0>
    static int64_t indent_width(int64_t _w = indent_width<Idx>())
        TIMEMORY_VISIBILITY("default");

    template <typename Tp>
    static size_t data_width(int64_t _idx, int64_t _w) TIMEMORY_VISIBILITY("default");

public:
    auto           ordering() const { return m_order; }
    auto           find(const std::string& _key, bool _exact = true);
    iterator       begin() { return m_data.begin(); }
    iterator       end() { return m_data.end(); }
    const_iterator begin() const { return m_data.cbegin(); }
    const_iterator end() const { return m_data.cend(); }
    const_iterator cbegin() const { return m_data.cbegin(); }
    const_iterator cend() const { return m_data.cend(); }

    template <typename Tp>
    Tp get(const std::string& _key, bool _exact = true);

    template <typename Tp>
    bool set(const std::string& _key, Tp&& _val, bool _exact = true);

    /// \fn bool update(const std::string& key, const std::string& val, bool exact)
    /// \param key Identifier for the setting. Either name, env-name, or command-line opt
    /// \param val Update value
    /// \param exact If true, match only options
    ///
    /// \brief Update a setting via a string. Returns whether a matching setting
    /// for the identifier was found (NOT whether the value was actually updated)
    bool update(const std::string& _key, const std::string& _val, bool _exact = false);

    template <typename Tp, typename Vp = Tp, typename... Args>
    auto insert(const std::string& _env, const std::string& _name,
                const std::string& _desc, Vp _init, Args&&... _args);

    template <typename Tp, typename Vp>
    auto insert(tsetting_pointer_t<Tp, Vp> _ptr, std::string _env = {});

protected:
    template <typename Tp>
    using serialize_func_t = std::function<void(Tp&, value_type)>;
    template <typename Tp>
    using serialize_pair_t = std::pair<std::type_index, serialize_func_t<Tp>>;
    template <typename Tp>
    using serialize_map_t = std::map<std::type_index, serialize_func_t<Tp>>;

    template <typename Archive, typename Tp>
    serialize_pair_t<Archive> get_serialize_pair() const
    {
        auto _func = [](Archive& _ar, value_type _val) {
            using Up = tsettings<Tp>;
            if(!_val)
                _val = std::make_shared<Up>();
            _ar(cereal::make_nvp(_val->get_env_name(), *static_cast<Up*>(_val.get())));
        };
        return serialize_pair_t<Archive>{ std::type_index(typeid(Tp)), _func };
    }

    template <typename Archive, typename... Tail>
    serialize_map_t<Archive> get_serialize_map(tim::type_list<Tail...>) const
    {
        serialize_map_t<Archive> _val;
        TIMEMORY_FOLD_EXPRESSION(_val.insert(get_serialize_pair<Archive, Tail>()));
        return _val;
    }

protected:
    data_type   m_data         = {};
    strvector_t m_order        = {};
    strvector_t m_command_line = {};
    strvector_t m_environment  = get_global_environment();

private:
    void initialize_core() TIMEMORY_VISIBILITY("hidden");
    void initialize_components() TIMEMORY_VISIBILITY("hidden");
    void initialize_io() TIMEMORY_VISIBILITY("hidden");
    void initialize_format() TIMEMORY_VISIBILITY("hidden");
    void initialize_parallel() TIMEMORY_VISIBILITY("hidden");
    void initialize_tpls() TIMEMORY_VISIBILITY("hidden");
    void initialize_roofline() TIMEMORY_VISIBILITY("hidden");
    void initialize_miscellaneous() TIMEMORY_VISIBILITY("hidden");
    void initialize_ert() TIMEMORY_VISIBILITY("hidden");
    void initialize_dart() TIMEMORY_VISIBILITY("hidden");
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
settings::deprecated_serialize(Archive& ar, const unsigned int)
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
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_COLLAPSE_PROCESSES", collapse_processes)
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
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_OUTPUT_PATH", output_path)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_OUTPUT_PREFIX", output_prefix)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INPUT_PATH", input_path)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INPUT_PREFIX", input_prefix)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_INPUT_EXTENSIONS", input_extensions)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_TYPE", dart_type)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_COUNT", dart_count)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_DART_LABEL", dart_label)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_CPU_AFFINITY", cpu_affinity)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MAX_THREAD_BOOKMARKS", max_thread_bookmarks)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TARGET_PID", target_pid)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_STACK_CLEARING", stack_clearing)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_ADD_SECONDARY", add_secondary)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_THROTTLE_COUNT", throttle_count)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_THROTTLE_VALUE", throttle_value)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_GLOBAL_COMPONENTS", global_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TUPLE_COMPONENTS", tuple_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_LIST_COMPONENTS", list_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_OMPT_COMPONENTS", ompt_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPIP_COMPONENTS", mpip_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_TRACE_COMPONENTS", trace_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_PROFILER_COMPONENTS", profiler_components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_COMPONENTS", components)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_INIT", mpi_init)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_FINALIZE", mpi_finalize)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_THREAD", mpi_thread)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_MPI_THREAD_TYPE", mpi_thread_type)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_UPCXX_INIT", upcxx_init)
    TIMEMORY_SETTINGS_TRY_CATCH_NVP("TIMEMORY_UPCXX_FINALIZE", upcxx_finalize)
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
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::load(Archive& ar, const unsigned int version)
{
    using map_type = std::map<std::string, std::shared_ptr<vsettings>>;
    map_type _data;
    for(const auto& itr : m_data)
        _data.insert({ itr.first, itr.second->clone() });
    auto _map = get_serialize_map<Archive>(data_type_list_t{});
    for(const auto& itr : _data)
    {
        auto mitr = _map.find(itr.second->get_type_index());
        if(mitr != _map.end())
            mitr->second(ar, itr.second);
    }
    ar(cereal::make_nvp("command_line", m_command_line),
       cereal::make_nvp("environment", m_environment));
    for(const auto& itr : _data)
    {
        if(m_data.find(itr.first) != m_data.end())
            m_data[itr.first]->clone(itr.second);
        else
        {
            m_data.insert({ itr.first, itr.second });
        }
    }
    consume_parameters(version);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::save(Archive& ar, const unsigned int) const
{
    using map_type = std::map<std::string, std::shared_ptr<vsettings>>;
    map_type _data;
    for(const auto& itr : m_data)
        _data.insert({ itr.first, itr.second->clone() });

    auto _map = get_serialize_map<Archive>(data_type_list_t{});
    for(const auto& itr : _data)
    {
        auto mitr = _map.find(itr.second->get_type_index());
        if(mitr != _map.end())
            mitr->second(ar, itr.second);
    }
    ar(cereal::make_nvp("command_line", m_command_line),
       cereal::make_nvp("environment", m_environment));
}
//
//----------------------------------------------------------------------------------//
//
inline auto
settings::find(const std::string& _key, bool _exact)
{
    // exact match to map key
    auto itr = m_data.find(_key);
    if(itr != m_data.end())
        return itr;

    // match against env_name, name, command-line options
    for(auto ditr = begin(); ditr != end(); ++ditr)
    {
        if(ditr->second && ditr->second->matches(_key, _exact))
            return ditr;
    }

    // not found
    return m_data.end();
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
settings::get(const std::string& _key, bool _exact)
{
    auto itr = find(_key, _exact);
    if(itr != m_data.end() && itr->second)
    {
        auto _vptr = itr->second;
        auto _tidx = std::type_index(typeid(Tp));
        auto _vidx = std::type_index(typeid(Tp&));
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _tidx)
            return static_cast<tsettings<Tp, Tp>*>(_vptr.get())->get();
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _vidx)
            return static_cast<tsettings<Tp, Tp&>*>(_vptr.get())->get();
    }
    return Tp{};
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
bool
settings::set(const std::string& _key, Tp&& _val, bool _exact)
{
    auto itr = find(_key, _exact);
    if(itr != m_data.end() && itr->second)
    {
        using Up   = decay_t<Tp>;
        auto _tidx = std::type_index(typeid(Up));
        auto _vidx = std::type_index(typeid(Up&));
        if(itr->second->get_type_index() == _tidx &&
           itr->second->get_value_index() == _tidx)
            return (static_cast<tsettings<Tp>*>(itr->second.get())->get() =
                        std::forward<Tp>(_val),
                    true);
        if(itr->second->get_type_index() == _tidx &&
           itr->second->get_value_index() == _vidx)
            return (static_cast<tsettings<Tp, Tp&>*>(itr->second.get())->get() =
                        std::forward<Tp>(_val),
                    true);
    }
    return false;
}
//
//----------------------------------------------------------------------------------//
//
inline bool
settings::update(const std::string& _key, const std::string& _val, bool _exact)
{
    auto itr = find(_key, _exact);
    if(itr == m_data.end())
    {
        if(get_verbose() > 0 || get_debug())
            PRINT_HERE("Key: \"%s\" did not match any known setting", _key.c_str());
        return false;
    }

    itr->second->parse(_val);
    return true;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp, typename... Args>
auto
settings::insert(const std::string& _env, const std::string& _name,
                 const std::string& _desc, Vp _init, Args&&... _args)
{
    static_assert(is_one_of<Tp, data_type_list_t>::value,
                  "Error! Data type is not supported. See settings::data_type_list_t");
    set_env(_env, _init, 0);
    m_order.push_back(_env);
    return m_data.insert(
        { _env, std::make_shared<tsettings<Tp, Vp>>(_init, _name, _env, _desc,
                                                    std::forward<Args>(_args)...) });
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp>
auto
settings::insert(tsetting_pointer_t<Tp, Vp> _ptr, std::string _env)
{
    static_assert(is_one_of<Tp, data_type_list_t>::value,
                  "Error! Data type is not supported. See settings::data_type_list_t");
    if(_ptr)
    {
        if(_env.empty())
            _env = _ptr->get_env_name();
        set_env(_env, _ptr->as_string(), 0);
        m_order.push_back(_env);
        return m_data.insert({ _env, _ptr });
    }

    return std::make_pair(m_data.end(), false);
}
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim

// TIMEMORY_SETTINGS_EXTERN_TEMPLATE(api::native_tag)

#if !defined(_TIMEMORY_INTEL)
CEREAL_CLASS_VERSION(tim::settings, 2)
#endif
