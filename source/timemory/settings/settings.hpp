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
#include "timemory/macros.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/tsettings.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/settings/vsettings.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

#include <ctime>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(TIMEMORY_UNIX)
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
    using value_type     = std::shared_ptr<vsettings>;
    using data_type      = std::unordered_map<string_view_t, value_type>;
    using iterator       = typename data_type::iterator;
    using const_iterator = typename data_type::const_iterator;
    using pointer_t      = std::shared_ptr<settings>;

    template <typename Tp, typename Vp>
    using tsetting_pointer_t = std::shared_ptr<tsettings<Tp, Vp>>;

    template <typename Tag = TIMEMORY_API>
    static std::time_t* get_launch_time(Tag = {});
    template <typename Tag>
    static TIMEMORY_HOT pointer_t shared_instance() TIMEMORY_VISIBILITY("default");
    template <typename Tag>
    static TIMEMORY_HOT settings* instance() TIMEMORY_VISIBILITY("default");
    static TIMEMORY_HOT pointer_t shared_instance() TIMEMORY_VISIBILITY("default");
    static TIMEMORY_HOT settings* instance() TIMEMORY_VISIBILITY("default");

    settings();
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

    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, config_file)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, suppress_parsing)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, suppress_config)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, enabled)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, auto_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, cout_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, file_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, text_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, json_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, tree_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, dart_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, time_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, plot_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, diff_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, flamegraph_output)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, ctest_notes)
    TIMEMORY_SETTINGS_MEMBER_DECL(int, verbose)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, debug)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, banner)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, collapse_threads)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, collapse_processes)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint16_t, max_depth)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, time_format)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, precision)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, width)
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, max_width)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, scientific)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, timing_precision)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, timing_width)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, timing_units)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, timing_scientific)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, memory_precision)
    TIMEMORY_SETTINGS_MEMBER_DECL(int16_t, memory_width)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, memory_units)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, memory_scientific)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, output_path)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, output_prefix)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, input_path)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, input_prefix)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, input_extensions)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, dart_type)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, dart_count)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, dart_label)
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, max_thread_bookmarks)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, cpu_affinity)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, stack_clearing)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, add_secondary)
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, throttle_count)
    TIMEMORY_SETTINGS_MEMBER_DECL(size_t, throttle_value)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, global_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, tuple_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, list_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, ompt_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, mpip_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, ncclp_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, trace_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, profiler_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, kokkos_components)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, components)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_init)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_finalize)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, mpi_thread)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, mpi_thread_type)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, upcxx_init)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, upcxx_finalize)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_threading)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_multiplexing)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_fail_on_error)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_quiet)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, papi_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, papi_attach)
    TIMEMORY_SETTINGS_MEMBER_DECL(int, papi_overflow)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, cuda_event_batch_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, nvtx_marker_device_sync)
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, cupti_activity_level)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cupti_activity_kinds)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cupti_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cupti_metrics)
    TIMEMORY_SETTINGS_MEMBER_DECL(int, cupti_device)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, roofline_mode)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cpu_roofline_mode)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, gpu_roofline_mode)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, cpu_roofline_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, gpu_roofline_events)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, roofline_type_labels_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, instruction_roofline)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_threads_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_num_streams)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_grid_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_block_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_alignment)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_min_working_size_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size_cpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(uint64_t, ert_max_data_size_gpu)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, ert_skip_ops)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, craypat_categories)
    TIMEMORY_SETTINGS_MEMBER_DECL(int32_t, node_count)
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, destructor_report)
    TIMEMORY_SETTINGS_MEMBER_DECL(string_t, python_exe)
    // stream
    TIMEMORY_SETTINGS_MEMBER_DECL(int64_t, separator_frequency)
    // signals
    TIMEMORY_SETTINGS_MEMBER_DECL(bool, enable_signal_handler)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, allow_signal_handler)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, enable_all_signals)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, disable_all_signals)
    // miscellaneous ref
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, flat_profile)
    TIMEMORY_SETTINGS_REFERENCE_DECL(bool, timeline_profile)
    TIMEMORY_SETTINGS_REFERENCE_DECL(process::id_t, target_pid)

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
                                            bool    _mpi_init = false,
                                            int32_t _mpi_rank = -1, bool fake = false,
                                            std::string _explicit = "")
        TIMEMORY_VISIBILITY("default");
    static string_t compose_input_filename(const string_t& _tag, string_t _ext,
                                           bool _mpi_init = false, int32_t _mpi_rank = -1,
                                           std::string _explicit = "")
        TIMEMORY_VISIBILITY("default");

    static void parse(settings* = instance<TIMEMORY_API>())
        TIMEMORY_VISIBILITY("default");

    static void parse(std::shared_ptr<settings>) TIMEMORY_VISIBILITY("default");

public:
    template <typename Archive>
    void load(Archive& ar, unsigned int);

    template <typename Archive>
    void save(Archive& ar, unsigned int) const;

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

public:
    TIMEMORY_NODISCARD auto ordering() const { return m_order; }
    iterator                begin() { return m_data.begin(); }
    iterator                end() { return m_data.end(); }
    TIMEMORY_NODISCARD const_iterator begin() const { return m_data.cbegin(); }
    TIMEMORY_NODISCARD const_iterator end() const { return m_data.cend(); }
    TIMEMORY_NODISCARD const_iterator cbegin() const { return m_data.cbegin(); }
    TIMEMORY_NODISCARD const_iterator cend() const { return m_data.cend(); }

    template <typename Sp = string_t>
    auto find(Sp&& _key, bool _exact = true);

    template <typename Tp, typename Sp = string_t>
    Tp get(Sp&& _key, bool _exact = true);

    template <typename Tp, typename Sp = string_t>
    bool get(Sp&& _key, Tp& _val, bool _exact);

    template <typename Tp, typename Sp = string_t>
    bool set(Sp&& _key, Tp&& _val, bool _exact = true);

    /// \fn bool update(const std::string& key, const std::string& val, bool exact)
    /// \param key Identifier for the setting. Either name, env-name, or command-line opt
    /// \param val Update value
    /// \param exact If true, match only options
    ///
    /// \brief Update a setting via a string. Returns whether a matching setting
    /// for the identifier was found (NOT whether the value was actually updated)
    bool update(const std::string& _key, const std::string& _val, bool _exact = false);

    /// \tparam Tp Data-type of the setting
    /// \tparam Vp Value-type of the setting (Tp or Tp&)
    /// \tparam Sp String-type
    template <typename Tp, typename Vp, typename Sp, typename... Args>
    auto insert(Sp&& _env, const std::string& _name, const std::string& _desc, Vp _init,
                Args&&... _args);

    /// \tparam Tp Data-type of the setting
    /// \tparam Vp Value-type of the setting (Tp or Tp&)
    /// \tparam Sp String-type
    template <typename Tp, typename Vp, typename Sp = string_t>
    auto insert(tsetting_pointer_t<Tp, Vp> _ptr, Sp&& _env = {});

    /// \tparam API Tagged type
    ///
    /// \brief Make a copy of the current settings and return a new instance whose values
    /// can be modified, used, and then discarded. The values modified do not change
    /// any settings accessed through static methods or the non-templated instance method.
    /// E.g. `tim::settings::enabled()` will not be affected by changes to settings
    /// instance returned by this method.
    template <typename Tag>
    static pointer_t push();

    /// \tparam API Tagged type
    ///
    /// \brief Restore the settings from a previous push operations.
    template <typename Tag>
    static pointer_t pop();

protected:
    template <typename Archive, typename Tp>
    TIMEMORY_NODISCARD auto get_serialize_pair() const  // NOLINT
    {
        using serialize_func_t = std::function<void(Archive&, value_type)>;
        using serialize_pair_t = std::pair<std::type_index, serialize_func_t>;

        auto _func = [](Archive& _ar, value_type _val) {
            using Up = tsettings<Tp>;
            if(!_val)
                _val = std::make_shared<Up>();
            _ar(cereal::make_nvp(_val->get_env_name(), *static_cast<Up*>(_val.get())));
        };
        return serialize_pair_t{ std::type_index(typeid(Tp)), _func };
    }

    template <typename Archive, typename... Tail>
    TIMEMORY_NODISCARD auto get_serialize_map(tim::type_list<Tail...>) const  // NOLINT
    {
        using serialize_func_t = std::function<void(Archive&, value_type)>;
        using serialize_map_t  = std::map<std::type_index, serialize_func_t>;

        serialize_map_t _val{};
        TIMEMORY_FOLD_EXPRESSION(_val.insert(get_serialize_pair<Archive, Tail>()));
        return _val;
    }

private:
    using settings_stack_t = std::stack<pointer_t>;

    template <typename Tag>
    static TIMEMORY_HOT pointer_t& private_shared_instance(
        enable_if_t<std::is_same<Tag, TIMEMORY_API>::value, int> = 0);

    template <typename Tag>
    static TIMEMORY_HOT pointer_t& private_shared_instance(
        enable_if_t<!std::is_same<Tag, TIMEMORY_API>::value, long> = 0);

    template <typename Tag>
    static settings_stack_t& get_stack()
    {
        static auto _instance = settings_stack_t{};
        return _instance;
    }

private:
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

    static auto& indent_width_map()
    {
        static std::map<size_t, std::map<std::type_index, int64_t>> _instance{};
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::time_t* settings::get_launch_time(Tag)
{
    // statically store the launch time, intentional memory leak
    static std::time_t* _time = new std::time_t{ std::time(nullptr) };
    return _time;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>&
settings::private_shared_instance(
    enable_if_t<std::is_same<Tag, TIMEMORY_API>::value, int>)
{
    // this is the original
    static std::shared_ptr<settings> _instance = std::make_shared<settings>();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>&
settings::private_shared_instance(
    enable_if_t<!std::is_same<Tag, TIMEMORY_API>::value, long>)
{
    // make a copy of the original
    static std::shared_ptr<settings> _instance =
        std::make_shared<settings>(*private_shared_instance<TIMEMORY_API>());
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>
settings::push()
{
    // ensure the non-template methods have their own static copies
    static auto* _discard_ptr  = instance();
    static auto* _discard_sptr = instance<TIMEMORY_API>();
    consume_parameters(_discard_ptr, _discard_sptr);

    auto _old = shared_instance<Tag>();
    get_stack<Tag>().push(_old);
    private_shared_instance<Tag>() = std::make_shared<settings>(*_old);
    return private_shared_instance<Tag>();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>
settings::pop()
{
    auto& _stack = get_stack<Tag>();
    if(_stack.empty())
    {
        PRINT_HERE("%s", "Ignoring settings::pop() on empty stack");
        return shared_instance<Tag>();
    }

    auto _top                      = _stack.top();
    private_shared_instance<Tag>() = _top;
    _stack.pop();
    return _top;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
std::shared_ptr<settings>
settings::shared_instance()
{
    static std::shared_ptr<settings>& _instance = private_shared_instance<Tag>();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag>
settings*
settings::instance()
{
    return shared_instance<Tag>().get();
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx>
int64_t
settings::indent_width(int64_t _w)
{
    auto        _tidx = std::type_index{ typeid(TIMEMORY_API) };
    auto_lock_t _lk{ type_mutex<settings, TIMEMORY_API>() };
    auto&       _itr = indent_width_map()[Idx][_tidx];
    return (_itr = std::max<int64_t>(_itr, _w));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, size_t Idx>
int64_t
settings::indent_width(int64_t _w)
{
    auto        _tidx = std::type_index{ typeid(Tp) };
    auto_lock_t _lk{ type_mutex<settings, TIMEMORY_API>() };
    auto&       _itr = indent_width_map()[Idx][_tidx];
    return (_itr = std::max<int64_t>(_itr, _w));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::load(Archive& ar, unsigned int)
{
#if !defined(TIMEMORY_DISABLE_SETTINGS_SERIALIZATION)
    using map_type = std::map<std::string, std::shared_ptr<vsettings>>;
    map_type _data;
    for(const auto& itr : m_data)
        _data.insert({ std::string{ itr.first }, itr.second->clone() });
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
        auto ditr = m_data.find(itr.first);
        if(ditr != m_data.end())
        {
            ditr->second->clone(itr.second);
        }
        else
        {
            m_order.push_back(itr.first);
            m_data.insert({ m_order.back(), itr.second });
        }
    }
#else
    consume_parameters(ar);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::save(Archive& ar, unsigned int) const
{
#if !defined(TIMEMORY_DISABLE_SETTINGS_SERIALIZATION)
    using map_type = std::map<std::string, std::shared_ptr<vsettings>>;
    map_type _data;
    for(const auto& itr : m_data)
        _data.insert({ std::string{ itr.first }, itr.second->clone() });

    auto _map = get_serialize_map<Archive>(data_type_list_t{});
    for(const auto& itr : _data)
    {
        auto mitr = _map.find(itr.second->get_type_index());
        if(mitr != _map.end())
            mitr->second(ar, itr.second);
    }
    ar(cereal::make_nvp("command_line", m_command_line),
       cereal::make_nvp("environment", m_environment));
#else
    consume_parameters(ar);
#endif
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Sp>
inline auto
settings::find(Sp&& _key, bool _exact)
{
    // exact match to map key
    auto itr = m_data.find(std::forward<Sp>(_key));
    if(itr != m_data.end())
        return itr;

    // match against env_name, name, command-line options
    for(auto ditr = begin(); ditr != end(); ++ditr)
    {
        if(ditr->second && ditr->second->matches(std::forward<Sp>(_key), _exact))
            return ditr;
    }

    // not found
    return m_data.end();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Sp>
Tp
settings::get(Sp&& _key, bool _exact)
{
    auto itr = find(std::forward<Sp>(_key), _exact);
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
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Sp>
bool
settings::get(Sp&& _key, Tp& _val, bool _exact)
{
    auto itr = find(std::forward<Sp>(_key), _exact);
    if(itr != m_data.end() && itr->second)
    {
        auto _vptr = itr->second;
        auto _tidx = std::type_index(typeid(Tp));
        auto _vidx = std::type_index(typeid(Tp&));
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _tidx)
            return ((_val = static_cast<tsettings<Tp, Tp>*>(_vptr.get())->get()), true);
        if(_vptr->get_type_index() == _tidx && _vptr->get_value_index() == _vidx)
            return ((_val = static_cast<tsettings<Tp, Tp&>*>(_vptr.get())->get()), true);
    }
    return false;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp, typename Sp>
bool
settings::set(Sp&& _key, Tp&& _val, bool _exact)
{
    auto itr = find(std::forward<Sp>(_key), _exact);
    if(itr != m_data.end() && itr->second)
    {
        using Up   = decay_t<Tp>;
        auto _tidx = std::type_index(typeid(Up));
        auto _vidx = std::type_index(typeid(Up&));
        auto _tobj = dynamic_cast<tsettings<Up>*>(itr->second.get());
        auto _robj = dynamic_cast<tsettings<Up, Up&>*>(itr->second.get());
        if(itr->second->get_type_index() == _tidx &&
           itr->second->get_value_index() == _tidx && _tobj)
        {
            return (_tobj->set(std::forward<Tp>(_val)), true);
        }
        else if(itr->second->get_type_index() == _tidx &&
                itr->second->get_value_index() == _vidx && _robj)
        {
            return (_robj->set(std::forward<Tp>(_val)), true);
        }
        else
        {
            throw std::runtime_error(std::string{ "tim::settings::set(" } +
                                     std::string{ _key } + ", ...) failed");
        }
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
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
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp, typename Sp, typename... Args>
auto
settings::insert(Sp&& _env, const std::string& _name, const std::string& _desc, Vp _init,
                 Args&&... _args)
{
    static_assert(is_one_of<Tp, data_type_list_t>::value,
                  "Error! Data type is not supported. See settings::data_type_list_t");
    static_assert(std::is_same<decay_t<Tp>, decay_t<Vp>>::value,
                  "Error! Initializing value is not the same as the declared type");

    auto _sid = std::string{ std::forward<Sp>(_env) };
    set_env(_sid, _init, 0);
    m_order.push_back(_sid);
    return m_data.insert(
        { string_view_t{ m_order.back() },
          std::make_shared<tsettings<Tp, Vp>>(_init, _name, _sid, _desc,
                                              std::forward<Args>(_args)...) });
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp, typename Sp>
auto
settings::insert(tsetting_pointer_t<Tp, Vp> _ptr, Sp&& _env)
{
    static_assert(is_one_of<Tp, data_type_list_t>::value,
                  "Error! Data type is not supported. See settings::data_type_list_t");
    if(_ptr)
    {
        auto _sid = std::string{ std::forward<Sp>(_env) };
        if(_sid.empty())
            _sid = _ptr->get_env_name();
        if(!_sid.empty())
        {
            set_env(_sid, _ptr->as_string(), 0);
            m_order.push_back(_sid);
            return m_data.insert({ string_view_t{ m_order.back() }, _ptr });
        }
    }

    return std::make_pair(m_data.end(), false);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
