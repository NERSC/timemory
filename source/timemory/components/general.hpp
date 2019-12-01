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
//

#pragma once

#include "timemory/backends/gperf.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>

//======================================================================================//

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<trip_count>;

#endif

//--------------------------------------------------------------------------------------//
//
//          General Components with no specific category
//
//--------------------------------------------------------------------------------------//
// returns the trip count
//
struct trip_count : public base<trip_count>
{
    using value_type = int64_t;
    using this_type  = trip_count;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "trip_count"; }
    static std::string description() { return "trip counts"; }
    static value_type  record() { return 1; }

    value_type get() const { return accum; }
    value_type get_display() const { return get(); }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        accum += value;
        set_stopped();
    }
};

//--------------------------------------------------------------------------------------//
// start/stop gperftools cpu profiler
//
struct gperf_cpu_profiler
: public base<gperf_cpu_profiler, void, policy::thread_init, policy::global_finalize>
{
    using value_type = void;
    using this_type  = gperf_cpu_profiler;
    using base_type =
        base<this_type, value_type, policy::thread_init, policy::global_finalize>;

    static std::string label() { return "gperf_cpu_profiler"; }
    static std::string description() { return "gperftools cpu profiler"; }
    static value_type  record() {}

    static void invoke_thread_init(storage_type*) { gperf::cpu::register_thread(); }

    static void invoke_global_finalize(storage_type*)
    {
        if(gperf::cpu::is_running())
        {
            gperf::cpu::profiler_flush();
            gperf::cpu::profiler_stop();
        }
    }

    void start()
    {
        set_started();
        if(!gperf::cpu::is_running())
        {
            index                 = this_type::get_index()++;
            const auto& _mpi_info = get_mpi_info();
            bool        _mpi_init = std::get<0>(_mpi_info);
            int32_t     _mpi_rank = std::get<1>(_mpi_info);
            auto        fname     = settings::compose_output_filename(
                label() + "_" + std::to_string(index), ".dat", _mpi_init, _mpi_rank);
            auto ret = gperf::cpu::profiler_start(fname);
            if(ret == 0)
                fprintf(stderr, "[gperf_cpu_profiler]> Error starting %s...",
                        fname.c_str());
        }
    }

    void stop()
    {
        if(index >= 0)
        {
            gperf::cpu::profiler_flush();
            gperf::cpu::profiler_stop();
        }
        set_stopped();
    }

protected:
    int32_t index = -1;  // if this is >= zero, then we flush and stop

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance;
        return _instance;
    }

    using mpi_info_t = std::tuple<bool, int32_t, int32_t>;

    static const mpi_info_t& get_mpi_info()
    {
        static mpi_info_t _info{ mpi::is_initialized(), mpi::rank(), mpi::size() };
        return _info;
    }
};

//--------------------------------------------------------------------------------------//
// start/stop gperftools cpu profiler
//
struct gperf_heap_profiler
: public base<gperf_heap_profiler, void, policy::global_finalize>
{
    using value_type = void;
    using this_type  = gperf_heap_profiler;
    using base_type  = base<this_type, value_type, policy::global_finalize>;

    static std::string label() { return "gperf_heap_profiler"; }
    static std::string description() { return "gperftools heap profiler"; }
    static value_type  record() {}

    static void invoke_global_finalize(storage_type*)
    {
        if(gperf::heap::is_running())
        {
            gperf::heap::profiler_flush("global_finalize");
            gperf::heap::profiler_stop();
        }
    }

    void start()
    {
        set_started();
        if(!gperf::heap::is_running())
        {
            index      = this_type::get_index()++;
            auto fname = settings::compose_output_filename(label(), ".dat");
            auto ret   = gperf::heap::profiler_start(fname);
            if(ret > 0)
                fprintf(stderr, "[gperf_heap_profiler]> Error starting %s...",
                        prefix.c_str());
        }
    }

    void stop()
    {
        if(index >= 0)
        {
            gperf::heap::profiler_flush(prefix);
            gperf::heap::profiler_stop();
        }
        set_stopped();
    }

    void set_prefix(const std::string& _prefix) { prefix = _prefix; }

protected:
    std::string prefix;
    int32_t     index = -1;  // if this is >= zero, then we flush and stop

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <size_t _Idx = 0>
struct user_bundle : public base<user_bundle<_Idx>, void>
{
    using value_type = void;
    using this_type  = user_bundle<_Idx>;
    using base_type  = base<user_bundle, value_type>;

    using start_func_t = std::function<void*(const std::string&)>;
    using stop_func_t  = std::function<void(void*)>;

    static std::string label() { return "user_bundle"; }
    static std::string description() { return "user-defined bundle of tools"; }
    static value_type  record() {}

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename _Toolset>
    static void configure()
    {
        get_start() = [&](const std::string& _prefix) {
            _Toolset* _result = new _Toolset(_prefix);
            _result->start();
            return (void*) _result;
        };

        get_stop() = [&](void* v_result) {
            _Toolset* _result = static_cast<_Toolset*>(v_result);
            _result->stop();
            delete _result;
        };
    }

    //----------------------------------------------------------------------------------//
    //  Capture the statically-defined start/stop so these can be changed without
    //  affecting this instance
    //
    user_bundle()
    : m_prefix("")
    , m_bundle(nullptr)
    , m_start(get_start())
    , m_stop(get_stop())
    {}

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools with an initializer
    //
    template <typename _Toolset, typename _InitFunc>
    static void configure(_InitFunc&& _init)
    {
        get_start() = [&](const std::string& _prefix) {
            _Toolset* _result = new _Toolset(_prefix);
            std::forward<_InitFunc>(_init)(*_result);
            _result->start();
            return (void*) _result;
        };

        get_stop() = [&](void* v_result) {
            _Toolset* _result = static_cast<_Toolset*>(v_result);
            _result->stop();
            delete _result;
        };
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly configure the start function
    //
    static start_func_t& get_start()
    {
        static start_func_t _instance = [](const std::string&) {
            return (void*) nullptr;
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly configure the stop function
    //
    static stop_func_t& get_stop()
    {
        static stop_func_t _instance = [](void*) {};
        return _instance;
    }

    void start() { m_bundle = (void*) m_start(m_prefix); }
    void stop() { m_stop(m_bundle); }
    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

protected:
    std::string  m_prefix;
    void*        m_bundle;
    start_func_t m_start;
    stop_func_t  m_stop;
};

//--------------------------------------------------------------------------------------//

}  // namespace component

//--------------------------------------------------------------------------------------//
}  // namespace tim
