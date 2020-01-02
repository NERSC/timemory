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

/** \file components/user_bundle.hpp
 * \headerfile components/user_bundle.hpp "timemory/components/user_bundle.hpp"
 * Defines the user_bundle component which can be used to inject components
 * at runtime. There are very useful for dynamically assembling collections
 * of tools at runtime
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>

//======================================================================================//

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<user_bundle<10101, native_tag>, void>;
extern template struct base<user_bundle<11011, native_tag>, void>;

#endif

//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Tag>
struct user_bundle : public base<user_bundle<_Idx, _Tag>, void>
{
    using value_type   = void;
    using this_type    = user_bundle<_Idx, _Tag>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;

    using start_func_t = std::function<void*(const std::string&)>;
    using stop_func_t  = std::function<void(void*)>;

    using start_func_vec_t = std::vector<start_func_t>;
    using stop_func_vec_t  = std::vector<stop_func_t>;
    using void_vec_t       = std::vector<void*>;
    using typeid_set_t     = std::unordered_set<size_t>;

    using mutex_t = std::mutex;
    using lock_t  = std::unique_lock<mutex_t>;

    static std::string label() { return "user_bundle"; }
    static std::string description() { return "user-defined bundle of tools"; }
    static value_type  record() {}

public:
    //----------------------------------------------------------------------------------//
    //  Capture the statically-defined start/stop so these can be changed without
    //  affecting this instance
    //
    user_bundle(const std::string& _prefix = "")
    : m_prefix(_prefix)
    , m_bundle(void_vec_t{})
    , m_start(get_start())
    , m_stop(get_stop())
    {
        assert(m_start.size() == m_stop.size());
    }

    //----------------------------------------------------------------------------------//
    //  Pass in the start and stop functions
    //
    user_bundle(const start_func_vec_t& _start, const stop_func_vec_t& _stop)
    : m_prefix("")
    , m_bundle(std::max(_start.size(), _stop.size()), nullptr)
    , m_start(_start)
    , m_stop(_stop)
    {
        assert(m_start.size() == m_stop.size());
    }

    //----------------------------------------------------------------------------------//
    //  Pass in the prefix, start, and stop functions
    //
    user_bundle(const std::string& _prefix, const start_func_vec_t& _start,
                const stop_func_vec_t& _stop)
    : m_prefix(_prefix)
    , m_bundle(std::max(_start.size(), _stop.size()), nullptr)
    , m_start(_start)
    , m_stop(_stop)
    {
        assert(m_start.size() == m_stop.size());
    }

public:
    //----------------------------------------------------------------------------------//
    //  Configure the tool with a specific start and stop
    //
    static void configure(start_func_t&& _start, stop_func_t&& _stop)
    {
        internal_init();
        lock_t lk(get_lock());
        get_start().emplace_back(std::forward<start_func_t>(_start));
        get_stop().emplace_back(std::forward<stop_func_t>(_stop));
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename _Toolset, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int>                = 0,
              enable_if_t<(trait::is_available<_Toolset>::value), int> = 0,
              enable_if_t<(_Toolset::is_component), char>              = 0>
    static void configure(bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Toolset>().c_str());

        if(!trait::is_available<_Toolset>::value)
            return;

        auto _typeid_hash = std::hash<std::string>()(demangle<_Toolset>());
        if(get_typeids().count(_typeid_hash) > 0)
            return;

        internal_init<_Toolset>();

        using _Toolset_t = auto_tuple<_Toolset>;
        auto _start      = [=](const std::string& _prefix) {
            _Toolset_t* _result = new _Toolset_t(_prefix, _flat);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            _Toolset_t* _result = static_cast<_Toolset_t*>(v_result);
            _result->stop();
            delete _result;
        };

        lock_t lk(get_lock());
        get_typeids().insert(_typeid_hash);
        get_start().emplace_back(_start);
        get_stop().emplace_back(_stop);
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename _Toolset, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int>                = 0,
              enable_if_t<(trait::is_available<_Toolset>::value), int> = 0,
              enable_if_t<!(_Toolset::is_component), char>             = 0>
    static void configure(bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Toolset>().c_str());

        if(_Toolset::size() == 0)
            return;

        auto _typeid_hash = std::hash<std::string>()(demangle<_Toolset>());
        if(get_typeids().count(_typeid_hash) > 0)
            return;

        internal_init();

        auto _start = [=](const std::string& _prefix) {
            constexpr bool is_component_type = _Toolset::is_component_type;
            _Toolset* _result = (is_component_type) ? new _Toolset(_prefix, true, _flat)
                                                    : new _Toolset(_prefix, _flat);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            _Toolset* _result = static_cast<_Toolset*>(v_result);
            _result->stop();
            delete _result;
        };

        lock_t lk(get_lock());
        get_typeids().insert(_typeid_hash);
        get_start().emplace_back(_start);
        get_stop().emplace_back(_stop);
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools with an initializer
    //
    template <typename _Toolset, typename _InitFunc,
              enable_if_t<(trait::is_available<_Toolset>::value), int> = 0,
              enable_if_t<!(_Toolset::is_component), char>             = 0>
    static void configure(_InitFunc&& _init, bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Toolset>().c_str());

        if(_Toolset::size() == 0)
            return;

        auto _typeid_hash = std::hash<std::string>()(demangle<_Toolset>());
        if(get_typeids().count(_typeid_hash) > 0)
            return;

        internal_init();

        auto _start = [=](const std::string& _prefix) {
            constexpr bool is_component_type = _Toolset::is_component_type;
            _Toolset* _result = (is_component_type) ? new _Toolset(_prefix, true, _flat)
                                                    : new _Toolset(_prefix, _flat);
            _init(*_result);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            _Toolset* _result = static_cast<_Toolset*>(v_result);
            _result->stop();
            delete _result;
        };

        lock_t lk(get_lock());
        get_typeids().insert(_typeid_hash);
        get_start().emplace_back(_start);
        get_stop().emplace_back(_stop);
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a variadic list of tools
    //
    template <typename _Head, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    static void configure(bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Head>().c_str());

        configure<_Head>(_flat);
        configure<_Tail...>(_flat);
    }

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename _Toolset, typename... _Tail, typename... _Args,
              enable_if_t<(sizeof...(_Tail) == 0), int>                 = 0,
              enable_if_t<!(trait::is_available<_Toolset>::value), int> = 0>
    static void configure(_Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename _Toolset, typename... _Tail, typename... _Args,
              enable_if_t<(sizeof...(_Tail) > 0), int>                  = 0,
              enable_if_t<!(trait::is_available<_Toolset>::value), int> = 0>
    static void configure(_Args&&... _args)
    {
        configure<_Tail...>(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly configure the start functions
    //
    static start_func_vec_t& get_start()
    {
        static start_func_vec_t _instance{};
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly configure the stop functions
    //
    static stop_func_vec_t& get_stop()
    {
        static stop_func_vec_t _instance{};
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  The configuration strings
    //
    static typeid_set_t& get_typeids()
    {
        static typeid_set_t _instance{};
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly clear the previous configurations
    //
    static void reset()
    {
        lock_t lk(get_lock());
        get_start().clear();
        get_stop().clear();
        get_typeids().clear();
    }

public:
    //----------------------------------------------------------------------------------//
    //  Member functions
    //
    void start()
    {
        base_type::set_started();
        m_bundle.resize(m_start.size(), nullptr);
        for(int64_t i = 0; i < (int64_t) m_start.size(); ++i)
            m_bundle[i] = m_start[i](m_prefix);
    }

    void stop()
    {
        assert(m_stop.size() == m_bundle.size());
        for(int64_t i = 0; i < (int64_t) m_stop.size(); ++i)
            m_stop[i](m_bundle[i]);
        base_type::set_stopped();
    }

    void clear()
    {
        if(base_type::is_running)
            stop();
        m_start.clear();
        m_stop.clear();
        m_bundle.clear();
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

public:
    //----------------------------------------------------------------------------------//
    //  Configure the tool with a specific start and stop
    //
    void insert(start_func_t&& _start, stop_func_t&& _stop)
    {
        internal_init();
        m_start.emplace_back(std::forward<start_func_t>(_start));
        m_stop.emplace_back(std::forward<stop_func_t>(_stop));
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename _Toolset, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int>                = 0,
              enable_if_t<(trait::is_available<_Toolset>::value), int> = 0,
              enable_if_t<(_Toolset::is_component), char>              = 0>
    void insert(bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Toolset>().c_str());

        internal_init<_Toolset>();

        using _Toolset_t = auto_tuple<_Toolset>;
        auto _start      = [=](const std::string& _prefix) {
            _Toolset_t* _result = new _Toolset_t(_prefix, _flat);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            _Toolset_t* _result = static_cast<_Toolset_t*>(v_result);
            _result->stop();
            delete _result;
        };

        m_start.emplace_back(_start);
        m_stop.emplace_back(_stop);
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename _Toolset, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int>                = 0,
              enable_if_t<(trait::is_available<_Toolset>::value), int> = 0,
              enable_if_t<!(_Toolset::is_component), char>             = 0>
    void insert(bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Toolset>().c_str());

        internal_init();

        auto _start = [=](const std::string& _prefix) {
            constexpr bool is_component_type = _Toolset::is_component_type;
            _Toolset* _result = (is_component_type) ? new _Toolset(_prefix, true, _flat)
                                                    : new _Toolset(_prefix, _flat);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            _Toolset* _result = static_cast<_Toolset*>(v_result);
            _result->stop();
            delete _result;
        };

        m_start.emplace_back(_start);
        m_stop.emplace_back(_stop);
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools with an initializer
    //
    template <typename _Toolset, typename _InitFunc,
              enable_if_t<(trait::is_available<_Toolset>::value), int> = 0,
              enable_if_t<!(_Toolset::is_component), char>             = 0>
    void insert(_InitFunc&& _init, bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Toolset>().c_str());

        internal_init();

        auto _start = [=](const std::string& _prefix) {
            constexpr bool is_component_type = _Toolset::is_component_type;
            _Toolset* _result = (is_component_type) ? new _Toolset(_prefix, true, _flat)
                                                    : new _Toolset(_prefix, _flat);
            _init(*_result);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            _Toolset* _result = static_cast<_Toolset*>(v_result);
            _result->stop();
            delete _result;
        };

        m_start.emplace_back(_start);
        m_stop.emplace_back(_stop);
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a variadic list of tools
    //
    template <typename _Head, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void insert(bool _flat = settings::flat_profile())
    {
        DEBUG_PRINT_HERE("%s", demangle<_Head>().c_str());

        configure<_Head>(_flat);
        configure<_Tail...>(_flat);
    }

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename _Toolset, typename... _Tail, typename... _Args,
              enable_if_t<(sizeof...(_Tail) == 0), int>                 = 0,
              enable_if_t<!(trait::is_available<_Toolset>::value), int> = 0>
    void insert(_Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename _Toolset, typename... _Tail, typename... _Args,
              enable_if_t<(sizeof...(_Tail) > 0), int>                  = 0,
              enable_if_t<!(trait::is_available<_Toolset>::value), int> = 0>
    void insert(_Args&&... _args)
    {
        insert<_Tail...>(std::forward<_Args>(_args)...);
    }

protected:
    std::string      m_prefix;
    void_vec_t       m_bundle;
    start_func_vec_t m_start;
    stop_func_vec_t  m_stop;

private:
    //----------------------------------------------------------------------------------//
    //  Get lock
    //
    static mutex_t& get_lock()
    {
        static mutex_t _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  Initialize the storage
    //
    static void internal_init()
    {
        static auto _instance = []() {
            auto ret = storage_type::instance();
            ret->initialize();
            return true;
        };
        static bool _inited = _instance();
        consume_parameters(_inited);
    }

    //
    template <typename _Tool, enable_if_t<(_Tool::is_component), char> = 0>
    static void internal_init()
    {
        internal_init();
        static auto _instance = []() {
            using tool_storage_type = typename _Tool::storage_type;
            auto ret                = tool_storage_type::instance();
            ret->initialize();
            return true;
        };
        static bool _inited = _instance();
        consume_parameters(_inited);
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace component

//--------------------------------------------------------------------------------------//
}  // namespace tim
