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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/user_bundle/backends.hpp
 * \brief Implementation of the user_bundle functions/utilities
 */

#pragma once

#include "timemory/components/gotcha/backends.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/variadic/types.hpp"
#include "timemory/storage/declaration.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
//======================================================================================//
//
namespace tim
{
namespace component
{
//
struct opaque
{
    using string_t = std::string;

    using init_func_t   = std::function<void()>;
    using start_func_t  = std::function<void*(const string_t&, bool)>;
    using stop_func_t   = std::function<void(void*)>;
    using get_func_t    = std::function<void(void*, void*&, size_t)>;
    using delete_func_t = std::function<void(void*)>;

    template <typename InitF, typename StartF, typename StopF, typename GetF,
              typename DelF>
    opaque(bool _valid, size_t _typeid, InitF&& _init, StartF&& _start, StopF&& _stop,
           GetF&& _get, DelF&& _del)
    : m_valid(_valid)
    , m_typeid(_typeid)
    , m_init(std::move(_init))
    , m_start(std::move(_start))
    , m_stop(std::move(_stop))
    , m_get(std::move(_get))
    , m_del(std::move(_del))
    {}

    ~opaque()
    {
        if(m_data)
        {
            m_stop(m_data);
            m_del(m_data);
        }
    }

    opaque()              = default;
    opaque(const opaque&) = default;
    opaque(opaque&&)      = default;
    opaque& operator=(const opaque&) = default;
    opaque& operator=(opaque&&) = default;

    operator bool() const { return m_valid; }

    void init() { m_init(); }

    void start(const string_t& _prefix, bool flat)
    {
        if(m_data)
        {
            stop();
            cleanup();
        }
        m_data  = m_start(_prefix, flat);
        m_valid = (m_data != nullptr);
    }

    void stop()
    {
        if(m_data)
            m_stop(m_data);
    }

    void cleanup()
    {
        if(m_data && !m_copy)
        {
            // gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());
            m_del(m_data);
        }
        m_data = nullptr;
    }

    void get(void*& ptr, size_t _hash) const
    {
        if(m_data)
            m_get(m_data, ptr, _hash);
    }

    void set_copy(bool val) { m_copy = val; }

    bool          m_valid  = false;
    bool          m_copy   = false;
    size_t        m_typeid = 0;
    void*         m_data   = nullptr;
    init_func_t   m_init   = []() {};
    start_func_t  m_start  = [](const string_t&, bool) { return nullptr; };
    stop_func_t   m_stop   = [](void*) {};
    get_func_t    m_get    = [](void*, void*&, size_t) {};
    delete_func_t m_del    = [](void*) {};
};
//
//--------------------------------------------------------------------------------------//
//
namespace factory
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<(concepts::is_comp_wrapper<Tp>::value), int> = 0>
static auto
create_heap_variadic(Label&& _label, bool flat, Args&&... args)
{
    std::string msg =
        apply<std::string>::join(", ", _label, "true", flat, demangle<Args>()...);
    PRINT_HERE("%s(%s)", demangle<Tp>().c_str(), msg.c_str());
    return new Tp(std::forward<Label>(_label), true, flat, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<(concepts::is_auto_wrapper<Tp>::value), int> = 0>
static auto
create_heap_variadic(Label&& _label, bool flat, Args&&... args)
{
    std::string msg = apply<std::string>::join(", ", _label, flat, demangle<Args>()...);
    PRINT_HERE("%s(%s)", demangle<Tp>().c_str(), msg.c_str());
    return new Tp(std::forward<Label>(_label), flat, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<!(concepts::is_auto_wrapper<Tp>::value ||
                        concepts::is_auto_wrapper<Tp>::value),
                      int> = 0>
static auto
create_heap_variadic(Label&& _label, bool, Args&&... args)
{
    std::string msg = apply<std::string>::join(", ", _label, demangle<Args>()...);
    PRINT_HERE("%s(%s)", demangle<Tp>().c_str(), msg.c_str());
    return new Tp(std::forward<Label>(_label), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
namespace hidden
{
//
//--------------------------------------------------------------------------------------//
//
static inline size_t
get_opaque_hash(const std::string& key)
{
    auto ret = std::hash<std::string>()(key);
    return ret;
}
//
//--------------------------------------------------------------------------------------//
//
//      simplify forward declaration
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific component
//
template <typename Toolset, typename... Args,
          enable_if_t<(trait::is_available<Toolset>::value &&
                       !concepts::is_wrapper<Toolset>::value),
                      int> = 0>
static auto
get_opaque(bool flat, Args&&... args)
{
    using Toolset_t = component_tuple<Toolset>;

    PRINT_HERE("%s", demangle<Toolset>().c_str());

    auto _typeid_hash = get_opaque_hash(demangle<Toolset>());

    auto _init = []() {
        static bool _inited = []() {
            operation::init_storage<Toolset>{};
            return true;
        }();
        consume_parameters(_inited);
    };

    auto _start = [=, &args...](const string_t& _prefix, bool argflat) {
        Toolset_t* _result =
            new Toolset_t(_prefix, true, flat || argflat, std::forward<Args>(args)...);
        _result->start();
        return (void*) _result;
    };

    auto _stop = [=](void* v_result) {
        Toolset_t* _result = static_cast<Toolset_t*>(v_result);
        _result->stop();
    };

    auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
        if(_hash == _typeid_hash && v_result && !ptr)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            ptr                = static_cast<void*>(_result->template get<Toolset>());
        }
    };

    auto _del = [=](void* v_result) {
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            delete _result;
        }
    };

    return opaque(true, _typeid_hash, _init, _start, _stop, _get, _del);
}
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific set of tools
//
template <typename Toolset, typename... Args,
          enable_if_t<(concepts::is_wrapper<Toolset>::value), int> = 0>
static auto
get_opaque(bool flat, Args&&... args)
{
    using Toolset_t = Toolset;

    PRINT_HERE("%s", demangle<Toolset>().c_str());

    if(Toolset::size() == 0)
    {
        DEBUG_PRINT_HERE("returning! %s is empty", demangle<Toolset>().c_str());
        return opaque{};
    }

    auto _typeid_hash = get_opaque_hash(demangle<Toolset>());

    auto _init = []() {};

    auto _start = [=, &args...](const string_t& _prefix, bool argflat) {
        Toolset_t* _result = create_heap_variadic<Toolset_t>(_prefix, flat || argflat,
                                                             std::forward<Args>(args)...);
        return (void*) _result;
    };

    auto _stop = [=](void* v_result) {
        Toolset_t* _result = static_cast<Toolset_t*>(v_result);
        _result->stop();
    };

    auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
        if(v_result && !ptr)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->get(ptr, _hash);
        }
    };

    auto _del = [=](void* v_result) {
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            delete _result;
        }
    };

    return opaque(true, _typeid_hash, _init, _start, _stop, _get, _del);
}
//
//--------------------------------------------------------------------------------------//
//
//  If a tool is not avail or has no contents return empty opaque
//
template <typename Toolset, typename... Args,
          enable_if_t<(!trait::is_available<Toolset>::value), int> = 0>
static auto
get_opaque(bool, Args&&...)
{
    return opaque{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct opaque_typeids
{
    using result_type = std::set<size_t>;

    template <typename U = T, enable_if_t<(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        return result_type({ get_opaque_hash(demangle<T>()) });
    }

    template <typename U = T, enable_if_t<!(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        return result_type({ 0 });
    }

    template <typename U = T, enable_if_t<(trait::is_available<U>::value), int> = 0>
    static auto hash()
    {
        return get_opaque_hash(demangle<T>());
    }

    template <typename U = T, enable_if_t<!(trait::is_available<U>::value), int> = 0>
    static auto hash() -> size_t
    {
        return 0;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class Tuple, typename... T>
struct opaque_typeids<Tuple<T...>>
{
    using result_type = std::set<size_t>;

    template <typename U>
    static auto get(result_type& ret)
    {
        ret.insert(get_opaque_hash(demangle<U>()));
    }

    template <typename U                                        = Tuple<T...>,
              enable_if_t<(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        result_type ret;
        get<Tuple<T...>>(ret);
        TIMEMORY_FOLD_EXPRESSION(get<T>(ret));
        return ret;
    }

    template <typename U                                         = Tuple<T...>,
              enable_if_t<!(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        return result_type({ 0 });
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace hidden
//
//--------------------------------------------------------------------------------------//
//
//  Generic handler
//
template <typename Toolset, typename Arg, typename... Args>
static auto
get_opaque(Arg&& arg, Args&&... args)
{
    return hidden::get_opaque<Toolset>(std::forward<Arg>(arg),
                                       std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
static auto
get_opaque()
{
    return hidden::get_opaque<Toolset>(settings::flat_profile());
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
static auto
get_typeids()
{
    return hidden::opaque_typeids<Toolset>::get();
}
//
}  // namespace factory
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
