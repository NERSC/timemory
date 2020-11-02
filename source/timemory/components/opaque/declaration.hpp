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

#pragma once

#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
struct opaque
{
    using string_t = std::string;

    using init_func_t   = std::function<void()>;
    using setup_func_t  = std::function<void*(void*, const string_t&, scope::config)>;
    using push_func_t   = std::function<void(void*&, const string_t&, scope::config)>;
    using start_func_t  = std::function<void(void*)>;
    using stop_func_t   = std::function<void(void*)>;
    using pop_func_t    = std::function<void(void*)>;
    using get_func_t    = std::function<void(void*, void*&, size_t)>;
    using delete_func_t = std::function<void(void*)>;

    template <typename InitF, typename StartF, typename StopF, typename GetF,
              typename DelF, typename SetupF, typename PushF, typename PopF>
    opaque(bool _valid, size_t _typeid, InitF&& _init, StartF&& _start, StopF&& _stop,
           GetF&& _get, DelF&& _del, SetupF&& _setup, PushF&& _push, PopF&& _pop)
    : m_valid(_valid)
    , m_copy(false)
    , m_typeid(_typeid)
    , m_data(nullptr)
    , m_init(std::move(_init))
    , m_setup(std::move(_setup))
    , m_push(std::move(_push))
    , m_start(std::move(_start))
    , m_stop(std::move(_stop))
    , m_pop(std::move(_pop))
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

    void setup(const string_t& _prefix, scope::config _scope)
    {
        if(m_data)
        {
            stop();
            cleanup();
        }
        m_data  = m_setup(m_data, _prefix, _scope);
        m_valid = (m_data != nullptr);
    }

    void push(const string_t& _prefix, scope::config _scope)
    {
        if(m_data)
            m_push(m_data, _prefix, _scope);
    }

    void start()
    {
        if(m_data)
            m_start(m_data);
    }

    void stop()
    {
        if(m_data)
            m_stop(m_data);
    }

    void pop()
    {
        if(m_data)
            m_pop(m_data);
    }

    void cleanup()
    {
        if(m_data && !m_copy)
            m_del(m_data);
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
    setup_func_t  m_setup = [](void*, const string_t&, scope::config) { return nullptr; };
    push_func_t   m_push  = [](void*&, const string_t&, scope::config) {};
    start_func_t  m_start = [](void*) {};
    stop_func_t   m_stop  = [](void*) {};
    pop_func_t    m_pop   = [](void*) {};
    get_func_t    m_get   = [](void*, void*&, size_t) {};
    delete_func_t m_del   = [](void*) {};
};
//
//--------------------------------------------------------------------------------------//
//
namespace factory
{
//
template <typename Toolset, typename Arg, typename... Args>
opaque
get_opaque(Arg&& arg, Args&&... args);
//
template <typename Toolset>
opaque
get_opaque();
//
template <typename Toolset>
opaque
get_opaque(scope::config _scope);
//
template <typename Toolset>
opaque
get_opaque(bool _flat);
//
template <typename Toolset>
std::set<size_t>
get_typeids();
//
}  // namespace factory
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
