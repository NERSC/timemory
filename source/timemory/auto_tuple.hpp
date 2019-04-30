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

/** \file auto_tuple.hpp
 * \headerfile auto_tuple.hpp "timemory/auto_tuple.hpp"
 * Automatic timers
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_TUPLE()
 *    \param TIMEMORY_BASIC_AUTO_TUPLE()
 *    \param auto t = TIMEMORY_AUTO_TUPLE_OBJ()
 *    \param auto t = TIMEMORY_BASIC_AUTO_TUPLE_OBJ()
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/auto_macros.hpp"
#include "timemory/component_tuple.hpp"
#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/utility.hpp"

TIM_NAMESPACE_BEGIN

//--------------------------------------------------------------------------------------//

template <typename... Types>
class auto_tuple
: public tim::counted_object<auto_tuple<Types...>>
, public tim::hashed_object<auto_tuple<Types...>>
{
public:
    using object_type  = implemented_component_tuple<Types...>;
    using this_type    = auto_tuple<Types...>;
    using counter_type = tim::counted_object<this_type>;
    using counter_void = tim::counted_object<void>;
    using hashed_type  = tim::hashed_object<this_type>;
    using string_t     = std::string;
    using string_hash  = std::hash<string_t>;

public:
    // standard constructor
    auto_tuple(const string_t&, const int32_t& lineno = 0, const string_t& = "cxx",
               bool report_at_exit = false);
    // destructor
    virtual ~auto_tuple();

    // copy and move
    auto_tuple(const this_type&) = default;
    auto_tuple(this_type&&)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) = default;

public:
    // public member functions
    object_type&       local_object() { return m_temp_object; }
    const object_type& local_object() const { return m_temp_object; }

    // partial interface to underlying component_tuple
    void record() { m_temp_object.record(); }
    void pause() { m_temp_object.pause(); }
    void resume() { m_temp_object.resume(); }
    void start() { m_temp_object.start(); }
    void stop() { m_temp_object.stop(); }
    void push() { m_temp_object.push(); }
    void pop() { m_temp_object.pop(); }

public:
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << obj.m_temp_object;
        return os;
    }

private:
    bool        m_enabled;
    bool        m_report_at_exit;
    uint64_t    m_hash;
    object_type m_temp_object;
};

//======================================================================================//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(const string_t& object_tag, const int32_t& lineno,
                                 const string_t& lang_tag, bool report_at_exit)
: counter_type()
, hashed_type((counter_type::enable())
                  ? (string_hash()(lang_tag + object_tag) *
                     (counter_type::live() + hashed_type::live() + lineno))
                  : 0)
, m_enabled(counter_type::enable())
, m_report_at_exit(report_at_exit)
, m_temp_object(object_tag, lang_tag, counter_type::m_count, hashed_type::m_hash, true)
{
    if(m_enabled)
    {
        m_temp_object.start();
    }
}

//======================================================================================//

template <typename... Types>
auto_tuple<Types...>::~auto_tuple()
{
    if(m_enabled)
    {
        // stop the timer
        m_temp_object.stop();

        // report timer at exit
        if(m_report_at_exit)
        {
            std::stringstream ss;
            ss << m_temp_object;
            std::cout << ss.str() << std::endl;
        }
    }
}

//======================================================================================//

TIM_NAMESPACE_END

//======================================================================================//

#define TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_BLANK_AUTO_OBJECT(auto_tuple_type, __VA_ARGS__)

#define TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_BASIC_AUTO_OBJECT(auto_tuple_type, __VA_ARGS__)

#define TIMEMORY_AUTO_TUPLE(auto_tuple_type, ...)                                        \
    TIMEMORY_AUTO_OBJECT(auto_tuple_type, __VA_ARGS__)

#define TIMEMORY_AUTO_TUPLE_OBJ(auto_tuple_type, ...)                                    \
    TIMEMORY_AUTO_OBJECT_OBJ(auto_tuple_type, __VA_ARGS__)

#define TIMEMORY_BASIC_AUTO_TUPLE_OBJ(auto_tuple_type, ...)                              \
    TIMEMORY_BASIC_AUTO_OBJECT_OBJ(auto_tuple_type, __VA_ARGS__)

#define TIMEMORY_DEBUG_BASIC_AUTO_TUPLE(auto_tuple_type, ...)                            \
    TIMEMORY_DEBUG_BASIC_AUTO_OBJECT(auto_tuple_type, __VA_ARGS__)

#define TIMEMORY_DEBUG_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_DEBUG_AUTO_OBJECT(auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// variadic versions

#define TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE(tag, ...)                                     \
    typedef tim::auto_tuple<__VA_ARGS__> AUTO_TYPEDEF(__LINE__);                         \
    TIMEMORY_BASIC_AUTO_TUPLE(AUTO_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_TUPLE(tag, ...)                                           \
    typedef tim::auto_tuple<__VA_ARGS__> AUTO_TYPEDEF(__LINE__);                         \
    TIMEMORY_AUTO_TUPLE(AUTO_TYPEDEF(__LINE__), tag);

//======================================================================================//
