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
 * Automatic starting and stopping of components. Accept unlimited number of
 * parameters. The constructor starts the components, the destructor stops the
 * components
 *
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
#include "timemory/utility.hpp"

TIM_NAMESPACE_BEGIN

//--------------------------------------------------------------------------------------//

template <typename... Types>
class auto_tuple
: public tim::counted_object<auto_tuple<Types...>>
, public tim::hashed_object<auto_tuple<Types...>>
{
public:
    using component_type = implemented_component_tuple<Types...>;
    using this_type      = auto_tuple<Types...>;
    using counter_type   = tim::counted_object<this_type>;
    using counter_void   = tim::counted_object<void>;
    using hashed_type    = tim::hashed_object<this_type>;
    using string_t       = std::string;
    using string_hash    = std::hash<string_t>;
    using base_type      = implemented_component_tuple<Types...>;
    using language_t     = tim::language;

public:
    inline auto_tuple(const string_t&, const int64_t& lineno = 0,
                      const language_t& lang           = language_t::cxx(),
                      bool              report_at_exit = false);
    inline auto_tuple(component_type& tmp, const int64_t& lineno = 0,
                      bool report_at_exit = false);
    inline ~auto_tuple();

    // copy and move
    inline auto_tuple(const this_type&) = default;
    inline auto_tuple(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size()
    {
        return std::tuple_size<std::tuple<Types...>>::value;
    }

public:
    // public member functions
    inline component_type&       component_tuple() { return m_temporary_object; }
    inline const component_type& component_tuple() const { return m_temporary_object; }

    // partial interface to underlying component_tuple
    inline void record() { m_temporary_object.record(); }
    inline void pause() { m_temporary_object.pause(); }
    inline void resume() { m_temporary_object.resume(); }
    inline void start() { m_temporary_object.start(); }
    inline void stop() { m_temporary_object.stop(); }
    inline void push() { m_temporary_object.push(); }
    inline void pop() { m_temporary_object.pop(); }
    inline void conditional_start() { m_temporary_object.conditional_start(); }
    inline void conditional_stop() { m_temporary_object.conditional_stop(); }

public:
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << obj.m_temporary_object;
        return os;
    }

private:
    bool            m_enabled        = true;
    bool            m_report_at_exit = false;
    component_type  m_temporary_object;
    component_type* m_reference_object = nullptr;
};

//======================================================================================//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(const string_t& object_tag, const int64_t& lineno,
                                 const language_t& lang, bool report_at_exit)
: counter_type()
, hashed_type((counter_type::enable())
                  ? (string_hash()(object_tag) * static_cast<int64_t>(lang) +
                     (counter_type::live() + hashed_type::live() + lineno))
                  : 0)
, m_enabled(counter_type::enable() && tim::settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(object_tag, lang, counter_type::m_count, hashed_type::m_hash,
                     m_enabled)
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(component_type& tmp, const int64_t& lineno,
                                 bool report_at_exit)
: counter_type()
, hashed_type((counter_type::enable())
                  ? (string_hash()(tmp.key()) * static_cast<int64_t>(tmp.lang()) +
                     (counter_type::live() + hashed_type::live() + lineno))
                  : 0)
, m_enabled(true)
, m_report_at_exit(report_at_exit)
, m_temporary_object(tmp)
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        m_temporary_object.hash()  = hashed_type::m_hash;
        m_temporary_object.store() = true;
        m_temporary_object.push();
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename... Types>
auto_tuple<Types...>::~auto_tuple()
{
    if(m_enabled)
    {
        // stop the timer
        m_temporary_object.stop();
        m_temporary_object.pop();

        // report timer at exit
        if(m_report_at_exit)
        {
            std::stringstream ss;
            ss << m_temporary_object;
            std::cout << ss.str() << std::endl;
        }

        if(m_reference_object)
        {
            *m_reference_object += m_temporary_object;
        }
    }
}

//======================================================================================//

TIM_NAMESPACE_END

//======================================================================================//

#define TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_type, signature)                            \
    TIMEMORY_BLANK_AUTO_OBJECT(auto_tuple_type, signature)

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
    using AUTO_TYPEDEF(__LINE__) = tim::auto_tuple<__VA_ARGS__>;                         \
    TIMEMORY_BASIC_AUTO_TUPLE(AUTO_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_TUPLE(tag, ...)                                           \
    using AUTO_TYPEDEF(__LINE__) = tim::auto_tuple<__VA_ARGS__>;                         \
    TIMEMORY_AUTO_TUPLE(AUTO_TYPEDEF(__LINE__), tag);

//======================================================================================//
