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
 * \headerfile auto_tuple.hpp "timemory/variadic/auto_tuple.hpp"
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

#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/macros.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
class auto_tuple
: public counted_object<auto_tuple<Types...>>
, public hashed_object<auto_tuple<Types...>>
{
public:
    using component_type = component_tuple<Types...>;
    using this_type      = auto_tuple<Types...>;
    using data_type      = typename component_type::data_type;
    using counter_type   = counted_object<this_type>;
    using counter_void   = counted_object<void>;
    using hashed_type    = hashed_object<this_type>;
    using string_t       = std::string;
    using string_hash    = std::hash<string_t>;
    using base_type      = component_type;
    using language_t     = language;

public:
    inline explicit auto_tuple(const string_t&, const int64_t& lineno = 0,
                               const language_t& lang           = language_t::cxx(),
                               bool              report_at_exit = false);
    inline explicit auto_tuple(component_type& tmp, const int64_t& lineno = 0,
                               bool report_at_exit = false);
    inline ~auto_tuple();

    // copy and move
    inline auto_tuple(const this_type&) = default;
    inline auto_tuple(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

    static constexpr std::size_t available_size()
    {
        return component_type::available_size();
    }

public:
    // public member functions
    inline component_type&       get_component_type() { return m_temporary_object; }
    inline const component_type& get_component_type() const { return m_temporary_object; }

    // partial interface to underlying component_tuple
    inline void record() { m_temporary_object.record(); }
    inline void start() { m_temporary_object.start(); }
    inline void stop() { m_temporary_object.stop(); }
    inline void push() { m_temporary_object.push(); }
    inline void pop() { m_temporary_object.pop(); }
    inline void mark_begin() { m_temporary_object.mark_begin(); }
    inline void mark_end() { m_temporary_object.mark_end(); }

    inline void report_at_exit(bool val) { m_report_at_exit = val; }
    inline bool report_at_exit() const { return m_report_at_exit; }

    inline const bool&      store() const { return m_temporary_object.store(); }
    inline const data_type& data() const { return m_temporary_object.data(); }
    inline int64_t          laps() const { return m_temporary_object.laps(); }
    inline const int64_t&   hash() const { return m_temporary_object.hash(); }
    inline const string_t&  key() const { return m_temporary_object.key(); }
    inline const language&  lang() const { return m_temporary_object.lang(); }
    inline const string_t&  identifier() const { return m_temporary_object.identifier(); }
    inline void rekey(const string_t& _key) { m_temporary_object.rekey(_key); }

public:
    template <std::size_t _N>
    typename std::tuple_element<_N, data_type>::type& get()
    {
        return m_temporary_object.template get<_N>();
    }

    template <std::size_t _N>
    const typename std::tuple_element<_N, data_type>::type& get() const
    {
        return m_temporary_object.template get<_N>();
    }

    template <typename _Tp>
    auto get() -> decltype(std::declval<component_type>().template get<_Tp>())
    {
        return m_temporary_object.template get<_Tp>();
    }

    template <typename _Tp>
    auto get() const -> decltype(std::declval<const component_type>().template get<_Tp>())
    {
        return m_temporary_object.template get<_Tp>();
    }

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
, m_enabled(counter_type::enable() && settings::enabled())
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
, m_temporary_object(tmp.clone(hashed_type::m_hash, true))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
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
        m_temporary_object.conditional_stop();

        // report timer at exit
        if(m_report_at_exit)
        {
            std::stringstream ss;
            ss << m_temporary_object;
            if(ss.str().length() > 0)
                std::cout << ss.str() << std::endl;
        }

        if(m_reference_object)
        {
            *m_reference_object += m_temporary_object;
        }
    }
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

//  DEPRECATED use macros in timemory/variadic/macros.hpp!
/// DEPRECATED
#define TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_BLANK_OBJECT(auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_BASIC_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_BASIC_OBJECT(auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_AUTO_TUPLE(auto_tuple_type, ...)                                        \
    TIMEMORY_OBJECT(auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// caliper versions -- DEPRECATED use macros in timemory/variadic/macros.hpp!

/// DEPRECATED
#define TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(id, auto_tuple_type, ...)                      \
    TIMEMORY_BLANK_CALIPER(id, auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_BASIC_AUTO_TUPLE_CALIPER(id, auto_tuple_type, ...)                      \
    TIMEMORY_BASIC_CALIPER(id, auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_AUTO_TUPLE_CALIPER(id, auto_tuple_type, ...)                            \
    TIMEMORY_CALIPER(id, auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// instance versions -- DEPRECATED use macros in timemory/variadic/macros.hpp!

/// DEPRECATED
#define TIMEMORY_BLANK_AUTO_TUPLE_INSTANCE(auto_tuple_type, ...)                         \
    TIMEMORY_BLANK_INSTANCE(auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_BASIC_AUTO_TUPLE_INSTANCE(auto_tuple_type, ...)                         \
    TIMEMORY_BASIC_INSTANCE(auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_AUTO_TUPLE_INSTANCE(auto_tuple_type, ...)                               \
    TIMEMORY_INSTANCE(auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// debug versions -- DEPRECATED use macros in timemory/variadic/macros.hpp!

/// DEPRECATED
#define TIMEMORY_DEBUG_BASIC_AUTO_TUPLE(auto_tuple_type, ...)                            \
    TIMEMORY_DEBUG_BASIC_OBJECT(auto_tuple_type, __VA_ARGS__)

/// DEPRECATED
#define TIMEMORY_DEBUG_AUTO_TUPLE(auto_tuple_type, ...)                                  \
    TIMEMORY_DEBUG_OBJECT(auto_tuple_type, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
// variadic versions

#define TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE(tag, ...)                                     \
    using _AUTO_TYPEDEF(__LINE__) = tim::auto_tuple<__VA_ARGS__>;                        \
    TIMEMORY_BASIC_AUTO_TUPLE(_AUTO_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_TUPLE(tag, ...)                                           \
    using _AUTO_TYPEDEF(__LINE__) = tim::auto_tuple<__VA_ARGS__>;                        \
    TIMEMORY_AUTO_TUPLE(_AUTO_TYPEDEF(__LINE__), tag);

//======================================================================================//
