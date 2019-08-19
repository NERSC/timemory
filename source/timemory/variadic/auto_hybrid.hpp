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

/** \file auto_hybrid.hpp
 * \headerfile auto_hybrid.hpp "timemory/variadic/auto_hybrid.hpp"
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
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/macros.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename _CompTuple, typename _CompList>
class auto_hybrid
: public counted_object<auto_hybrid<_CompTuple, _CompList>>
, public hashed_object<auto_hybrid<_CompTuple, _CompList>>
{
    static_assert(_CompTuple::is_component_tuple && _CompList::is_component_list,
                  "Error! _CompTuple must be tim::component_tuple<...> and _CompList "
                  "must be tim::component_list<...>");

public:
    using tuple_type     = _CompTuple;
    using list_type      = _CompList;
    using component_type = component_hybrid<tuple_type, list_type>;
    using this_type      = auto_hybrid<tuple_type, list_type>;
    using data_type      = typename component_type::data_type;
    using counter_type   = counted_object<this_type>;
    using counter_void   = counted_object<void>;
    using hashed_type    = hashed_object<this_type>;
    using string_t       = std::string;
    using string_hash    = std::hash<string_t>;
    using base_type      = component_type;
    using language_t     = language;

public:
    inline explicit auto_hybrid(const string_t&, const int64_t& lineno = 0,
                                const language_t& lang           = language_t::cxx(),
                                bool              report_at_exit = false);
    inline explicit auto_hybrid(component_type& tmp, const int64_t& lineno = 0,
                                bool report_at_exit = false);
    inline ~auto_hybrid();

    // copy and move
    inline auto_hybrid(const this_type&) = default;
    inline auto_hybrid(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    inline component_type&       component_tuple() { return m_temporary_object; }
    inline const component_type& component_tuple() const { return m_temporary_object; }

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

public:
    tuple_type&       get_tuple() { return m_temporary_object.get_tuple(); }
    const tuple_type& get_tuple() const { return m_temporary_object.get_tuple(); }
    list_type&        get_list() { return m_temporary_object.get_list(); }
    const list_type&  get_list() const { return m_temporary_object.get_list(); }

    tuple_type&       get_first() { return m_temporary_object.get_tuple(); }
    const tuple_type& get_first() const { return m_temporary_object.get_tuple(); }
    list_type&        get_second() { return m_temporary_object.get_list(); }
    const list_type&  get_second() const { return m_temporary_object.get_list(); }

    tuple_type&       get_lhs() { return m_temporary_object.get_tuple(); }
    const tuple_type& get_lhs() const { return m_temporary_object.get_tuple(); }
    list_type&        get_rhs() { return m_temporary_object.get_list(); }
    const list_type&  get_rhs() const { return m_temporary_object.get_list(); }

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

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::auto_hybrid(const string_t&   object_tag,
                                                const int64_t&    lineno,
                                                const language_t& lang,
                                                bool              report_at_exit)
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

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::auto_hybrid(component_type& tmp,
                                                const int64_t&  lineno,
                                                bool            report_at_exit)
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

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::~auto_hybrid()
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
