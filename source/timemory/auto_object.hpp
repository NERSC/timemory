// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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

/** \file auto_object.hpp
 * \headerfile auto_object.hpp "timemory/auto_object.hpp"
 * Automatic timers
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_TIMER()
 *    \param TIMEMORY_BASIC_AUTO_TIMER()
 *    \param auto t = TIMEMORY_AUTO_TIMER_OBJ()
 *    \param auto t = TIMEMORY_BASIC_AUTO_TIMER_OBJ()
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/string.hpp"
#include "timemory/utility.hpp"

TIM_NAMESPACE_BEGIN

//--------------------------------------------------------------------------------------//

template <typename AutoType, typename ObjectType>
class tim_api auto_object
: public tim::counted_object<AutoType>
, public tim::hashed_object<AutoType>
{
public:
    typedef tim::counted_object<AutoType>     counter_type;
    typedef tim::counted_object<void>         counter_void;
    typedef tim::hashed_object<AutoType>      hashed_type;
    typedef ObjectType                        object_type;
    typedef auto_object<AutoType, ObjectType> this_type;
    typedef tim::string                       string_t;

public:
    // standard constructor
    auto_object(const string_t&, const int32_t& lineno, const string_t& = "cxx",
                bool report_at_exit = false);
    // destructor
    virtual ~auto_object();

    // copy and move
    auto_object(const this_type&) = default;
    auto_object(this_type&&)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) = default;

public:
    // public member functions
    object_type&       local_object() { return m_temp_object; }
    const object_type& local_object() const { return m_temp_object; }

protected:
    inline string_t get_tag(const string_t& timer_tag, const string_t& lang_tag);

private:
    bool        m_enabled;
    bool        m_report_at_exit;
    uintmax_t   m_hash;
    object_type m_temp_object;
};

//============================================================================//

template <typename AutoType, typename ObjectType>
auto_object<AutoType, ObjectType>::auto_object(const string_t& object_tag,
                                               const int32_t&  lineno,
                                               const string_t& lang_tag,
                                               bool            report_at_exit)
: counter_type()
, hashed_type((counter_type::enable()) ? (lineno + std::hash<string_t>()(object_tag)) : 0)
, m_enabled(counter_type::enable())
, m_report_at_exit(report_at_exit)
, m_temp_object(object_type(
      m_enabled, (m_enabled)
                     ? &manager::instance()->get<ObjectType>(
                           object_tag, lang_tag,
                           (m_enabled) ? counter_type::live() : counter_type::zero(),
                           (m_enabled) ? hashed_type::live() : hashed_type::zero())
                     : nullptr))
{
}

//============================================================================//

template <typename AutoType, typename ObjectType>
auto_object<AutoType, ObjectType>::~auto_object()
{
    if(m_enabled)
    {
        // stop the timer
        m_temp_object.stop();
        assert(m_temp_object.summation_object() != nullptr);
        *m_temp_object.summation_object() += m_temp_object;

        // report timer at exit
        if(m_report_at_exit)
        {
            // m_temp_object.grab_metadata(*(m_temp_object.summation_object()));

            // show number of laps in temporary timer
            auto _laps = m_temp_object.summation_object()->accum().size();
            m_temp_object.accum().size() += _laps;

            // threadsafe output w.r.t. other timers
            m_temp_object.grab_metadata(*m_temp_object.summation_object());
            m_temp_object.report(std::cout, true, true);
        }
        // count and hash keys already taken care of so just pop the graph
        // manager::instance()->pop_graph<typename _Tp::value_type>();
        manager::instance()->pop_graph<ObjectType>();
    }
}

//============================================================================//

template <typename AutoType, typename ObjectType>
typename auto_object<AutoType, ObjectType>::string_t
auto_object<AutoType, ObjectType>::get_tag(const string_t& timer_tag,
                                           const string_t& lang_tag)
{
#if defined(TIMEMORY_USE_MPI)
    std::stringstream ss;
    if(tim::mpi_is_initialized())
        ss << tim::mpi_rank();
    ss << "> [" << lang_tag << "] " << timer_tag;
    return string_t(ss.str().c_str());
#else
    std::stringstream ss;
    ss << "> [" << lang_tag << "] " << timer_tag;
    return ss.str().c_str();
#endif
}

//============================================================================//

TIM_NAMESPACE_END
