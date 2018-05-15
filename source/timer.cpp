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

/** \file timer.cpp
 * Primary timer class
 * Inherits from base_timer
 */

#include "timemory/timer.hpp"
#include "timemory/serializer.hpp"

#include <algorithm>
#include <cassert>

//============================================================================//

namespace tim
{

bool& timer::f_record_memory()
{
    static bool _record_memory = true;
    return _record_memory;
}

//============================================================================//

timer::timer(bool _auto_start, timer* _sum_timer)
: base_type(nullptr, timer::default_record_memory()),
  m_sum_timer(_sum_timer)
{
    if(_auto_start)
        this->start();
}

//============================================================================//

timer::timer(const string_t& _prefix,
             const string_t& _format,
             bool _record_memory)
: base_type(timer_format_t(new format_type(_prefix, _format)), _record_memory),
  m_sum_timer(nullptr)
{ }

//============================================================================//

timer::timer(const format_type& _format, bool _record_memory)
: base_type(timer_format_t(new format_type(_format)), _record_memory),
  m_sum_timer(nullptr)
{ }

//============================================================================//

timer::timer(timer_format_t _format, bool _record_memory)
: base_type(_format, _record_memory),
  m_sum_timer(nullptr)
{ }

//============================================================================//

timer::timer(const this_type* rhs, const string_t& _prefix,
             bool _align_width, bool _record_memory)
: base_type(timer_format_t(
                new format_type(_prefix,
                                (rhs) ? rhs->format()->format()
                                      : format::timer::default_format())),
            _record_memory),
  m_sum_timer(nullptr)
{
    if(rhs)
        this->sync(*rhs);
    this->format()->align_width(_align_width);
}

//============================================================================//

timer::~timer()
{ }

//============================================================================//

timer::timer(const this_type& rhs)
: base_type(timer_format_t(new format_type(rhs.format()->prefix(),
                                           rhs.format()->format())),
            rhs.m_record_memory),
  m_sum_timer(rhs.m_sum_timer)
{
    m_accum = rhs.get_accum();
}

//============================================================================//

timer::this_type& timer::operator=(const this_type& rhs)
{
    if(this != &rhs)
    {
        base_type::operator=(rhs);
        if(!m_format.get())
            m_format = timer_format_t(new format_type());
        if(rhs.format().get())
            *m_format = *(rhs.format().get());
        m_accum = rhs.get_accum();
        m_sum_timer = rhs.m_sum_timer;
    }
    return *this;
}

//============================================================================//

void timer::grab_metadata(const this_type& rhs)
{
    if(!rhs.m_format.get())
        return;

    if(!m_format.get())
        m_format = timer_format_t(new format_type());

    *(m_format.get()) = *(rhs.m_format.get());
}

//============================================================================//

} // namespace tim
