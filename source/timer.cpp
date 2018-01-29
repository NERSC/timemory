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
#include <algorithm>
#include <cassert>

//============================================================================//

CEREAL_CLASS_VERSION(NAME_TIM::timer, TIMEMORY_TIMER_VERSION)

//============================================================================//

uint64_t NAME_TIM::timer::f_output_width = 10;

//============================================================================//

std::string NAME_TIM::timer::default_format
    =  " : %w wall, %u user + %s system = %t CPU [sec] (%p%)"
       // bash expansion:
       //   total_RSS_current, total_RSS_peak
       //   self_RSS_current self_RSS_peak
       " : RSS {tot,self}_{curr,peak}"
       " : (%C|%M)"
       " | (%c|%m) [MB]";

//============================================================================//

namespace NAME_TIM
{

//============================================================================//

uint16_t timer::default_precision = 3;

//============================================================================//

void timer::propose_output_width(uint64_t _w)
{
    f_output_width = std::max(f_output_width, _w);
}

//============================================================================//

timer::timer(const string_t& _begin,
             const string_t& _close,
             bool _use_static_width,
             uint16_t prec)
: base_type(prec, _begin + default_format + _close),
  m_use_static_width(_use_static_width),
  m_parent(nullptr),
  m_begin(_begin), m_close(_close)
{ }

//============================================================================//

timer::timer(const string_t& _begin,
             const string_t& _end,
             const string_t& _fmt,
             bool _use_static_width,
             uint16_t prec)
: base_type(prec, _begin + _fmt + _end),
  m_use_static_width(_use_static_width),
  m_parent(nullptr),
  m_begin(_begin), m_close(_end)
{ }

//============================================================================//

timer::~timer()
{
    if(m_parent)
    {
        auto_lock_t l(m_mutex);
        m_parent->get_accum() += m_accum;
    }
}

//============================================================================//

void timer::compose()
{
    std::stringstream ss;
    if(m_use_static_width)
    {
        ss << std::setw(f_output_width + 1)
           << std::left << m_begin
           << std::right << default_format
           << m_close;
    }
    else
    {
        ss << std::left << m_begin
           << std::right << default_format
           << m_close;
    }
    m_format_string = ss.str();
}

//============================================================================//

timer timer::clone() const
{
    this_type _clone(*this);
    _clone.set_parent(const_cast<this_type*>(this));
    return _clone;
}

//============================================================================//

void timer::grab_metadata(const this_type& rhs)
{
    m_begin = rhs.m_begin;
    m_close = rhs.m_close;
    m_use_static_width = rhs.m_use_static_width;
    m_precision = rhs.m_precision;
    m_format_string = rhs.m_format_string;
}

//============================================================================//

} // namespace NAME_TIM
