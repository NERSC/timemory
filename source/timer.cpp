// MIT License
//
// Copyright (c) 2018 Jonathan R. Madsen
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

#include "timemory/timer.hpp"
#include <algorithm>
#include <cassert>

//============================================================================//

CEREAL_CLASS_VERSION(NAME_TIM::util::timer, TIMEMORY_TIMER_VERSION)

//============================================================================//

thread_local uint64_t NAME_TIM::util::timer::f_output_width = 10;

//============================================================================//

std::string NAME_TIM::util::timer::default_format
    =  " : %w wall, %u user + %s system = %t CPU [sec] (%p%)"
       // bash expansion:
       //   total_RSS_current, total_RSS_peak
       //   self_RSS_current self_RSS_peak
       " : RSS {tot,self}_{curr,peak}"
       " : (%C|%M)"
       " | (%c|%m) [MB]";

//============================================================================//

uint16_t NAME_TIM::util::timer::default_precision = 3;

//============================================================================//

void NAME_TIM::util::timer::propose_output_width(uint64_t _w)
{
    f_output_width = std::max(f_output_width, _w);
}

//============================================================================//

NAME_TIM::util::timer::timer(const string_t& _begin,
                        const string_t& _close,
                        bool _use_static_width,
                        uint16_t prec)
: base_type(prec, _begin + default_format + _close),
  m_use_static_width(_use_static_width),
  m_begin(_begin), m_close(_close)
{ }

//============================================================================//

NAME_TIM::util::timer::timer(const string_t& _begin,
                        const string_t& _end,
                        const string_t& _fmt,
                        bool _use_static_width,
                        uint16_t prec)
: base_type(prec, _begin + _fmt + _end),
  m_use_static_width(_use_static_width),
  m_begin(_begin), m_close(_end)
{ }

//============================================================================//

NAME_TIM::util::timer::~timer()
{ }

//============================================================================//

void NAME_TIM::util::timer::compose()
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
