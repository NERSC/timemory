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

/** \file base_timer.cpp
 * Base class for the timer class
 * Not directly used
 */

#include <algorithm>
#include <cassert>

#include "timemory/base_timer.hpp"
#include "timemory/serializer.hpp"
#include "timemory/utility.hpp"

//======================================================================================//

namespace tim
{
namespace internal
{
//======================================================================================//

base_timer::mutex_map_t base_timer::f_mutex_map;

//======================================================================================//

base_timer::base_timer(timer_format_t _format, std::ostream* os)
: m_os(os)
, m_data(data_t())
, m_format(_format)
{
}

//======================================================================================//

base_timer::base_timer(const base_timer& rhs)
: m_os(rhs.m_os)
, m_data(rhs.m_data)
, m_accum(rhs.m_accum)
, m_format(rhs.m_format)
{
}

//======================================================================================//

base_timer::~base_timer()
{
    if(m_data.running())
    {
        this->stop();
        if(m_os != &std::cout && *m_os)
            this->report();
    }
}

//======================================================================================//

base_timer&
base_timer::operator=(const base_timer& rhs)
{
    if(this != &rhs)
    {
        m_os     = rhs.m_os;
        m_data   = rhs.m_data;
        m_accum  = rhs.m_accum;
        m_format = rhs.m_format;
    }
    return *this;
}

//======================================================================================//

void
base_timer::sync(const this_type& rhs)
{
    if(this != &rhs)
    {
        m_os    = rhs.m_os;
        m_data  = rhs.m_data;
        m_accum = rhs.m_accum;
    }
}

//======================================================================================//

bool
base_timer::above_cutoff(bool ign_cutoff) const
{
    if(ign_cutoff)
        return true;

    double _cpu = user_elapsed() + system_elapsed();

    double tmin = 1.0 / (pow((uint32_t) 10, (uint32_t) m_format->precision()));
    // skip if it will be reported as all zeros
    // e.g. tmin = ( 1. / 10^3 ) = 0.001;
    if((real_elapsed() < tmin && _cpu < tmin) || (_cpu / real_elapsed()) < 0.001)
        return false;

    return true;
}

//======================================================================================//

void
base_timer::report(bool endline) const
{
    this->report(*m_os, endline);
}

//======================================================================================//

}  // namespace internal

}  // namespace tim

//======================================================================================//
