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

CEREAL_CLASS_VERSION(tim::timer, TIMEMORY_TIMER_VERSION)

//============================================================================//

namespace tim
{

//============================================================================//

timer::timer(const string_t& _prefix,
             const string_t& _format)
: base_type(timer_format_t(new format_type(_prefix, _format)))
{ }

//============================================================================//
timer::timer(const format_type& _format)
: base_type(timer_format_t(new format_type(_format)))
{ }

//============================================================================//
timer::timer(timer_format_t _format)
: base_type(_format)
{ }

//============================================================================//

timer::~timer()
{ }

//============================================================================//

void timer::grab_metadata(const this_type& rhs)
{
    m_format = rhs.m_format;
}

//============================================================================//

} // namespace tim
