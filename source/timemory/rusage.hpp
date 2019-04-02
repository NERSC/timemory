//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>

#include "timemory/formatters.hpp"
#include "timemory/macros.hpp"
#include "timemory/serializer.hpp"

//============================================================================//

#if defined(_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(_MACOS)
#        include <mach/mach.h>
#    endif
#elif defined(_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    include <psapi.h>
#    include <stdio.h>
#    include <windows.h>
#else
#    error "Cannot define get_peak_rss() or get_current_rss() for an unknown OS."
#endif

// RSS - Resident set size (physical memory use, not in swap)

//----------------------------------------------------------------------------//

namespace tim
{
//----------------------------------------------------------------------------//

namespace rusage
{
//----------------------------------------------------------------------------//

enum record_type
{
    PEAK_RSS,
    CURRENT_RSS,
    STACK_RSS,
    DATA_RSS,
    NUM_SWAP,
    NUM_IO_IN,
    NUM_IO_OUT,
    NUM_MINOR_PAGE_FAULTS,
    NUM_MAJOR_PAGE_FAULTS
};

//----------------------------------------------------------------------------//

intmax_t
get_peak_rss();
intmax_t
get_current_rss();
intmax_t
get_stack_rss();
intmax_t
get_data_rss();
intmax_t
get_num_swap();
intmax_t
get_num_io_in();
intmax_t
get_num_io_out();
intmax_t
get_num_minor_page_faults();
intmax_t
get_num_major_page_faults();

//----------------------------------------------------------------------------//

template <typename _Tp>
struct base
{
    typedef _Tp Type;
    intmax_t    value = 0;

    base(intmax_t _value = 0)
    : value(_value)
    {
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(serializer::make_nvp(Type::label(), value),
           serializer::make_nvp("units", Type::units));
    }

    intmax_t operator()() { return (value = Type::record()); }
    intmax_t start() { return (*this)(); }
    intmax_t stop() { return (*this)(); }

    _Tp& max(const base<_Tp>& rhs) { return (value = std::max(value, rhs.value)); }
    _Tp  max(const base<_Tp>& rhs) const { return std::max(value, rhs.value); }
    _Tp& min(const base<_Tp>& rhs) { return (value = std::min(value, rhs.value)); }
    _Tp  min(const base<_Tp>& rhs) const { return std::min(value, rhs.value); }
    _Tp& operator+=(const base<_Tp>& rhs) const
    {
        value += rhs.value;
        return static_cast<_Tp&>(*this);
    }
    _Tp& operator-=(const base<_Tp>& rhs) const
    {
        value -= rhs.value;
        return static_cast<_Tp&>(*this);
    }
    _Tp& operator*=(const uintmax_t& rhs)
    {
        value *= rhs;
        return static_cast<_Tp&>(*this);
    }
    _Tp& operator/=(const uintmax_t& rhs)
    {
        value /= rhs;
        return static_cast<_Tp&>(*this);
    }
};

//----------------------------------------------------------------------------//

struct peak_rss : public base<peak_rss>
{
    static const record_type category = PEAK_RSS;
    static std::string       label() { return "peak RSS"; }
    static const intmax_t    units = units::kilobyte;
    static intmax_t          record() { return get_peak_rss(); }
};

//----------------------------------------------------------------------------//

struct current_rss : public base<current_rss>
{
    static const record_type category = CURRENT_RSS;
    static std::string       label() { return "current RSS"; }
    static const intmax_t    units = units::kilobyte;
    static intmax_t          record() { return get_current_rss(); }
};

//----------------------------------------------------------------------------//

struct stack_rss : public base<stack_rss>
{
    static const record_type category = STACK_RSS;
    static std::string       label() { return "stack RSS"; }
    static const intmax_t    units = units::kilobyte;
    static intmax_t          record() { return get_stack_rss(); }
};

//----------------------------------------------------------------------------//

struct data_rss : public base<data_rss>
{
    static const record_type category = DATA_RSS;
    static std::string       label() { return "data RSS"; }
    static const intmax_t    units = units::kilobyte;
    static intmax_t          record() { return get_data_rss(); }
};

//----------------------------------------------------------------------------//

struct num_swap : public base<num_swap>
{
    static const record_type category = NUM_SWAP;
    static std::string       label() { return "num swap"; }
    static const intmax_t    units = 1;
    static intmax_t          record() { return get_num_swap(); }
};

//----------------------------------------------------------------------------//

struct num_io_in : public base<num_io_in>
{
    static const record_type category = NUM_IO_IN;
    static std::string       label() { return "num I/O in"; }
    static const intmax_t    units = 1;
    static intmax_t          record() { return get_num_io_in(); }
};

//----------------------------------------------------------------------------//

struct num_io_out : public base<num_io_out>
{
    static const record_type category = NUM_IO_OUT;
    static std::string       label() { return "num I/O out"; }
    static const intmax_t    units = 1;
    static intmax_t          record() { return get_num_io_out(); }
};

//----------------------------------------------------------------------------//

struct num_minor_page_faults : public base<num_minor_page_faults>
{
    static const record_type category = NUM_MINOR_PAGE_FAULTS;
    static std::string       label() { return "num minor page faults"; }
    static const intmax_t    units = 1;
    static intmax_t          record() { return get_num_minor_page_faults(); }
};

//----------------------------------------------------------------------------//

struct num_major_page_faults : public base<num_major_page_faults>
{
    static const record_type category = NUM_MAJOR_PAGE_FAULTS;
    static std::string       label() { return "num major page faults"; }
    static const intmax_t    units = 1;
    static intmax_t          record() { return get_num_major_page_faults(); }
};

//----------------------------------------------------------------------------//

typedef std::tuple<peak_rss, current_rss, stack_rss, data_rss, num_swap, num_io_in,
                   num_io_out, num_minor_page_faults, num_major_page_faults>
    types_t;

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

template <typename _Tp>
struct max : public base<_Tp>
{
    max(base<_Tp>& obj) { obj.value = std::max(obj.value, _Tp::record()); }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct record : public base<_Tp>
{
    record(base<_Tp>& obj) { obj(); }
    record(base<_Tp>& obj, const base<_Tp>& rhs) { obj = obj.max(rhs); }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct reset : public base<_Tp>
{
    reset(base<_Tp>& obj) { obj.value = 0; }
};

//----------------------------------------------------------------------------//

template <typename _Tp>
struct print : public base<_Tp>
{
    print(base<_Tp>& obj, std::ostream& os) { os << obj << std::endl; }
};

}  // namespace rusage

//----------------------------------------------------------------------------//

}  // namespace tim

//----------------------------------------------------------------------------//

#include "timemory/rusage.icpp"
