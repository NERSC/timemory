//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file units.hpp
 * \headerfile units.hpp "timemory/units.hpp"
 * Timing and memory units
 *
 */

#pragma once

#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <ratio>
#include <string>

#if defined(_UNIX)
#    include <unistd.h>
#endif

#if defined(_WINDOWS)
// clang-format off
#    include <windows.h>
#    include <sysinfoapi.h>
// clang-format on
#endif

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace units
{
const int64_t psec = std::pico::den;
const int64_t nsec = std::nano::den;
const int64_t usec = std::micro::den;
const int64_t msec = std::milli::den;
const int64_t csec = std::centi::den;
const int64_t dsec = std::deci::den;
const int64_t sec  = 1;

const int64_t byte     = 1;
const int64_t kilobyte = 1000 * byte;
const int64_t megabyte = 1000 * kilobyte;
const int64_t gigabyte = 1000 * megabyte;
const int64_t terabyte = 1000 * gigabyte;
const int64_t petabyte = 1000 * terabyte;

const int64_t kibibyte = 1024 * byte;
const int64_t mebibyte = 1024 * kibibyte;
const int64_t gibibyte = 1024 * mebibyte;
const int64_t tebibyte = 1024 * gibibyte;
const int64_t pebibyte = 1024 * tebibyte;

const int64_t B  = 1;
const int64_t KB = 1000 * B;
const int64_t MB = 1000 * KB;
const int64_t GB = 1000 * MB;
const int64_t TB = 1000 * GB;
const int64_t PB = 1000 * TB;

const int64_t Bi  = 1;
const int64_t KiB = 1024 * Bi;
const int64_t MiB = 1024 * KiB;
const int64_t GiB = 1024 * MiB;
const int64_t TiB = 1024 * GiB;
const int64_t PiB = 1024 * TiB;

#if defined(_LINUX)

inline int64_t
get_page_size()
{
    return ::sysconf(_SC_PAGESIZE);
}
const int64_t clocks_per_sec = ::sysconf(_SC_CLK_TCK);

#elif defined(_MACOS)

inline int64_t
get_page_size()
{
    return getpagesize();
}
const int64_t clocks_per_sec = ::sysconf(_SC_CLK_TCK);

#elif defined(_WINDOWS)

inline int64_t
get_page_size()
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwPageSize;
}
const int64_t clocks_per_sec = CLOCKS_PER_SEC;

#endif

//--------------------------------------------------------------------------------------//

inline std::string
time_repr(const int64_t& _unit)
{
    std::string _sunit;
    switch(_unit)
    {
        case psec: _sunit = "psec"; break;
        case nsec: _sunit = "nsec"; break;
        case usec: _sunit = "usec"; break;
        case msec: _sunit = "msec"; break;
        case csec: _sunit = "csec"; break;
        case dsec: _sunit = "dsec"; break;
        case sec: _sunit = "sec"; break;
        default: _sunit = "UNK"; break;
    }
    return _sunit;
}

inline std::string
mem_repr(const int64_t& _unit)
{
    std::string _sunit;
    switch(_unit)
    {
        case byte: _sunit = "B"; break;
        case kilobyte: _sunit = "KB"; break;
        case megabyte: _sunit = "MB"; break;
        case gigabyte: _sunit = "GB"; break;
        case terabyte: _sunit = "TB"; break;
        case petabyte: _sunit = "PB"; break;
        default: _sunit = "UNK"; break;
    }
    return _sunit;
}

//--------------------------------------------------------------------------------------//

}  // namespace units

//======================================================================================//

}  // namespace tim
