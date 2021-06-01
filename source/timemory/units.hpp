//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/** \file timemory/units.hpp
 * \headerfile timemory/units.hpp "timemory/units.hpp"
 * Timing and memory units
 *
 */

#pragma once

#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ratio>
#include <string>
#include <tuple>
#include <vector>

#if defined(TIMEMORY_UNIX)
#    include <unistd.h>
#endif

#if defined(TIMEMORY_WINDOWS)
// without this, windows will define macros for min and max
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
#    if !defined(WIN32_LEAN_AND_MEAN)
#        define WIN32_LEAN_AND_MEAN
#    endif
// clang-format off
#    include <windows.h>
#    include <sysinfoapi.h>
#    include <time.h>
// clang-format on
#endif

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace units
{
static constexpr int64_t psec = std::pico::den;
static constexpr int64_t nsec = std::nano::den;
static constexpr int64_t usec = std::micro::den;
static constexpr int64_t msec = std::milli::den;
static constexpr int64_t csec = std::centi::den;
static constexpr int64_t dsec = std::deci::den;
static constexpr int64_t sec  = 1;

static constexpr int64_t byte     = 1;
static constexpr int64_t kilobyte = 1000 * byte;
static constexpr int64_t megabyte = 1000 * kilobyte;
static constexpr int64_t gigabyte = 1000 * megabyte;
static constexpr int64_t terabyte = 1000 * gigabyte;
static constexpr int64_t petabyte = 1000 * terabyte;

static constexpr int64_t kibibyte = 1024 * byte;
static constexpr int64_t mebibyte = 1024 * kibibyte;
static constexpr int64_t gibibyte = 1024 * mebibyte;
static constexpr int64_t tebibyte = 1024 * gibibyte;
static constexpr int64_t pebibyte = 1024 * tebibyte;

static constexpr int64_t B  = 1;
static constexpr int64_t KB = 1000 * B;
static constexpr int64_t MB = 1000 * KB;
static constexpr int64_t GB = 1000 * MB;
static constexpr int64_t TB = 1000 * GB;
static constexpr int64_t PB = 1000 * TB;

static constexpr int64_t Bi  = 1;
static constexpr int64_t KiB = 1024 * Bi;
static constexpr int64_t MiB = 1024 * KiB;
static constexpr int64_t GiB = 1024 * MiB;
static constexpr int64_t TiB = 1024 * GiB;
static constexpr int64_t PiB = 1024 * TiB;

#if defined(TIMEMORY_LINUX)

inline int64_t
get_page_size()
{
    static auto _pagesz = sysconf(_SC_PAGESIZE);
    return _pagesz;
}
const int64_t clocks_per_sec = sysconf(_SC_CLK_TCK);

#elif defined(TIMEMORY_MACOS)

inline int64_t
get_page_size()
{
    static auto _pagesz = getpagesize();
    return _pagesz;
}
const int64_t clocks_per_sec = sysconf(_SC_CLK_TCK);

#elif defined(TIMEMORY_WINDOWS)

inline int64_t
get_page_size()
{
    static auto _pagesz = []() {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return sysInfo.dwPageSize;
    }();
    return _pagesz;
}
const int64_t clocks_per_sec = CLOCKS_PER_SEC;

#endif

//--------------------------------------------------------------------------------------//

inline std::string
time_repr(int64_t _unit)
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

//--------------------------------------------------------------------------------------//

inline std::string
mem_repr(int64_t _unit)
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
        case kibibyte: _sunit = "KiB"; break;
        case mebibyte: _sunit = "MiB"; break;
        case gibibyte: _sunit = "GiB"; break;
        case tebibyte: _sunit = "TiB"; break;
        case pebibyte: _sunit = "PiB"; break;
        default: _sunit = "UNK"; break;
    }
    return _sunit;
}

//--------------------------------------------------------------------------------------//

inline std::tuple<std::string, int64_t>
get_memory_unit(std::string _unit)
{
    using string_t    = std::string;
    using return_type = std::tuple<string_t, int64_t>;
    using inner_t     = std::tuple<string_t, string_t, int64_t>;

    if(_unit.length() == 0)
        return return_type{ "MB", tim::units::megabyte };

    for(auto& itr : _unit)
        itr = tolower(itr);

    for(const auto& itr : { inner_t{ "byte", "b", tim::units::byte },
                            inner_t{ "kilobyte", "kb", tim::units::kilobyte },
                            inner_t{ "megabyte", "mb", tim::units::megabyte },
                            inner_t{ "gigabyte", "gb", tim::units::gigabyte },
                            inner_t{ "terabyte", "tb", tim::units::terabyte },
                            inner_t{ "petabyte", "pb", tim::units::petabyte },
                            inner_t{ "kibibyte", "kib", tim::units::KiB },
                            inner_t{ "mebibyte", "mib", tim::units::MiB },
                            inner_t{ "gibibyte", "gib", tim::units::GiB },
                            inner_t{ "tebibyte", "tib", tim::units::TiB },
                            inner_t{ "pebibyte", "pib", tim::units::PiB } })
    {
        if(_unit == std::get<0>(itr) || _unit == std::get<1>(itr))
        {
            if(std::get<0>(itr) == "byte")
                return return_type(std::get<0>(itr), std::get<2>(itr));
            return return_type(std::get<1>(itr), std::get<2>(itr));
        }
    }

    std::cerr << "Warning!! No memory unit matching \"" << _unit << "\". Using default..."
              << std::endl;

    return return_type{ "MB", tim::units::megabyte };
}

//--------------------------------------------------------------------------------------//

inline std::tuple<std::string, int64_t>
get_timing_unit(std::string _unit)
{
    using string_t    = std::string;
    using return_type = std::tuple<string_t, int64_t>;
    using inner_t     = std::tuple<string_t, string_t, int64_t>;

    if(_unit.length() == 0)
        return return_type{ "sec", tim::units::sec };

    for(auto& itr : _unit)
        itr = tolower(itr);

    for(const auto& itr : { inner_t{ "ps", "picosecond", tim::units::psec },
                            inner_t{ "ns", "nanosecond", tim::units::nsec },
                            inner_t{ "us", "microsecond", tim::units::usec },
                            inner_t{ "ms", "millisecond", tim::units::msec },
                            inner_t{ "cs", "centisecond", tim::units::csec },
                            inner_t{ "ds", "decisecond", tim::units::dsec },
                            inner_t{ "s", "second", tim::units::sec } })
    {
        if(_unit == std::get<0>(itr) || _unit == std::get<1>(itr) ||
           _unit == (std::get<0>(itr) + "ec") || _unit == (std::get<1>(itr) + "s"))
        {
            return return_type{ std::get<0>(itr) + "ec", std::get<2>(itr) };
        }
    }

    std::cerr << "Warning!! No timing unit matching \"" << _unit << "\". Using default..."
              << std::endl;

    return return_type{ "sec", tim::units::sec };
}

//--------------------------------------------------------------------------------------//

}  // namespace units

//======================================================================================//

}  // namespace tim
