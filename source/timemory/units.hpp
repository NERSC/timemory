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

/** \file units.hpp
 * \headerfile units.hpp "timemory/units.hpp"
 * Timing and memory units
 *
 */

#ifndef units_hpp_
#define units_hpp_

#include "timemory/macros.hpp"

#include <ratio>
#include <string>
#include <cstdint>

#if defined(_UNIX)
#   include <unistd.h>
#endif

namespace tim
{

//----------------------------------------------------------------------------//

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
const int64_t kilobyte = 1024 * byte;
const int64_t megabyte = 1024 * kilobyte;
const int64_t gigabyte = 1024 * megabyte;
const int64_t terabyte = 1024 * gigabyte;
const int64_t petabyte = 1024 * terabyte;

const double  Bi       = 1.0;
const double  KiB      = 1024.0 * Bi;
const double  MiB      = 1024.0 * KiB;
const double  GiB      = 1024.0 * MiB;
const double  TiB      = 1024.0 * GiB;
const double  PiB      = 1024.0 * TiB;

#if defined(_UNIX)
const int64_t page_size = sysconf(_SC_PAGESIZE);
#endif

//----------------------------------------------------------------------------//

inline
std::string time_repr(const int64_t& _unit)
{
    std::string _sunit;
    switch (_unit)
    {
        case psec:
            _sunit = "psec";
            break;
        case nsec:
            _sunit = "nsec";
            break;
        case usec:
            _sunit = "usec";
            break;
        case msec:
            _sunit = "msec";
            break;
        case csec:
            _sunit = "csec";
            break;
        case dsec:
            _sunit = "dsec";
            break;
        case sec:
            _sunit = "sec";
            break;
        default:
            _sunit = "UNK";
            break;
    }
    return _sunit;
}

inline
std::string mem_repr(const int64_t& _unit)
{
    std::string _sunit;
    switch (_unit)
    {
        case byte:
            _sunit = "B";
            break;
        case kilobyte:
            _sunit = "KB";
            break;
        case megabyte:
            _sunit = "MB";
            break;
        case gigabyte:
            _sunit = "GB";
            break;
        case terabyte:
            _sunit = "TB";
            break;
        case petabyte:
            _sunit = "PB";
            break;
        default:
            _sunit = "UNK";
            break;
    }
    return _sunit;
}

//----------------------------------------------------------------------------//

} // namespace units

//============================================================================//

} // namespace tim

#endif

