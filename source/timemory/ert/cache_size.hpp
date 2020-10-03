// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

/** \file timemory/ert/cache_size.hpp
 * \headerfile timemory/ert/cache_size.hpp "timemory/ert/cache_size.hpp"
 * Provides routines for getting cache information
 *
 */

#pragma once

#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MACOS)
#    include <sys/sysctl.h>
#endif

namespace tim
{
namespace ert
{
using std::size_t;

//--------------------------------------------------------------------------------------//
//  get the size of the L1 (data), L2, or L3 cache
//
namespace cache_size
{
namespace impl
{
//--------------------------------------------------------------------------------------//

#if defined(_MACOS)

//--------------------------------------------------------------------------------------//

inline size_t
cache_size(const int& level)
{
    // configure sysctl query
    //      L1  ->  hw.l1dcachesize
    //      L2  ->  hw.l2cachesize
    //      L3  ->  hw.l3cachesize
    //
    std::stringstream query;
    query << "hw.l" << level;
    if(level == 1)
        query << "d";
    query << "cachesize";

    size_t line_size        = 0;
    size_t sizeof_line_size = sizeof(line_size);
    sysctlbyname(query.str().c_str(), &line_size, &sizeof_line_size, 0, 0);
    return line_size;
}

//--------------------------------------------------------------------------------------//

#elif defined(_WINDOWS)

//--------------------------------------------------------------------------------------//

inline size_t
cache_size(const int& level)
{
    size_t                                line_size   = 0;
    DWORD                                 buffer_size = 0;
    DWORD                                 i           = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer      = 0;

    GetLogicalProcessorInformation(0, &buffer_size);
    buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*) malloc(buffer_size);
    if(!buffer)
        return static_cast<size_t>(4096);
    GetLogicalProcessorInformation(&buffer[0], &buffer_size);

    for(i = 0; i != buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
    {
        if(buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == level)
        {
            line_size = buffer[i].Cache.Size;
            break;
        }
    }

    free(buffer);
    return line_size;
}

//--------------------------------------------------------------------------------------//

#elif defined(_LINUX)

//--------------------------------------------------------------------------------------//

inline size_t
cache_size(const int& _level)
{
    // L1 has a data and instruction cache, index0 should be data
    auto level = (_level == 1) ? 0 : (_level);
    // location of files
    std::stringstream fpath;
    fpath << "/sys/devices/system/cpu/cpu0/cache/index" << level << '/';

    // files to read
    const std::array<std::string, 3> files(
        { { "number_of_sets", "ways_of_associativity", "coherency_line_size" } });

    uint64_t product = 1;
    for(unsigned i = 0; i < files.size(); ++i)
    {
        std::string   fname = fpath.str() + files[i];
        std::ifstream ifs(fname.c_str());
        if(ifs)
        {
            uint64_t val;
            ifs >> val;
            product *= val;
        }
        else
        {
            throw std::runtime_error("Unable to open file: " + fname);
        }
        ifs.close();
    }
    return (product > 1) ? product : 0;
}

//--------------------------------------------------------------------------------------//

#else

//--------------------------------------------------------------------------------------//

#    warning Unrecognized platform
inline size_t
cache_size()
{
    return 0;
}

//--------------------------------------------------------------------------------------//

#endif

}  // namespace impl

//======================================================================================//

template <size_t Level>
inline size_t
get()
{
    // only enable queries 1, 2, 3
    static_assert(Level > 0 && Level < 4,
                  "Request for cache level that is not supported");

    // avoid multiple queries
    static size_t _value = impl::cache_size(Level);
    return _value;
}

//--------------------------------------------------------------------------------------//

inline size_t
get(const int& _level)
{
    // only enable queries 1, 2, 3
    if(_level < 1 || _level > 3)
    {
        std::stringstream ss;
        ss << "tim::ert::cache_size::get(" << _level << ") :: "
           << "Requesting invalid cache level";
        throw std::runtime_error(ss.str());
    }
    // avoid multiple queries
    static std::vector<size_t> _values(
        { { impl::cache_size(1), impl::cache_size(2), impl::cache_size(3) } });
    return _values.at(_level - 1);
}

//--------------------------------------------------------------------------------------//

inline size_t
get_max()
{
    // this is useful for system like KNL that do not have L3 cache
    for(auto level : { 3, 2, 1 })
    {
        try
        {
            size_t sz = impl::cache_size(level);
            // if this succeeded, we can return the value
            return sz;
        } catch(...)
        {
            continue;
        }
    }
    return 0;
}

//--------------------------------------------------------------------------------------//

}  // namespace cache_size
}  // namespace ert
}  // namespace tim
