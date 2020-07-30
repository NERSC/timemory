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

/**
 * \file timemory/components/rusage/backends.hpp
 * \brief Implementation of the rusage functions/utilities
 */

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#if defined(_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(_MACOS)
#        include <libproc.h>
#        include <mach/mach.h>
#    endif
#elif defined(_WINDOWS)
#    if !defined(NOMINMAX)
#        define NOMINMAX
#    endif
// currently, this is causing a bunch of errors, need to disable
// #    include <psapi.h>
#    include <stdio.h>
#    include <windows.h>
#else
#    error "Cannot define get_peak_rss() or get_page_rss() for an unknown OS."
#endif

//======================================================================================//
//
namespace tim
{
//
int64_t
get_bytes_read();
int64_t
get_bytes_written();
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_bytes_read()
{
#if defined(_MACOS)
    rusage_info_current rusage;
    if(proc_pid_rusage(process::get_target_id(), RUSAGE_INFO_CURRENT, (void**) &rusage) ==
       0)
        return rusage.ri_diskio_bytesread;
#elif defined(_LINUX)

#    if defined(TIMEM_DEBUG)
    if(get_env("TIMEM_DEBUG", false))
        printf("[%s@%s:%i]> using pid %li\n", __func__, __FILE__, __LINE__,
               (long int) process::get_target_id());
#    endif

    std::stringstream fio;
    fio << "/proc/" << process::get_target_id() << "/io";
    std::string   label = "";
    int64_t       value = 0;
    std::ifstream ifs(fio.str().c_str());
    if(ifs)
    {
        static constexpr int max_lines = 1;
        for(int i = 0; i < max_lines && !ifs.eof(); ++i)
        {
            ifs >> label;
            ifs >> value;
            // if(label.find("read_bytes") != std::string::npos)
            //    return value;
            // if(label.find("rchar") != std::string::npos)
            //    return value;
        }
        if(!ifs.eof())
            return value;
    }
#endif
    return 0;
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_bytes_written()
{
#if defined(_MACOS)
    rusage_info_current rusage;
    if(proc_pid_rusage(process::get_target_id(), RUSAGE_INFO_CURRENT, (void**) &rusage) ==
       0)
        return rusage.ri_diskio_byteswritten;
#elif defined(_LINUX)

#    if defined(TIMEM_DEBUG)
    if(get_env("TIMEM_DEBUG", false))
        printf("[%s@%s:%i]> using pid %li\n", __func__, __FILE__, __LINE__,
               (long int) process::get_target_id());
#    endif

    std::stringstream fio;
    fio << "/proc/" << process::get_target_id() << "/io";
    std::string   label = "";
    int64_t       value = 0;
    std::ifstream ifs(fio.str().c_str());
    if(ifs)
    {
        static constexpr int max_lines = 2;
        for(int i = 0; i < max_lines && !ifs.eof(); ++i)
        {
            ifs >> label;
            ifs >> value;
            // if(label.find("write_bytes") != std::string::npos)
            //     return value;
            // if(label.find("wchar") != std::string::npos)
            //    return value;
        }
        if(!ifs.eof())
            return value;
    }
#endif
    return 0;
}
//
//--------------------------------------------------------------------------------------//
//
