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

/** \file rss.cpp
 * Resident set size handler
 *
 */

#include "timemory/rss.hpp"

//======================================================================================//

namespace tim
{

//======================================================================================//

namespace rss
{

//======================================================================================//

int64_t get_peak_rss()
{
#if defined(_UNIX)
    struct rusage _self_rusage, _child_rusage;
    getrusage( RUSAGE_SELF, &_self_rusage  );
    getrusage( RUSAGE_CHILDREN, &_child_rusage );

// Darwin reports in bytes, Linux reports in kilobytes
#if defined(_MACOS)
    int64_t _units = 1;
#else
    int64_t _units = units::kilobyte;
#endif

    return (int64_t) (_units * (_self_rusage.ru_maxrss + _child_rusage.ru_maxrss));

#elif defined(_WINDOWS)
    DWORD processID = GetCurrentProcessId();
    HANDLE hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    // Print the process identifier.
    // printf( "\nProcess ID: %u\n", processID );
    // Print information about the memory usage of the process.
    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION |
                           PROCESS_VM_READ,
                           TRUE, processID);
    if (NULL == hProcess)
        return (int64_t) 0;

    int64_t nsize = 0;
    if(GetProcessMemoryInfo( hProcess, &pmc, sizeof(pmc)))
        nsize = (int64_t) pmc.PeakWorkingSetSize;

    CloseHandle( hProcess );

    return nsize;
#else
    return (int64_t) 0;
#endif
}

//======================================================================================//

int64_t get_current_rss()
{
#if defined(_UNIX)
#   if defined(_MACOS)
    // OSX
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if(task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t) &info, &infoCount) != KERN_SUCCESS)
        return (int64_t) 0L;      /* Can't access? */
    // Darwin reports in bytes
    return (int64_t) (info.resident_size);

#   else // Linux

    int64_t rss = 0;
    FILE* fp = fopen("/proc/self/statm", "r");
    if(fp && fscanf(fp, "%*s%ld", &rss) == 1)
    {
        fclose(fp);
        return (int64_t) (rss * units::page_size);
    }

    if(fp)
        fclose(fp);

    return (int64_t) (0);

#   endif
#elif defined(_WINDOWS)
    DWORD processID = GetCurrentProcessId();
    HANDLE hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    // Print the process identifier.
    // printf( "\nProcess ID: %u\n", processID );
    // Print information about the memory usage of the process.
    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION |
                               PROCESS_VM_READ,
                           FALSE, processID);
    if (NULL == hProcess)
        return (int64_t) 0;

    int64_t nsize = 0;
    if(GetProcessMemoryInfo( hProcess, &pmc, sizeof(pmc)))
        nsize = (int64_t) pmc.WorkingSetSize;

    CloseHandle( hProcess );

    return nsize;
#else
    return (int64_t) 0;
#endif
}

//======================================================================================//

} // namespace rss

//======================================================================================//

} // namespace tim

//======================================================================================//
