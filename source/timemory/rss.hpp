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

/** \file rss.hpp
 * Resident set size handler
 *
 */

#ifndef rss_hpp_
#define rss_hpp_

#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <cstdint>
#include <algorithm>

#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/macros.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>

#include "timemory/namespace.hpp"

//============================================================================//

#if defined(_UNIX)
#   include <unistd.h>
#   include <sys/resource.h>
#   if defined(_MACOS)
#       include <mach/mach.h>
#   endif
#elif defined(_WINDOWS)
#   include <windows.h>
#   include <stdio.h>
#   include <psapi.h>
#else
#   error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

// RSS - Resident set size (physical memory use, not in swap)

namespace NAME_TIM
{

namespace rss
{

    // Using the SI convention for kilo-, mega-, giga-, and peta-
    // because this is more intuitive
    namespace units
    {
    const int64_t byte     = 1;
    const int64_t kilobyte = 1000*byte;
    const int64_t megabyte = 1000*kilobyte;
    const int64_t gigabyte = 1000*megabyte;
    const int64_t petabyte = 1000*gigabyte;
    const double  Bi       = 1.0;
    const double  KiB      = 1024.0 * Bi;
    const double  MiB      = 1024.0 * KiB;
    const double  GiB      = 1024.0 * MiB;
    const double  PiB      = 1024.0 * GiB;
    }

    /**
     * Returns the peak (maximum so far) resident set size (physical
     * memory use) measured in bytes, or zero if the value cannot be
     * determined on this OS.
     */
    static inline
    int64_t get_peak_rss()
    {
#if defined(_UNIX)
        struct rusage rusage;
        getrusage( RUSAGE_SELF, &rusage );
#   if defined(__APPLE__) && defined(__MACH__)
        return (int64_t) (rusage.ru_maxrss / ((int64_t) units::KiB) * units::kilobyte);
#   else
        return (int64_t) (rusage.ru_maxrss / ((int64_t) units::KiB) * units::megabyte);
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
            nsize = (int64_t) pmc.PeakWorkingSetSize;

        CloseHandle( hProcess );

        return nsize;
#else
        return (int64_t) 0;
#endif
    }

    /**
     * Returns the current resident set size (physical memory use) measured
     * in bytes, or zero if the value cannot be determined on this OS.
     */
    static inline
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
        return (int64_t) (info.resident_size / ((int64_t) units::KiB) * units::kilobyte);

#   else // Linux
        long rss = 0L;
        FILE* fp = fopen("/proc/self/statm", "r");
        if(fp == nullptr)
            return (int64_t) 0L;
        if(fscanf(fp, "%*s%ld", &rss) != 1)
        {
            fclose(fp);
            return (int64_t) 0L;
        }
        fclose(fp);
        return (int64_t) (rss * (int64_t) sysconf( _SC_PAGESIZE) / ((int64_t) units::KiB) *
                          units::kilobyte);
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

    //========================================================================//

    struct usage
    {
        typedef usage           this_type;
        typedef int64_t         size_type;
        size_type               m_peak_rss;
        size_type               m_curr_rss;

        usage()
        : m_peak_rss(0), m_curr_rss(0)
        { }

        usage(size_type minus)
        : m_peak_rss(0), m_curr_rss(0)
        {
            record();
            if(minus > 0)
            {
                if(minus < m_curr_rss)
                    m_curr_rss -= minus;
                else
                    m_curr_rss = 1;

                if(minus < m_peak_rss)
                    m_peak_rss -= minus;
                else
                    m_peak_rss = 1;
            }
        }

        usage(const usage& rhs)
        : m_peak_rss(rhs.m_peak_rss),
          m_curr_rss(rhs.m_curr_rss)
        { }

        usage& operator=(const usage& rhs)
        {
            if(this != &rhs)
            {
                m_peak_rss = rhs.m_peak_rss;
                m_curr_rss = rhs.m_curr_rss;
            }
            return *this;
        }

        void record();
        void record(const usage& rhs);

        friend bool operator<(const this_type& lhs, const this_type& rhs)
        { return lhs.m_peak_rss < rhs.m_peak_rss; }
        friend bool operator==(const this_type& lhs, const this_type& rhs)
        { return lhs.m_peak_rss == rhs.m_peak_rss; }
        friend bool operator!=(const this_type& lhs, const this_type& rhs)
        { return !(lhs.m_peak_rss == rhs.m_peak_rss); }
        friend bool operator>(const this_type& lhs, const this_type& rhs)
        { return rhs.m_peak_rss < lhs.m_peak_rss; }
        friend bool operator<=(const this_type& lhs, const this_type& rhs)
        { return !(lhs > rhs); }
        friend bool operator>=(const this_type& lhs, const this_type& rhs)
        { return !(lhs < rhs); }
        bool operator()(const this_type& rhs) const
        { return *this < rhs; }

        static usage max(const usage& lhs, const usage& rhs)
        {
            usage ret;
            ret.m_curr_rss = ::std::max(lhs.m_curr_rss, rhs.m_curr_rss);
            ret.m_peak_rss = ::std::max(lhs.m_peak_rss, rhs.m_peak_rss);
            return ret;
        }

        static usage min(const usage& lhs, const usage& rhs)
        {
            usage ret;
            ret.m_curr_rss = ::std::min(lhs.m_curr_rss, rhs.m_curr_rss);
            ret.m_peak_rss = ::std::min(lhs.m_peak_rss, rhs.m_peak_rss);
            return ret;
        }

        friend this_type operator-(const this_type& lhs, const this_type& rhs)
        {
            this_type r = lhs;

            if(rhs.m_peak_rss < r.m_peak_rss)
                r.m_peak_rss -= rhs.m_peak_rss;
            else
                r.m_peak_rss = 1;

            if(rhs.m_curr_rss < r.m_curr_rss)
                r.m_curr_rss -= rhs.m_curr_rss;
            else
                r.m_curr_rss = 1;

            return r;
        }

        this_type& operator+=(const this_type& rhs)
        {
            m_peak_rss += rhs.m_peak_rss;
            m_curr_rss += rhs.m_curr_rss;
            return *this;
        }

        this_type& operator-=(const this_type& rhs)
        {
            m_peak_rss -= rhs.m_peak_rss;
            m_curr_rss -= rhs.m_curr_rss;
            return *this;
        }

        double current(int64_t _unit = units::megabyte) const
        {
            return static_cast<double>(m_curr_rss) / _unit;
        }

        double peak(int64_t _unit = units::megabyte) const
        {
            return static_cast<double>(m_peak_rss) / _unit;
        }

        friend std::ostream& operator<<(std::ostream& os, const usage& m)
        {
            using std::setw;
            std::stringstream ss;
            ss.precision(1);
            int _w = 5;
            double _curr = m.m_curr_rss / units::megabyte;
            double _peak = m.m_peak_rss / units::megabyte;

            ss << std::fixed;
            ss << "rss curr|peak = "
               << std::setw(_w) << _curr << "|"
               << std::setw(_w) << _peak
               << " MB";
            os << ss.str();
            return os;
        }

        template <typename Archive> void
        serialize(Archive& ar, const unsigned int /*version*/)
        {
            ar(cereal::make_nvp("current", current()),
               cereal::make_nvp("peak",    peak()));
        }

    };

    //------------------------------------------------------------------------//

    inline void usage::record()
    {
        // everything is kB
        m_curr_rss = std::max(m_curr_rss, get_current_rss());
        m_peak_rss = std::max(m_peak_rss, get_peak_rss());
    }

    //------------------------------------------------------------------------//

    inline void usage::record(const usage& rhs)
    {
        // everything is kB
        m_curr_rss = std::max(m_curr_rss - rhs.m_curr_rss,
                              get_current_rss() - rhs.m_curr_rss);
        m_peak_rss = std::max(m_peak_rss - rhs.m_peak_rss,
                              get_peak_rss() - rhs.m_peak_rss);
    }

    //------------------------------------------------------------------------//

} // namespace rss

} // namespace NAME_TIM

//----------------------------------------------------------------------------//


#endif // rss_hpp_
