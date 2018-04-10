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
#include <fstream>

#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/macros.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>

#include "timemory/macros.hpp"

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

namespace tim
{

namespace rss
{

//----------------------------------------------------------------------------//
// Using the SI convention for kilo-, mega-, giga-, and peta-
// because this is more intuitive
namespace units
{
    const int64_t byte     = 1;
    const int64_t kilobyte = 1024*byte;
    const int64_t megabyte = 1024*kilobyte;
    const int64_t gigabyte = 1024*megabyte;
    const int64_t petabyte = 1024*gigabyte;
    const double  Bi       = 1.0;
    const double  KiB      = 1024.0 * Bi;
    const double  MiB      = 1024.0 * KiB;
    const double  GiB      = 1024.0 * MiB;
    const double  PiB      = 1024.0 * GiB;
#   if defined(_UNIX)
    const int64_t page_size = sysconf(_SC_PAGESIZE);
#   endif
}

//----------------------------------------------------------------------------//

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
    // Darwin reports in bytes
    return (int64_t) (rusage.ru_maxrss);
#   else
    // Linux reports in kilobytes
    return (int64_t) (rusage.ru_maxrss * units::kilobyte);
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

//----------------------------------------------------------------------------//

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

//============================================================================//

class usage; // declaration for usage_format

//============================================================================//

class tim_api usage_format
{
public:
    typedef std::stringstream   stringstream_t;
    typedef std::string         string_t;
    typedef int16_t             size_type;

public:
    usage_format(string_t _format = "RSS {curr,peak} : (%C|%M) [MB]",
                 int64_t _unit = units::megabyte,
                 size_type _prec = 1,
                 size_type _width = 5,
                 char _curr = 'C',
                 char _peak = 'M')
    : m_precision(_prec),
      m_width(_width),
      m_unit(_unit),
      m_curr(""),
      m_peak(""),
      m_format(_format)
    {
        stringstream_t _ss_c, _ss_p;
        _ss_c << "%" << _curr;
        _ss_p << "%" << _peak;
        m_curr = _ss_c.str();
        m_peak = _ss_p.str();
    }

    string_t operator()(const usage* m) const;

protected:
    size_type   m_precision;
    size_type   m_width;
    int64_t     m_unit;
    string_t    m_curr;
    string_t    m_peak;
    string_t    m_format;
};

//============================================================================//

class tim_api usage
{
public:
    typedef usage                           this_type;
    typedef int64_t                         size_type;
    typedef std::shared_ptr<usage_format>   usage_format_t;

public:
    usage(usage_format_t _fmt = usage_format_t())
    : m_curr_rss(0), m_peak_rss(0), m_format(_fmt)
    { }

    usage(size_type minus, usage_format_t _fmt = usage_format_t())
    : m_curr_rss(0), m_peak_rss(0), m_format(_fmt)
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

    usage(size_type _curr, size_type _peak,
          usage_format_t _fmt = usage_format_t())
    : m_curr_rss(_curr), m_peak_rss(_peak), m_format(_fmt)
    { }

    usage(const usage& rhs)
    : m_curr_rss(rhs.m_curr_rss),
      m_peak_rss(rhs.m_peak_rss),
      m_format(rhs.m_format)
    { }

    usage& operator=(const usage& rhs)
    {
        if(this != &rhs)
        {
            m_curr_rss = rhs.m_curr_rss;
            m_peak_rss = rhs.m_peak_rss;
            m_format = rhs.m_format;
        }
        return *this;
    }

public:
    void set_format(const usage_format& _format);
    void set_format(usage_format_t _format);
    usage_format_t format() const;

    void record();
    void record(const usage& rhs);

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

    double current(int64_t _unit = units::megabyte) const
    {
        return static_cast<double>(m_curr_rss) / _unit;
    }

    double peak(int64_t _unit = units::megabyte) const
    {
        return static_cast<double>(m_peak_rss) / _unit;
    }

    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(cereal::make_nvp("current", current()),
           cereal::make_nvp("peak",    peak()));
    }

    std::string str() const
    {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

public:
    //------------------------------------------------------------------------//
    //          operator <
    //------------------------------------------------------------------------//
    friend bool operator<(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.m_peak_rss == rhs.m_peak_rss)
                ? (lhs.m_curr_rss < rhs.m_curr_rss)
                : (lhs.m_peak_rss < rhs.m_peak_rss);
    }
    //------------------------------------------------------------------------//
    //          operator ==
    //------------------------------------------------------------------------//
    friend bool operator==(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.m_peak_rss == rhs.m_peak_rss) &&
                (lhs.m_curr_rss == rhs.m_curr_rss);
    }
    //------------------------------------------------------------------------//
    //          operator !=
    //------------------------------------------------------------------------//
    friend bool operator!=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs == rhs);
    }
    //------------------------------------------------------------------------//
    //          operator >
    //------------------------------------------------------------------------//
    friend bool operator>(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.m_peak_rss == rhs.m_peak_rss)
                ? (lhs.m_curr_rss > rhs.m_curr_rss)
                : (lhs.m_peak_rss > rhs.m_peak_rss);
    }
    //------------------------------------------------------------------------//
    //          operator <=
    //------------------------------------------------------------------------//
    friend bool operator<=(const this_type& lhs, const this_type& rhs)
    { return !(lhs > rhs); }
    //------------------------------------------------------------------------//
    //          operator >=
    //------------------------------------------------------------------------//
    friend bool operator>=(const this_type& lhs, const this_type& rhs)
    { return !(lhs < rhs); }
    //------------------------------------------------------------------------//
    //          operator ()
    //------------------------------------------------------------------------//
    bool operator()(const this_type& rhs) const
    { return (*this < rhs); }
    //------------------------------------------------------------------------//
    //          operator +
    //------------------------------------------------------------------------//
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type r = lhs;
        r.m_curr_rss += rhs.m_curr_rss;
        r.m_peak_rss += rhs.m_peak_rss;
        return r;
    }
    //------------------------------------------------------------------------//
    //          operator -
    //------------------------------------------------------------------------//
    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type r = lhs;
        r.m_curr_rss -= rhs.m_curr_rss;
        r.m_peak_rss -= rhs.m_peak_rss;
        return r;
    }
    //------------------------------------------------------------------------//
    //          operator +=
    //------------------------------------------------------------------------//
    this_type& operator+=(const this_type& rhs)
    {
        m_curr_rss += rhs.m_curr_rss;
        m_peak_rss += rhs.m_peak_rss;
        return *this;
    }
    //------------------------------------------------------------------------//
    //          operator -=
    //------------------------------------------------------------------------//
    this_type& operator-=(const this_type& rhs)
    {
        m_curr_rss -= rhs.m_curr_rss;
        m_peak_rss -= rhs.m_peak_rss;
        return *this;
    }
    //------------------------------------------------------------------------//
    //          operator <<
    //------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const usage& m)
    {
        usage_format _format = (m.format().get())
                               ? (*(m.format().get())) : usage_format();
        os << _format(&m);
        return os;
    }

protected:
    size_type       m_curr_rss;
    size_type       m_peak_rss;
    usage_format_t  m_format;
};

//----------------------------------------------------------------------------//
inline
void usage::set_format(const usage_format& _format)
{
    m_format = usage_format_t(new usage_format(_format));
}
//----------------------------------------------------------------------------//
inline
void usage::set_format(usage_format_t _format)
{
    m_format = _format;
}
//----------------------------------------------------------------------------//
inline
usage::usage_format_t usage::format() const
{
    return m_format;
}
//----------------------------------------------------------------------------//
inline void usage::record()
{
    // everything is kB
    m_curr_rss = std::max(m_curr_rss, get_current_rss());
    m_peak_rss = std::max(m_peak_rss, get_peak_rss());
}
//----------------------------------------------------------------------------//
inline void usage::record(const usage& rhs)
{
    // everything is kB
    m_curr_rss = std::max(m_curr_rss - rhs.m_curr_rss,
                          get_current_rss() - rhs.m_curr_rss);
    m_peak_rss = std::max(m_peak_rss - rhs.m_peak_rss,
                          get_peak_rss() - rhs.m_peak_rss);
}
//----------------------------------------------------------------------------//
inline
usage_format::string_t usage_format::operator()(const usage* m) const
{
    using std::setw;
    stringstream_t ss_c;
    stringstream_t ss_p;

    ss_c.precision(m_precision);
    ss_p.precision(m_precision);

    ss_c << std::fixed << setw(m_width) << m->current(m_unit);
    ss_p << std::fixed << setw(m_width) << m->peak(m_unit);

    string_t _str = m_format;

    /*
    int _n = 0;
    auto _debug = [&] ()
    {
        std::cout << "(" << _n++ <<") STR: \"" << _str
                  << "\", m_curr = \"" << m_curr
                  << "\", m_peak = \"" << m_peak << "\"...\n" << std::endl;
    };
    _debug();
    */

    while(_str.find(m_curr) != string_t::npos)
        _str = _str.replace(_str.find(m_curr), m_curr.length(), ss_c.str().c_str());

    while(_str.find(m_peak) != string_t::npos)
        _str = _str.replace(_str.find(m_peak), m_peak.length(), ss_p.str().c_str());


    return _str;
}
//----------------------------------------------------------------------------//


} // namespace rss

} // namespace tim

//----------------------------------------------------------------------------//


#endif // rss_hpp_
