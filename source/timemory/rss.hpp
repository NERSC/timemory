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
 * \headerfile rss.hpp "timemory/rss.hpp"
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

#include "timemory/macros.hpp"
#include "timemory/formatters.hpp"
#include "timemory/serializer.hpp"

//============================================================================//

#if defined(_UNIX)
#   include <unistd.h>
#   include <sys/resource.h>
#   if defined(_MACOS)
#       include <mach/mach.h>
#   endif
#elif defined(_WINDOWS)
#   if !defined(NOMINMAX)
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <stdio.h>
#   include <psapi.h>
#else
#   error "Cannot define get_peak_rss() or get_current_rss() for an unknown OS."
#endif

// RSS - Resident set size (physical memory use, not in swap)

namespace tim
{

namespace rss
{

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

class tim_api usage
{
public:
    typedef usage                           this_type;
    typedef int64_t                         size_type;
    typedef format::rss                     format_type;
    typedef std::shared_ptr<format_type>    usage_format_t;

public:
    //------------------------------------------------------------------------//
    //      Default constructor variants with usage_format_t
    //------------------------------------------------------------------------//
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

    //------------------------------------------------------------------------//
    //      Constructor variants with format_type
    //------------------------------------------------------------------------//
    usage(format_type _fmt)
    : m_curr_rss(0), m_peak_rss(0),
      m_format(usage_format_t(new format_type(_fmt)))
    { }

    usage(size_type minus, format_type _fmt)
    : m_curr_rss(0), m_peak_rss(0),
      m_format(usage_format_t(new format_type(_fmt)))
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

    usage(size_type _curr, size_type _peak, format_type _fmt)
    : m_curr_rss(_curr), m_peak_rss(_peak),
      m_format(usage_format_t(new format_type(_fmt)))
    { }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    usage(const usage& rhs)
    : m_curr_rss(rhs.m_curr_rss),
      m_peak_rss(rhs.m_peak_rss),
      m_format(usage_format_t())
    {
        if(rhs.format().get())
            m_format = usage_format_t(new format_type(*(rhs.format().get())));
    }

    usage& operator=(const usage& rhs)
    {
        if(this != &rhs)
        {
            m_curr_rss = rhs.m_curr_rss;
            m_peak_rss = rhs.m_peak_rss;
            if(!m_format.get())
                m_format = usage_format_t(new format_type());
            if(rhs.m_format.get())
                *m_format = *(rhs.m_format.get());

            m_format = rhs.m_format;
        }
        return *this;
    }

public:
    void set_format(const format_type& _format);
    void set_format(usage_format_t _format);
    usage_format_t format() const;

    void record();
    void record(const usage& rhs);

    void reset()
    {
        m_curr_rss = 0;
        m_peak_rss = 0;
    }

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

    template <typename _Tp = double>
    _Tp current(int64_t _unit = units::megabyte) const
    {
        return static_cast<_Tp>(m_curr_rss) / _unit;
    }

    template <typename _Tp = double>
    _Tp peak(int64_t _unit = units::megabyte) const
    {
        return static_cast<_Tp>(m_peak_rss) / _unit;
    }

    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(serializer::make_nvp("current", current()),
           serializer::make_nvp("peak",    peak()));
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
        format_type _format = (m.format().get())
                               ? (*(m.format().get())) : format_type();
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
void usage::set_format(const format_type& _format)
{
    m_format = usage_format_t(new format_type(_format));
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
    // everything is bytes
    m_curr_rss = std::max(m_curr_rss, get_current_rss());
    m_peak_rss = std::max(m_peak_rss, get_peak_rss());
}
//----------------------------------------------------------------------------//
inline void usage::record(const usage& rhs)
{
    // everything is bytes
    m_curr_rss = std::max(m_curr_rss - rhs.m_curr_rss,
                          get_current_rss() - rhs.m_curr_rss);
    m_peak_rss = std::max(m_peak_rss - rhs.m_peak_rss,
                          get_peak_rss() - rhs.m_peak_rss);
}
//----------------------------------------------------------------------------//

//============================================================================//
//
//  Class for handling the RSS memory usage (total and self)
//
//============================================================================//

class tim_api usage_delta
{
public:
    typedef usage_delta                     this_type;
    typedef usage                           base_type;
    typedef format::rss                     format_type;
    typedef std::shared_ptr<format_type>    usage_format_t;

public:
    //------------------------------------------------------------------------//
    //      Default constructor variants with usage_format_t
    //------------------------------------------------------------------------//
    usage_delta(usage_format_t _fmt = usage_format_t())
    : m_format(_fmt)
    { }

    //------------------------------------------------------------------------//
    //      Constructor variants with format_type
    //------------------------------------------------------------------------//
    usage_delta(format_type _fmt)
    : m_format(usage_format_t(new format_type(_fmt)))
    { }

    //------------------------------------------------------------------------//
    //      Copy construct
    //------------------------------------------------------------------------//
    usage_delta(const usage_delta& rhs)
    : m_rss_tot(rhs.m_rss_tot),
      m_rss_self(rhs.m_rss_self),
      m_rss_tmp(rhs.m_rss_tmp),
      m_rss_tot_min(rhs.m_rss_tot_min),
      m_rss_self_min(rhs.m_rss_self_min),
      m_format(usage_format_t())
    {
        if(rhs.format().get())
            m_format = usage_format_t(new format_type(*(rhs.format().get())));
    }

public:
    void set_format(const format_type& _format);
    void set_format(usage_format_t _format);
    usage_format_t format() const;

    inline void init()
    {
        m_rss_tmp.record();
    }

    inline void record()
    {
        m_rss_self.record(m_rss_tmp);
        m_rss_tot.record();
    }

    void reset()
    {
        m_rss_self.reset();
        m_rss_tmp.reset();
        m_rss_tot.reset();
        m_rss_self_min.reset();
        m_rss_tot_min.reset();
    }

    base_type& total() { return m_rss_tot; }
    base_type& self()  { return m_rss_self; }

    const base_type& total() const { return m_rss_tot; }
    const base_type& self() const { return m_rss_self; }

    base_type& total_min() { return m_rss_tot_min; }
    base_type& self_min()  { return m_rss_self_min; }

    const base_type& total_min() const { return m_rss_tot_min; }
    const base_type& self_min() const { return m_rss_self_min; }

    void max(const this_type& rhs)
    {
        m_rss_tot  = tim::rss::usage::max(m_rss_tot, rhs.total());
        m_rss_self = tim::rss::usage::max(m_rss_self, rhs.self());
        m_rss_tot_min = tim::rss::usage::min(m_rss_tot, rhs.total());
        m_rss_self_min = tim::rss::usage::min(m_rss_self, rhs.self());
    }

    std::string str() const
    {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

public:
    //------------------------------------------------------------------------//
    //          operator +=
    //------------------------------------------------------------------------//
    inline this_type& operator+=(const base_type& rhs)
    {
        m_rss_tot += rhs;
        m_rss_tot_min += rhs;
        return *this;
    }
    //------------------------------------------------------------------------//
    //          operator -=
    //------------------------------------------------------------------------//
    inline this_type& operator-=(const base_type& rhs)
    {
        m_rss_tot -= rhs;
        m_rss_tot_min -= rhs;
        return *this;
    }
    //------------------------------------------------------------------------//
    //          operator =
    //------------------------------------------------------------------------//
    this_type& operator=(const this_type& rhs)
    {
        if(this != &rhs)
        {
            if(!m_format.get())
                m_format = usage_format_t(new format_type(format::timer::default_rss_format()));
            if(rhs.m_format.get())
                *m_format = *(rhs.m_format.get());
            m_rss_tot = rhs.m_rss_tot;
            m_rss_self = rhs.m_rss_self;
            m_rss_tmp = rhs.m_rss_tmp;
            m_rss_tot_min = rhs.m_rss_tot_min;
            m_rss_self_min = rhs.m_rss_self_min;
        }
        return *this;
    }
    //------------------------------------------------------------------------//
    //          operator <<
    //------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const this_type& m)
    {
        format_type _format = (m.format().get())
                               ? (*(m.format().get()))
                               : format::timer::default_rss_format();
        os << _format(&m);
        return os;
    }

protected:
    // memory usage
    base_type       m_rss_tot;
    base_type       m_rss_self;
    base_type       m_rss_tmp;
    base_type       m_rss_tot_min;
    base_type       m_rss_self_min;
    usage_format_t  m_format;

};

//----------------------------------------------------------------------------//
inline
void usage_delta::set_format(const format_type& _format)
{
    m_format = usage_format_t(new format_type(_format));
}
//----------------------------------------------------------------------------//
inline
void usage_delta::set_format(usage_format_t _format)
{
    m_format = _format;
}
//----------------------------------------------------------------------------//
inline
usage_delta::usage_format_t usage_delta::format() const
{
    return m_format;
}
//----------------------------------------------------------------------------//

} // namespace rss

} // namespace tim

//----------------------------------------------------------------------------//


#endif // rss_hpp_
