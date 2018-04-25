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

/** \file formatters.hpp
 * \headerfile formatters.hpp "timemory/formatters.hpp"
 * Format of timers and RSS usage output
 *
 */

#ifndef formatters_hpp_
#define formatters_hpp_

#include "timemory/macros.hpp"
#include "timemory/units.hpp"

#include <string>
#include <sstream>
#include <ios>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <utility>

#if defined(_UNIX)
#   include <unistd.h>
#endif

namespace tim
{

//============================================================================//
namespace internal
{
    class base_timer; // declaration for format::timer
    class base_rss_usage; // declaration for format::timer
}
namespace rss { class usage; } // declaration for format::rss
//============================================================================//

//----------------------------------------------------------------------------//

namespace format
{

//============================================================================//

class tim_api base_formatter
{
public:
    typedef std::stringstream           stringstream_t;
    typedef std::string                 string_t;
    typedef int16_t                     size_type;
    typedef int64_t                     unit_type;

public:
    // public constructors
    base_formatter(string_t _prefix,
                   string_t _suffix,
                   string_t _format,
                   unit_type _unit,
                   bool _align_width,
                   size_type _prec,
                   size_type _width)
    : m_align_width(_align_width),
      m_precision(_prec),
      m_width(_width),
      m_unit(_unit),
      m_prefix(_prefix),
      m_suffix(_suffix),
      m_format(_format)
    {}

    virtual ~base_formatter() { }

public:
    // public member functions
    void set_prefix(const string_t& _val) { m_prefix = _val; }
    void set_suffix(const string_t& _val) { m_suffix = _val; }
    void set_format(const string_t& _val) { m_format = _val; }
    void set_unit(const unit_type& _val) { m_unit = _val; }
    void set_precision(const size_type& _val) { m_precision = _val; }
    void set_width(const size_type& _val) { m_width = _val; }

    const string_t& prefix() const { return m_prefix; }
    const string_t& suffix() const { return m_suffix; }
    const string_t& format() const { return m_format; }
    const unit_type& unit() const { return m_unit; }
    const size_type& precision() const { return m_precision; }
    const size_type& width() const { return m_width; }

    void set_use_align_width(bool _val) { m_align_width = _val; }

protected:
    // protected member functions
    virtual string_t compose() const = 0;

protected:
    // protected member variables
    bool        m_align_width;
    size_type   m_precision;
    size_type   m_width;
    unit_type   m_unit;
    string_t    m_prefix;
    string_t    m_suffix;
    string_t    m_format;
};

//============================================================================//

class tim_api rss : public base_formatter
{
public:
    enum class field
    {
        current,
        peak,
        total_curr,
        total_peak,
        self_curr,
        self_peak,
        memory_unit
    };

    typedef std::pair<string_t, field>  field_pair_t;
    typedef std::vector<field_pair_t>   field_list_t;

public:
    rss(string_t _prefix = "",
        string_t _format = get_default_format(),
        unit_type _unit = get_default_unit(),
        bool _align_width = false)
    : base_formatter(_prefix, "", _format, _unit, _align_width,
                     get_default_precision(),
                     get_default_width())
    { }

    virtual ~rss() { }

public:
    // public member functions
    string_t operator()(const tim::rss::usage* m) const;
    string_t operator()(const tim::internal::base_rss_usage* m,
                        const string_t& base_string = "") const;
    rss* copy_from(const rss* rhs);

public:
    // public static functions
    static void propose_default_width(size_type);

    static void set_default_format(const string_t& _val) { f_default_format = _val; }
    static void set_default_unit(const unit_type& _val) { f_default_unit = _val; }
    static void set_default_precision(const size_type& _val) { f_default_precision = _val; }
    static void set_default_width(const size_type& _val) { f_default_width = _val; }

    static const string_t& get_default_format() { return f_default_format; }
    static const unit_type& get_default_unit() { return f_default_unit; }
    static const size_type& get_default_precision() { return f_default_precision; }
    static const size_type& get_default_width() { return f_default_width; }

    static void set_default(const rss& rhs);
    static rss  get_default();

protected:
    // protected member functions
    virtual string_t compose() const final;

private:
    // private static members
    static string_t     f_default_format;
    static size_type    f_default_precision;
    static size_type    f_default_width;
    static unit_type    f_default_unit;
    static field_list_t f_field_list;
};

//============================================================================//

class tim_api timer : public base_formatter
{
public:
    enum class field
    {
        wall,
        user,
        system,
        cpu,
        percent,
        rss,
        laps,
        timing_unit,
    };

    typedef std::pair<string_t, field>  field_pair_t;
    typedef std::vector<field_pair_t>   field_list_t;
    typedef rss                         rss_format_t;

public:
    // public constructors
    timer(string_t _prefix = "",
          string_t _format = get_default_format(),
          unit_type _unit = get_default_unit(),
          rss_format_t _rss_format = get_default_rss_format(),
          bool _align_width = false)
    : base_formatter(_prefix, "", _format, _unit, _align_width,
                     get_default_precision(),
                     get_default_width()),
      m_rss_format(_rss_format)
    { }

    virtual ~timer() { }

public:
    // public member functions
    string_t operator()(const internal::base_timer* m) const;
    void set_rss_format(const rss_format_t& _val) { m_rss_format = _val; }
    const rss_format_t& rss_format() const { return m_rss_format; }
    timer* copy_from(const timer* rhs);

public:
    // public static functions
    static void propose_default_width(size_type);

    static void set_default_format(const string_t& _val) { f_default_format = _val; }
    static void set_default_unit(const unit_type& _val) { f_default_unit = _val; }
    static void set_default_precision(const size_type& _val) { f_default_precision = _val; }
    static void set_default_width(const size_type& _val) { f_default_width = _val; }
    static void set_default_rss_format(const rss_format_t& _val) { f_default_rss_format = _val; }

    static const string_t&  get_default_format() { return f_default_format; }
    static const unit_type& get_default_unit() { return f_default_unit; }
    static const size_type& get_default_precision() { return f_default_precision; }
    static const size_type& get_default_width() { return f_default_width; }
    static const rss_format_t& get_default_rss_format() { return f_default_rss_format; }

    static void  set_default(const timer& rhs);
    static timer get_default();

protected:
    // protected member functions
    virtual string_t compose() const final;

protected:
    // protected member variables
    rss_format_t        m_rss_format;

private:
    // private static members
    static string_t     f_default_format;
    static size_type    f_default_precision;
    static size_type    f_default_width;
    static unit_type    f_default_unit;
    static rss_format_t f_default_rss_format;
    static field_list_t f_field_list;

};

//============================================================================//

} // namespace format

//----------------------------------------------------------------------------//

} // namespace tim

//----------------------------------------------------------------------------//

#endif

