//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
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

/** \file formatters.hpp
 * \headerfile formatters.hpp "timemory/formatters.hpp"
 * Format of timers and RSS usage output
 *
 */

#pragma once

#include "timemory/macros.hpp"
#include "timemory/string.hpp"
#include "timemory/units.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_UNIX)
#    include <unistd.h>
#endif

#define BACKWARD_COMPAT_SET(type, func)                                                  \
    static void set_##func(const type& _v) { func(_v); }

#define BACKWARD_COMPAT_GET(type, func)                                                  \
    static const type& get_##func() { return func(); }

namespace tim
{
//======================================================================================//

namespace internal
{
class base_timer;  // declaration for format::timer
}
// class base_usage;

//======================================================================================//

namespace format
{
//======================================================================================//

typedef std::tuple<int16_t, int16_t, intmax_t, tim::string, bool> core_tuple_t;

//======================================================================================//

tim_api class core_formatter
{
public:
    typedef core_tuple_t base_type;
    typedef tim::string  string_t;
    typedef int16_t      size_type;
    typedef intmax_t     unit_type;

public:
    core_formatter(size_type, size_type, unit_type, string_t, bool);

public:
    // public member functions
    void precision(const size_type& _val) { std::get<0>(m_data) = _val; }
    void width(const size_type& _val) { std::get<1>(m_data) = _val; }
    void unit(const unit_type& _val) { std::get<2>(m_data) = _val; }
    void format(const string_t& _val) { std::get<3>(m_data) = _val; }
    void fixed(const bool& _val) { std::get<4>(m_data) = _val; }
    void scientific(const bool& _val) { std::get<4>(m_data) = !(_val); }

    size_type& precision() { return std::get<0>(m_data); }
    size_type& width() { return std::get<1>(m_data); }
    unit_type& unit() { return std::get<2>(m_data); }
    string_t&  format() { return std::get<3>(m_data); }
    bool&      fixed() { return std::get<4>(m_data); }

    const size_type& precision() const { return std::get<0>(m_data); }
    const size_type& width() const { return std::get<1>(m_data); }
    const unit_type& unit() const { return std::get<2>(m_data); }
    const string_t&  format() const { return std::get<3>(m_data); }
    const bool&      fixed() const { return std::get<4>(m_data); }
    bool             scientific() const { return !(std::get<4>(m_data)); }

protected:
    core_tuple_t m_data;
};

//======================================================================================//

template <typename _Tp>
class formatter : public core_formatter
{
public:
    typedef std::stringstream          stringstream_t;
    typedef core_formatter             base_type;
    typedef core_formatter             core_type;
    typedef std::pair<string_t, int>   field_pair_t;
    typedef std::vector<field_pair_t>  field_list_t;
    typedef std::stack<core_formatter> storage_type;

public:
    // public constructors
    // public constructors
    formatter(string_t _prefix, string_t _suffix, string_t _format, unit_type _unit,
              bool _align_width, size_type _prec, size_type _width, bool _fixed = true);

    ~formatter() {}

public:
    // public member functions
    void align_width(const bool& _val) { m_align_width = _val; }
    void prefix(const string_t& _val) { m_prefix = _val; }
    void suffix(const string_t& _val) { m_suffix = _val; }

    bool&     align_width() { return m_align_width; }
    string_t& prefix() { return m_prefix; }
    string_t& suffix() { return m_suffix; }

    const bool&     align_width() const { return m_align_width; }
    const string_t& prefix() const { return m_prefix; }
    const string_t& suffix() const { return m_suffix; }

    string_t operator()(const _Tp* m) const;
    string_t operator()(const string_t& = "") const;
    _Tp*     copy_from(const _Tp* rhs);

    // public static functions
    static void propose_default_width(size_type _w) { m_width = std::max(m_width, _w); }

    /*
    static void default_precision(const size_type& _v) { f_current().precision() = _v; }
    static void default_width(const size_type& _v) { f_current().width() = _v; }
    static void default_unit(const unit_type& _v) { f_current().unit() = _v; }
    static void default_format(const string_t& _v) { f_current().format() = _v; }
    static void default_fixed(const bool& _v) { f_current().fixed() = _v; }
    static void default_scientific(const bool& _v) { f_current().fixed() = !(_v); }

    // defines set_<function>
    BACKWARD_COMPAT_SET(size_type, default_precision)
    BACKWARD_COMPAT_SET(size_type, default_width)
    BACKWARD_COMPAT_SET(unit_type, default_unit)
    BACKWARD_COMPAT_SET(string_t, default_format)
    BACKWARD_COMPAT_SET(bool, default_fixed)

    static const size_type& default_precision() { return f_current().precision(); }
    static const size_type& default_width() { return f_current().width(); }
    static const unit_type& default_unit() { return f_current().unit(); }
    static const string_t&  default_format() { return f_current().format(); }
    static const bool&      default_fixed() { return f_current().fixed(); }
    static bool             default_scientific() { return !(f_current().fixed()); }

    // defines get_<function>
    BACKWARD_COMPAT_GET(size_type, default_precision)
    BACKWARD_COMPAT_GET(size_type, default_width)
    BACKWARD_COMPAT_GET(unit_type, default_unit)
    BACKWARD_COMPAT_GET(string_t, default_format)
    BACKWARD_COMPAT_GET(bool, default_fixed)
    */
    // static void set_default(const _Tp& rhs);
    // static _Tp  get_default();

    static void push();
    static void pop();

protected:
    // protected member functions
    virtual string_t compose() const final;

protected:
    // protected member variables
    bool     m_align_width;
    bool     m_fixed;
    string_t m_prefix;
    string_t m_suffix;

    static size_type m_width;

private:
    // private static members
    static field_list_t get_field_list();
    // static core_formatter& f_current();
    // static storage_type&   f_history();
};

template <typename _Tp>
typename formatter<_Tp>::size_type formatter<_Tp>::m_width = 10;

//======================================================================================//

tim_api class rusage : public formatter<rusage>
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
};

//======================================================================================//

tim_api class timer : public formatter<timer>
{
public:
    enum class field
    {
        wall,
        user,
        system,
        cpu,
        percent,
        laps,
        timing_unit,
    };
};

//======================================================================================//
//
//                          CORE_FORMATTER
//
//======================================================================================//

inline core_formatter::core_formatter(size_type _prec, size_type _width, unit_type _unit,
                                      string_t _fmt, bool _fixed)
: m_data(_prec, _width, _unit, _fmt, _fixed)
{
}

//======================================================================================//
//
//                          BASE_FORMATTER
//
//======================================================================================//

template <typename _Tp>
inline formatter<_Tp>::formatter(string_t _prefix, string_t _suffix, string_t _format,
                                 unit_type _unit, bool _align_width, size_type _prec,
                                 size_type _width, bool _fixed)
: core_type(_prec, _width, _unit, _format, _fixed)
, m_align_width(_align_width)
, m_fixed(_fixed)
, m_prefix(_prefix)
, m_suffix(_suffix)
{
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline tim::string
formatter<_Tp>::compose() const
{
    std::stringstream _ss;
    if(m_align_width)
    {
        //_ss << std::setw(f_current().width() + 1) << std::left << m_prefix << " "
        //    << std::right << this->format() << std::left << m_suffix;
    }
    else
    {
        _ss << std::setw(width() + 1) << std::left << m_prefix << " " << std::right
            << this->format() << std::left << m_suffix;
    }
    return _ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline tim::string
formatter<_Tp>::operator()(const string_t& _base) const
{
    string_t _str = (_base.length() == 0) ? this->compose() : _base;

    for(auto itr : get_field_list())
    {
        auto _replace = [&](const string_t& _itr, const string_t& _rep) {
            auto _npos = tim::string::npos;
            while((_npos = _str.find(_itr)) != tim::string::npos)
                _str.replace(_npos, _itr.length(), _rep.c_str());
        };

        // if(itr.second == rss::field::memory_unit)
        {
            std::stringstream _ss;
            _ss.precision(this->precision());
            _ss << tim::units::mem_repr(this->unit());
            _replace(itr.first, _ss.str());
        }
        // else
        {
            // replace all instances
            _replace(", " + itr.first, "");        // CSV
            _replace("," + itr.first, "");         // CSV
            _replace(" " + itr.first + " ", " ");  // surrounding space
            _replace(" " + itr.first, "");         // leading space
            _replace(itr.first + " ", "");         // trailing space
            _replace(itr.first, "");               // every remaining instance
        }
    }

    string_t _pR   = "%R";
    auto     _npos = tim::string::npos;
    while((_npos = _str.find(_pR)) != tim::string::npos)
        _str = _str.replace(_npos, _pR.length(), "");

    return _str;
}

//======================================================================================//

}  // namespace format

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#undef BACKWARD_COMPAT_SET
#undef BACKWARD_COMPAT_GET
