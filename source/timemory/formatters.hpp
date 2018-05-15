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
#include <tuple>
#include <stack>
#include <initializer_list>

#if defined(_UNIX)
#   include <unistd.h>
#endif

#define BACKWARD_COMPAT_SET(type, func) \
    static void set_##func (const type & _v) { func (_v); }

#define BACKWARD_COMPAT_GET(type, func) \
    static const type & get_##func () { return func (); }

namespace tim
{

//============================================================================//

namespace internal
{
    class base_timer; // declaration for format::timer
}
namespace rss
{
    class usage; // declaration for format::rss
    class usage_delta; // declaration for format::rss_usage
}

//============================================================================//

namespace format
{

//============================================================================//

typedef std::tuple<int16_t, int16_t, int64_t, std::string, bool> core_tuple_t;

//============================================================================//

class tim_api core_formatter
{
public:
    typedef core_tuple_t    base_type;
    typedef std::string     string_t;
    typedef int16_t         size_type;
    typedef int64_t         unit_type;

public:
    core_formatter(size_type, size_type, unit_type, string_t, bool);

public:
    // public member functions
    void precision(const size_type& _val)   { std::get<0>(m_data) = _val;    }
    void width(const size_type& _val)       { std::get<1>(m_data) = _val;    }
    void unit(const unit_type& _val)        { std::get<2>(m_data) = _val;    }
    void format(const string_t& _val)       { std::get<3>(m_data) = _val;    }
    void fixed(const bool& _val)            { std::get<4>(m_data) = _val;    }
    void scientific(const bool& _val)       { std::get<4>(m_data) = !(_val); }

    size_type& precision()                  { return std::get<0>(m_data);    }
    size_type& width()                      { return std::get<1>(m_data);    }
    unit_type& unit()                       { return std::get<2>(m_data);    }
    string_t&  format()                     { return std::get<3>(m_data);    }
    bool&      fixed()                      { return std::get<4>(m_data);    }

    const size_type& precision() const      { return std::get<0>(m_data);    }
    const size_type& width() const          { return std::get<1>(m_data);    }
    const unit_type& unit() const           { return std::get<2>(m_data);    }
    const string_t&  format() const         { return std::get<3>(m_data);    }
    const bool& fixed() const               { return std::get<4>(m_data);    }
    bool scientific() const                 { return !(std::get<4>(m_data)); }

protected:
    core_tuple_t    m_data;
};

//============================================================================//

class tim_api base_formatter : public core_formatter
{
public:
    typedef std::stringstream           stringstream_t;
    typedef core_formatter              base_type;
    typedef core_formatter              core_type;

public:
    // public constructors
    // public constructors
    base_formatter(string_t _prefix, string_t _suffix,
                   string_t _format, unit_type _unit,
                   bool _align_width,
                   size_type _prec, size_type _width,
                   bool _fixed = true);

    virtual ~base_formatter() { }

public:
    // public member functions
    void align_width(const bool& _val)      { m_align_width = _val; }
    void prefix(const string_t& _val)       { m_prefix = _val; }
    void suffix(const string_t& _val)       { m_suffix = _val; }

    bool&           align_width()           { return m_align_width; }
    string_t&       prefix()                { return m_prefix; }
    string_t&       suffix()                { return m_suffix; }

    const bool&     align_width() const         { return m_align_width; }
    const string_t& prefix() const          { return m_prefix; }
    const string_t& suffix() const          { return m_suffix; }

protected:
    // protected member functions
    virtual string_t compose() const = 0;

protected:
    // protected member variables
    bool        m_align_width;
    bool        m_fixed;
    string_t    m_prefix;
    string_t    m_suffix;
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
    typedef std::stack<core_formatter>  storage_type;

public:
    rss(string_t _prefix = "",
        string_t _format = default_format(),
        unit_type _unit = default_unit(),
        bool _align_width = false,
        size_type _prec = default_precision(),
        size_type _width = default_width(),
        bool _fixed = default_fixed())
    : base_formatter(_prefix, "", _format, _unit, _align_width, _prec, _width, _fixed)
    {
        if(_align_width)
            propose_default_width(_prefix.length());
    }

    virtual ~rss() { }

public:
    // public member functions
    string_t operator()(const tim::rss::usage* m) const;
    string_t operator()(const tim::rss::usage_delta* m,
                        const string_t& = "") const;
    string_t operator()(const string_t& = "") const;
    rss* copy_from(const rss* rhs);

public:
    // public static functions
    static void propose_default_width(size_type);

    static void default_precision(const size_type& _v)  { f_current().precision() = _v;   }
    static void default_width(const size_type& _v)      { f_current().width() = _v;       }
    static void default_unit(const unit_type& _v)       { f_current().unit() = _v;        }
    static void default_format(const string_t& _v)      { f_current().format() = _v;      }
    static void default_fixed(const bool& _v)           { f_current().fixed() = _v;       }
    static void default_scientific(const bool& _v)      { f_current().fixed() = !(_v);    }

    // defines set_<function>
    BACKWARD_COMPAT_SET(size_type,  default_precision   )
    BACKWARD_COMPAT_SET(size_type,  default_width       )
    BACKWARD_COMPAT_SET(unit_type,  default_unit        )
    BACKWARD_COMPAT_SET(string_t,   default_format      )
    BACKWARD_COMPAT_SET(bool,       default_fixed       )

    static const size_type& default_precision()     { return f_current().precision(); }
    static const size_type& default_width()         { return f_current().width();     }
    static const unit_type& default_unit()          { return f_current().unit();      }
    static const string_t&  default_format()        { return f_current().format();    }
    static const bool&      default_fixed()         { return f_current().fixed();     }
    static bool             default_scientific()    { return !(f_current().fixed());  }

    // defines get_<function>
    BACKWARD_COMPAT_GET(size_type,  default_precision   )
    BACKWARD_COMPAT_GET(size_type,  default_width       )
    BACKWARD_COMPAT_GET(unit_type,  default_unit        )
    BACKWARD_COMPAT_GET(string_t,   default_format      )
    BACKWARD_COMPAT_GET(bool,       default_fixed       )

    static void set_default(const rss& rhs);
    static rss  get_default();

    static void push();
    static void pop();

protected:
    // protected member functions
    virtual string_t compose() const final;

private:
    // private static members
    static field_list_t     get_field_list();
    static core_formatter&  f_current();
    static storage_type&    f_history();

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

    typedef std::pair<string_t, field>      field_pair_t;
    typedef std::vector<field_pair_t>       field_list_t;
    typedef rss                             rss_format_t;
    typedef std::pair<core_formatter, rss>  format_pair_t;
    typedef std::stack<format_pair_t>       storage_type;

public:
    // public constructors
    timer(string_t _prefix = "",
          string_t _format = default_format(),
          unit_type _unit = default_unit(),
          rss_format_t _rss_format = default_rss_format(),
          bool _align_width = false,
          size_type _prec = default_precision(),
          size_type _width = default_width(),
          bool _fixed = default_fixed())
    : base_formatter(_prefix, "", _format, _unit, _align_width, _prec, _width, _fixed),
      m_rss_format(_rss_format)
    {
        if(_align_width)
            propose_default_width(_prefix.length());
    }

    virtual ~timer() { }

public:
    // public member functions
    string_t operator()(const internal::base_timer* m) const;
    timer*              copy_from(const timer* rhs);

    void                rss_format(const rss_format_t& _val)    { m_rss_format = _val; }
    rss_format_t&       rss_format()                            { return m_rss_format; }
    const rss_format_t& rss_format() const                      { return m_rss_format; }

public:
    // public static functions
    static void propose_default_width(size_type);

    static void default_precision(const size_type& _v)  { f_current().first.precision() = _v; }
    static void default_width(const size_type& _v)      { f_current().first.width() = _v;     }
    static void default_unit(const unit_type& _v)       { f_current().first.unit() = _v;      }
    static void default_format(const string_t& _v)      { f_current().first.format() = _v;    }
    static void default_fixed(const bool& _v)           { f_current().first.fixed() = _v;     }
    static void default_scientific(const bool& _v)      { f_current().first.fixed() = !(_v);  }
    static void default_rss_format(const rss& _v)       { f_current().second = _v;            }

    // defines set_<function>
    BACKWARD_COMPAT_SET(size_type,  default_precision   )
    BACKWARD_COMPAT_SET(size_type,  default_width       )
    BACKWARD_COMPAT_SET(unit_type,  default_unit        )
    BACKWARD_COMPAT_SET(string_t,   default_format      )
    BACKWARD_COMPAT_SET(bool,       default_fixed       )
    BACKWARD_COMPAT_SET(rss,        default_rss_format  )

    static const size_type& default_precision()     { return f_current().first.precision();   }
    static const size_type& default_width()         { return f_current().first.width();       }
    static const unit_type& default_unit()          { return f_current().first.unit();        }
    static const string_t&  default_format()        { return f_current().first.format();      }
    static const bool&      default_fixed()         { return f_current().first.fixed();       }
    static bool             default_scientific()    { return !(f_current().first.fixed());    }
    static const rss&       default_rss_format()    { return f_current().second;              }

    // defines get_<function>
    BACKWARD_COMPAT_GET(size_type,  default_precision   )
    BACKWARD_COMPAT_GET(size_type,  default_width       )
    BACKWARD_COMPAT_GET(unit_type,  default_unit        )
    BACKWARD_COMPAT_GET(string_t,   default_format      )
    BACKWARD_COMPAT_GET(bool,       default_fixed       )
    BACKWARD_COMPAT_GET(rss,        default_rss_format  )

    static void  set_default(const timer& rhs);
    static timer get_default();

    static void push();
    static void pop();

protected:
    // protected member functions
    virtual string_t compose() const final;

protected:
    // protected member variables
    rss_format_t        m_rss_format;

private:
    // private static members
    static field_list_t     get_field_list();
    static format_pair_t&   f_current();
    static storage_type&    f_history();
};

//============================================================================//

} // namespace format

//----------------------------------------------------------------------------//

} // namespace tim

//----------------------------------------------------------------------------//

#undef BACKWARD_COMPAT_SET
#undef BACKWARD_COMPAT_GET

#endif

