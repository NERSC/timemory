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

#include "timemory/formatters.hpp"
#include "timemory/rss.hpp"
#include "timemory/timer.hpp"
#include "timemory/base_timer.hpp"

namespace tim
{

namespace format
{

//============================================================================//
//
//                          CORE_FORMATTER
//
//============================================================================//

core_formatter::core_formatter(size_type _prec,
                               size_type _width,
                               unit_type _unit,
                               string_t _fmt,
                               bool _fixed)
: m_data(_prec, _width, _unit, _fmt, _fixed)
{ }

//============================================================================//
//
//                          BASE_FORMATTER
//
//============================================================================//

base_formatter::base_formatter(string_t _prefix, string_t _suffix,
                               string_t _format, unit_type _unit,
                               bool _align_width,
                               size_type _prec, size_type _width,
                               bool _fixed)
: core_type(_prec, _width, _unit, _format, _fixed),
  m_align_width(_align_width),
  m_prefix(_prefix),
  m_suffix(_suffix)
{ }

//============================================================================//
//      format strings
//============================================================================//

core_formatter&
rss::f_current()
{
    static core_formatter _instance =
            core_formatter(1,                       // precision
                           3,                       // min prefix field width
                           tim::units::megabyte,    // memory display units
                                                    // format string
                           ": RSS {curr,peak} : (%C|%M) [%A]",
                           true                     // std::fixed
                           );
    return _instance;
}

//----------------------------------------------------------------------------//

timer::format_pair_t&
timer::f_current()
{
    static timer::format_pair_t _instance =
            timer::format_pair_t(
                core_formatter(3,                   // precision
                               5,                   // min prefix field width
                               units::sec,          // timing display units
                                                    // format string
                               ": %w wall, %u user + %s system = %t CPU [%T] (%p%) %R (x%l laps)",
                               true                 // std::fixed
                               ),
                rss("",                             // prefix (ignored with timer)
                                                    // format string
                    ": RSS {tot,self}_{curr,peak} : (%C|%M) | (%c|%m) [%A]",
                    rss::default_unit(),            // memory display units
                    true,                           // align width
                    1,                              // precision
                    3,                              // min field width
                    true                            // std::fixed
                    )
                );

    return _instance;
}

//============================================================================//
//      field lists
//============================================================================//

rss::field_list_t
rss::get_field_list()
{
    return
    {
        rss::field_pair_t("%C", rss::field::current     ),
        rss::field_pair_t("%M", rss::field::peak        ),
        rss::field_pair_t("%c", rss::field::self_curr   ),
        rss::field_pair_t("%m", rss::field::self_peak   ),
        rss::field_pair_t("%C", rss::field::total_curr  ),
        rss::field_pair_t("%M", rss::field::total_peak  ),
        rss::field_pair_t("%A", rss::field::memory_unit )
    };
}

//----------------------------------------------------------------------------//

timer::field_list_t
timer::get_field_list()
{
    return
    {
        timer::field_pair_t("%w", timer::field::wall        ),
        timer::field_pair_t("%u", timer::field::user        ),
        timer::field_pair_t("%s", timer::field::system      ),
        timer::field_pair_t("%t", timer::field::cpu         ),
        timer::field_pair_t("%p", timer::field::percent     ),
        timer::field_pair_t("%R", timer::field::rss         ),
        timer::field_pair_t("%l", timer::field::laps        ),
        timer::field_pair_t("%T", timer::field::timing_unit ),
    };
}

//----------------------------------------------------------------------------//

rss::storage_type&
rss::f_history()
{
    static rss::storage_type _instance =
            rss::storage_type({ rss::f_current() });
    return _instance;
}

//----------------------------------------------------------------------------//

timer::storage_type&
timer::f_history()
{
    static timer::storage_type _instance =
            timer::storage_type({ timer::f_current() });
    return _instance;
}

//============================================================================//
//
//                          TIMING
//
//============================================================================//

void timer::push()
{
    // add current to stack
    f_history().push(f_current());
}

//----------------------------------------------------------------------------//

void timer::pop()
{
    // assign only if not empty
    if(f_history().size() > 0)
        f_current() = f_history().top();

    // don't completely empty
    if(f_history().size() > 1)
        f_history().pop();
}

//----------------------------------------------------------------------------//

void timer::propose_default_width(size_type _w)
{
    //_w += 2;    // a little padding
    f_current().first.width() = std::max(f_current().first.width(), _w);
}

//----------------------------------------------------------------------------//

timer::string_t
timer::compose() const
{
    std::stringstream _ss;
    if(m_align_width)
    {
        _ss << std::setw(f_current().first.width() + 1)
            << std::left << m_prefix << " "
            << std::right << this->format()
            << std::left << m_suffix;
    }
    else
    {
        _ss << std::setw(width() + 1)
            << std::left << m_prefix << " "
            << std::right << this->format()
            << std::left << m_suffix;
    }
    return _ss.str();
}

//----------------------------------------------------------------------------//

timer::string_t
timer::operator()(const internal::base_timer* t) const
{
    string_t _str = this->compose();

    double _real = t->real_elapsed() * this->unit();
    double _user = t->user_elapsed() * this->unit();
    double _system = t->system_elapsed() * this->unit();
    double _cpu = _user + _system;
    double _perc = (_cpu / _real) * 100.0;
    auto _laps = t->laps();
    if(!tim::isfinite(_perc))
        _perc = 0.0;

    // timing spacing
    static const uint16_t _wmin = 2;
    static uint16_t _noff = _wmin;
    uint16_t noff = (m_align_width) ? _noff : _wmin;
    for( double _time : { _real, _user, _system, _cpu } )
        if(_time > 10.0)
            noff = std::max(noff, (uint16_t) (log10(_time) + 2));
    _noff = std::max(noff, _noff);

    for(auto itr : get_field_list())
    {
        std::stringstream _ss;
        _ss.precision(this->precision());
        if(this->fixed())
            _ss << std::fixed;
        else
            _ss << std::scientific;
        switch (itr.second)
        {
            case timer::field::wall:
                // the real elapsed time
                _ss << std::setw(noff + this->precision())
                   << (_real);
                break;
            case timer::field::user:
                // CPU time of non-system calls
                _ss << std::setw(noff + this->precision())
                   << (_user);
                break;
            case timer::field::system:
                // thread specific CPU time, e.g. thread creation overhead
                _ss << std::setw(noff + this->precision())
                   << (_system);
                break;
            case timer::field::cpu:
                // total CPU time
                _ss << std::setw(noff + this->precision())
                   << (_cpu);
                break;
            case timer::field::percent:
                // percent CPU utilization
                _ss.precision(1);
                _ss << std::setw(5) << (_perc);
                break;
            case timer::field::rss:
                if(t->record_memory())
                    _ss << m_rss_format.format();
                break;
            case timer::field::laps:
                _ss.precision(1);
                _ss << _laps;
                break;
            case timer::field::timing_unit:
                _ss << units::time_repr(this->unit());
                break;
            default:
                _ss << "unknown field: " << itr.first;
                throw std::runtime_error(_ss.str().c_str());
                break;
        }

        if(_ss.str().length() == 0)
            continue;

        // replace all instances
        auto _npos = std::string::npos;
        while((_npos = _str.find(itr.first)) != std::string::npos)
            _str = _str.replace(_npos, itr.first.length(), _ss.str().c_str());
    }

    // add RSS
    if(t->record_memory())
        _str = m_rss_format(&t->accum().rss(), _str);
    else
        _str = m_rss_format(_str);

    return _str;
}

//----------------------------------------------------------------------------//

void timer::set_default(const timer& rhs)
{
    timer::default_format(rhs.format());
    timer::default_precision(rhs.precision());
    timer::default_rss_format(rhs.rss_format());
    timer::default_unit(rhs.unit());
    timer::default_fixed(rhs.fixed());
    timer::default_width(rhs.width());
}

//----------------------------------------------------------------------------//

timer timer::get_default()
{
    timer obj;
    obj.format(timer::default_format());
    obj.precision(timer::default_precision());
    obj.rss_format(timer::default_rss_format());
    obj.unit(timer::default_unit());
    obj.fixed(timer::default_fixed());
    obj.width(timer::default_width());
    return obj;
}

//----------------------------------------------------------------------------//

timer* timer::copy_from(const timer* rhs)
{
    this->precision() = rhs->precision();
    this->width() = rhs->width();
    this->unit() = rhs->unit();
    this->format() = rhs->format();
    this->fixed() = rhs->fixed();
    this->rss_format() = rhs->rss_format();
    this->align_width() = rhs->align_width();
    return this;
}

//============================================================================//
//
//                          RSS
//
//============================================================================//

void rss::push()
{
    // add current to stack
    f_history().push(f_current());
}

//----------------------------------------------------------------------------//

void rss::pop()
{
    // assign only if not empty
    if(f_history().size() > 0)
        f_current() = f_history().top();

    // don't completely empty
    if(f_history().size() > 1)
        f_history().pop();
}

//----------------------------------------------------------------------------//

void rss::propose_default_width(size_type _w)
{
    //_w += 2;    // a little padding
    f_current().width() = std::max(f_current().width(), _w);
}

//----------------------------------------------------------------------------//

rss::string_t
rss::compose() const
{
    std::stringstream _ss;
    if(m_align_width)
    {
        _ss << std::setw(f_current().width() + 1)
            << std::left << m_prefix << " "
            << std::right << this->format()
            << std::left << m_suffix;
    }
    else
    {
        _ss << std::setw(width() + 1)
            << std::left << m_prefix << " "
            << std::right << this->format()
            << std::left << m_suffix;
    }
    return _ss.str();
}

//----------------------------------------------------------------------------//

rss::string_t
rss::operator()(const string_t& _base) const
{
    string_t _str = (_base.length() == 0) ? this->compose() : _base;

    for(auto itr : get_field_list())
    {
        auto _replace = [&] (const string_t& _itr, const string_t& _rep)
        {
            auto _npos = std::string::npos;
            while((_npos = _str.find(_itr)) != std::string::npos)
                _str.replace(_npos, _itr.length(), _rep.c_str());
        };

        if(itr.second == rss::field::memory_unit)
        {
            std::stringstream _ss;
            _ss.precision(this->precision());
            _ss << tim::units::mem_repr(this->unit());
            _replace(itr.first, _ss.str());
        }
        else
        {
            // replace all instances
            _replace(", " + itr.first,      ""  );  // CSV
            _replace("," + itr.first,       ""  );  // CSV
            _replace(" " + itr.first + " ", " " );  // surrounding space
            _replace(" " + itr.first,       ""  );  // leading space
            _replace(itr.first + " ",       ""  );  // trailing space
            _replace(itr.first,             ""  );  // every remaining instance
        }
    }

    string_t _pR = "%R";
    auto _npos = std::string::npos;
    while((_npos = _str.find(_pR)) != std::string::npos)
        _str = _str.replace(_npos, _pR.length(), "");

    return _str;

}

//----------------------------------------------------------------------------//

rss::string_t
rss::operator()(const tim::rss::usage* m) const
{
    string_t _str = this->compose();

    double _peak = m->peak(this->unit());
    double _curr = m->current(this->unit());

    // rss spacing
    static const uint16_t _wmin = 2;
    static uint16_t _wrss = _wmin;
    uint16_t wrss = (m_align_width) ? _wrss : _wmin;
    for( double _mem : { _peak, _curr } )
        if(_mem > 10.0)
            wrss = std::max(wrss, (uint16_t) (log10(_mem) + 2));
    _wrss = std::max(wrss, _wrss);

    for(auto itr : get_field_list())
    {
        std::stringstream _ss;
        _ss.precision(this->precision());
        if(this->fixed())
            _ss << std::fixed;
        else
            _ss << std::scientific;
        switch (itr.second)
        {
            case rss::field::current:
            case rss::field::total_curr:
            case rss::field::self_curr:
                // RSS (current)
                _ss.precision(this->precision());
                _ss << std::setw(wrss+1)
                   << _curr;
                break;
            case rss::field::peak:
            case rss::field::total_peak:
            case rss::field::self_peak:
                // RSS (peak)
                _ss.precision(this->precision());
                _ss << std::setw(wrss+1)
                   << _peak;
                break;
            case rss::field::memory_unit:
                _ss.precision(this->precision());
                _ss << tim::units::mem_repr(this->unit());
                break;
            default:
                _ss << "unknown field: " << itr.first;
                throw std::runtime_error(_ss.str().c_str());
                break;
        }

        if(_ss.str().length() == 0)
            continue;

        // replace all instances
        auto _npos = std::string::npos;
        while((_npos = _str.find(itr.first)) != std::string::npos)
            _str = _str.replace(_npos, itr.first.length(), _ss.str().c_str());
    }

    _str = (*this)(_str);

    return _str;
}

//----------------------------------------------------------------------------//

rss::string_t
rss::operator()(const tim::rss::usage_delta* m,
                const string_t& _base) const
{
    string_t _str = (_base.length() == 0) ? this->compose() : _base;

    double _tot_peak = m->total().peak(this->unit());
    double _tot_curr = m->total().current(this->unit());
    double _self_peak = m->self().peak(this->unit());
    double _self_curr = m->self().current(this->unit());

    // rss spacing
    static const uint16_t _wmin = 2;
    static uint16_t _wrss = _wmin;
    uint16_t wrss = (m_align_width) ? _wrss : _wmin;
    for( double _mem : { _tot_peak, _tot_curr, _self_peak, _self_curr } )
        if(_mem > 10.0)
            wrss = std::max(wrss, (uint16_t) (log10(_mem) + 2));
    _wrss = std::max(wrss, _wrss);

    for(auto itr : get_field_list())
    {
        std::stringstream _ss;
        _ss.precision(this->precision());
        if(this->fixed())
            _ss << std::fixed;
        else
            _ss << std::scientific;
        switch (itr.second)
        {
            case rss::field::current:
            case rss::field::peak:
                continue;
                break;
            case rss::field::total_curr:
                // RSS (current)
                _ss << std::setw(wrss+1)
                   << _tot_curr;
                break;
            case rss::field::self_curr:
                // RSS (current)
                _ss << std::setw(wrss+1)
                   << _self_curr;
                break;
            case rss::field::total_peak:
                // RSS (peak)
                _ss << std::setw(wrss+1)
                   << _tot_peak;
                break;
            case rss::field::self_peak:
                // RSS (peak)
                _ss << std::setw(wrss+1)
                   << _self_peak;
                break;
            case rss::field::memory_unit:
                _ss << tim::units::mem_repr(this->unit());
                break;
            default:
                _ss << "unknown field: " << itr.first;
                throw std::runtime_error(_ss.str().c_str());
                break;
        }

        if(_ss.str().length() == 0)
            continue;

        // replace all instances
        auto _npos = std::string::npos;
        while((_npos = _str.find(itr.first)) != std::string::npos)
            _str = _str.replace(_npos, itr.first.length(), _ss.str().c_str());

    }

    _str = (*this)(_str);

    return _str;
}

//----------------------------------------------------------------------------//

void rss::set_default(const rss& rhs)
{
    rss::default_format(rhs.format());
    rss::default_precision(rhs.precision());
    rss::default_unit(rhs.unit());
    rss::default_width(rhs.width());
    rss::default_fixed(rhs.fixed());
}

//----------------------------------------------------------------------------//

rss rss::get_default()
{
    rss obj;
    obj.format(rss::default_format());
    obj.precision(rss::default_precision());
    obj.unit(rss::default_unit());
    obj.width(rss::default_width());
    obj.fixed(rss::default_fixed());
    return obj;
}

//----------------------------------------------------------------------------//

rss* rss::copy_from(const rss* rhs)
{
    this->precision() = rhs->precision();
    this->width() = rhs->width();
    this->unit() = rhs->unit();
    this->format() = rhs->format();
    this->fixed() = rhs->fixed();
    this->align_width() = rhs->align_width();
    return this;
}

//----------------------------------------------------------------------------//

} // namespace format

} // namespace tim
