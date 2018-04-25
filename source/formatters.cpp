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

//============================================================================//
//
//                          Default values
//
//============================================================================//

//============================================================================//
//      numerics
//============================================================================//

format::rss::size_type          format::rss::f_default_width        = 3;
format::rss::size_type          format::rss::f_default_precision    = 1;
format::rss::unit_type          format::rss::f_default_unit         = tim::units::megabyte;

format::timer::size_type        format::timer::f_default_width      = 8;
format::timer::size_type        format::timer::f_default_precision  = 3;
format::timer::unit_type        format::timer::f_default_unit       = units::sec;

//============================================================================//
//      format strings
//============================================================================//

format::rss::string_t
format::rss::f_default_format       = std::string(": RSS {curr,peak} : (%C|%M) [%A]");

format::timer::string_t
format::timer::f_default_format     = std::string(": %w wall, %u user + %s system = %t CPU [%T] (%p%)") +
                                      std::string("%R (x%l laps)");
format::timer::rss_format_t
format::timer::f_default_rss_format = format::rss("",
                                                  ": RSS {tot,self}_{curr,peak} : (%C|%M) | (%c|%m) [%A]",
                                                  format::rss::get_default_unit(),
                                                  false);

//============================================================================//
//      field lists
//============================================================================//

format::rss::field_list_t   format::rss::f_field_list =
{
    format::rss::field_pair_t("%C", format::rss::field::current     ),
    format::rss::field_pair_t("%M", format::rss::field::peak        ),
    format::rss::field_pair_t("%c", format::rss::field::self_curr   ),
    format::rss::field_pair_t("%m", format::rss::field::self_peak   ),
    format::rss::field_pair_t("%C", format::rss::field::total_curr  ),
    format::rss::field_pair_t("%M", format::rss::field::total_peak  ),
    format::rss::field_pair_t("%A", format::rss::field::memory_unit )
};


format::timer::field_list_t   format::timer::f_field_list =
{
    format::timer::field_pair_t("%w", format::timer::field::wall        ),
    format::timer::field_pair_t("%u", format::timer::field::user        ),
    format::timer::field_pair_t("%s", format::timer::field::system      ),
    format::timer::field_pair_t("%t", format::timer::field::cpu         ),
    format::timer::field_pair_t("%p", format::timer::field::percent     ),
    format::timer::field_pair_t("%R", format::timer::field::rss         ),
    format::timer::field_pair_t("%l", format::timer::field::laps        ),
    format::timer::field_pair_t("%T", format::timer::field::timing_unit ),
};

//============================================================================//
//
//                          BASE_FORMATTER
//
//============================================================================//



//============================================================================//
//
//                          TIMING
//
//============================================================================//

void format::timer::propose_default_width(size_type _w)
{
    f_default_width = std::max(f_default_width, _w);
}

//============================================================================//

format::timer::string_t
format::timer::compose() const
{
    std::stringstream _ss;
    if(m_align_width)
    {
        _ss << std::setw(f_default_width + 1)
            << std::left << m_prefix << " "
            << std::right << m_format
            << std::left << m_suffix;
    }
    else
    {
        _ss << std::left << m_prefix << " "
            << std::right << m_format
            << std::left << m_suffix;
    }
    return _ss.str();
}

//============================================================================//

format::timer::string_t
format::timer::operator()(const internal::base_timer* t) const
{
    string_t _str = this->compose();

    double _real = t->real_elapsed() * m_unit;
    double _user = t->user_elapsed() * m_unit;
    double _system = t->system_elapsed() * m_unit;
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

    for(auto itr : f_field_list)
    {
        std::stringstream _ss;
        _ss.precision(m_precision);
        _ss << std::fixed;
        switch (itr.second)
        {
            case format::timer::field::wall:
                // the real elapsed time
                _ss << std::setw(noff+m_precision)
                   << (_real);
                break;
            case format::timer::field::user:
                // CPU time of non-system calls
                _ss << std::setw(noff+m_precision)
                   << (_user);
                break;
            case format::timer::field::system:
                // thread specific CPU time, e.g. thread creation overhead
                _ss << std::setw(noff+m_precision)
                   << (_system);
                break;
            case format::timer::field::cpu:
                // total CPU time
                _ss << std::setw(noff+m_precision)
                   << (_cpu);
                break;
            case format::timer::field::percent:
                // percent CPU utilization
                _ss.precision(1);
                _ss << std::setw(5) << (_perc);
                break;
            case format::timer::field::rss:
                _ss << m_rss_format.format();
                break;
            case format::timer::field::laps:
                _ss.precision(1);
                _ss << _laps;
                break;
            case format::timer::field::timing_unit:
                _ss << units::time_repr(m_unit);
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
        {
            _str = _str.replace(_npos, itr.first.length(), _ss.str().c_str());
        }

    }

    // add RSS
    _str = m_rss_format(&t->accum().rss(), _str);

    return _str;
}

//============================================================================//

void format::timer::set_default(const timer& rhs)
{
    format::timer::set_default_format(rhs.format());
    format::timer::set_default_precision(rhs.precision());
    format::timer::set_default_rss_format(rhs.rss_format());
    format::timer::set_default_unit(rhs.unit());
    format::timer::set_default_width(rhs.width());
}

//============================================================================//

format::timer format::timer::get_default()
{
    format::timer obj;
    obj.set_format(format::timer::get_default_format());
    obj.set_precision(format::timer::get_default_precision());
    obj.set_rss_format(format::timer::get_default_rss_format());
    obj.set_unit(format::timer::get_default_unit());
    obj.set_width(format::timer::get_default_width());
    return obj;
}

//============================================================================//

format::timer* format::timer::copy_from(const timer* rhs)
{
    m_precision = rhs->precision();
    m_width = rhs->width();
    m_unit = rhs->unit();
    m_format = rhs->format();
    m_rss_format = rhs->rss_format();
    return this;
}

//============================================================================//
//
//                          RSS
//
//============================================================================//


format::rss::string_t
format::rss::compose() const
{
    std::stringstream _ss;
    if(m_align_width)
    {
        _ss << std::setw(f_default_width + 1)
            << std::left << m_prefix << " "
            << std::right << m_format
            << std::left << m_suffix;
    }
    else
    {
        _ss << std::left << m_prefix << " "
            << std::right << m_format
            << std::left << m_suffix;
    }
    return _ss.str();
}

//============================================================================//

format::rss::string_t
format::rss::operator()(const tim::rss::usage* m) const
{
    string_t _str = this->compose();

    double _peak = m->peak(m_unit);
    double _curr = m->current(m_unit);

    // rss spacing
    static const uint16_t _wmin = 2;
    static uint16_t _wrss = _wmin;
    uint16_t wrss = (m_align_width) ? _wrss : _wmin;
    for( double _mem : { _peak, _curr } )
        if(_mem > 10.0)
            wrss = std::max(wrss, (uint16_t) (log10(_mem) + 2));
    _wrss = std::max(wrss, _wrss);

    for(auto itr : f_field_list)
    {
        std::stringstream _ss;
        _ss.precision(m_precision);
        _ss << std::fixed;
        switch (itr.second)
        {
            case format::rss::field::current:
            case format::rss::field::total_curr:
            case format::rss::field::self_curr:
                // RSS (current)
                _ss.precision(m_precision);
                _ss << std::setw(wrss+1)
                   << _curr;
                break;
            case format::rss::field::peak:
            case format::rss::field::total_peak:
            case format::rss::field::self_peak:
                // RSS (peak)
                _ss.precision(m_precision);
                _ss << std::setw(wrss+1)
                   << _peak;
                break;
            case format::rss::field::memory_unit:
                _ss.precision(m_precision);
                _ss << tim::units::mem_repr(m_unit);
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
        {
            _str = _str.replace(_npos, itr.first.length(), _ss.str().c_str());
        }

    }

    return _str;
}

//============================================================================//

format::rss::string_t
format::rss::operator()(const tim::internal::base_rss_usage* m,
                        const string_t& _base) const
{
    string_t _str = (_base.length() == 0) ? this->compose() : _base;

    double _tot_peak = m->total().peak(m_unit);
    double _tot_curr = m->total().current(m_unit);
    double _self_peak = m->self().peak(m_unit);
    double _self_curr = m->self().current(m_unit);

    // rss spacing
    static const uint16_t _wmin = 2;
    static uint16_t _wrss = _wmin;
    uint16_t wrss = (m_align_width) ? _wrss : _wmin;
    for( double _mem : { _tot_peak, _tot_curr, _self_peak, _self_curr } )
        if(_mem > 10.0)
            wrss = std::max(wrss, (uint16_t) (log10(_mem) + 2));
    _wrss = std::max(wrss, _wrss);

    for(auto itr : f_field_list)
    {
        std::stringstream _ss;
        _ss.precision(m_precision);
        _ss << std::fixed;
        switch (itr.second)
        {
            case format::rss::field::current:
            case format::rss::field::peak:
                continue;
                break;
            case format::rss::field::total_curr:
                // RSS (current)
                _ss << std::setw(wrss+1)
                   << _tot_curr;
                break;
            case format::rss::field::self_curr:
                // RSS (current)
                _ss << std::setw(wrss+1)
                   << _self_curr;
                break;
            case format::rss::field::total_peak:
                // RSS (peak)
                _ss << std::setw(wrss+1)
                   << _tot_peak;
                break;
            case format::rss::field::self_peak:
                // RSS (peak)
                _ss << std::setw(wrss+1)
                   << _self_peak;
                break;
            case format::rss::field::memory_unit:
                _ss << tim::units::mem_repr(m_unit);
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
        {
            _str = _str.replace(_npos, itr.first.length(), _ss.str().c_str());
        }

    }

    return _str;
}

//============================================================================//

void format::rss::set_default(const rss& rhs)
{
    format::rss::set_default_format(rhs.format());
    format::rss::set_default_precision(rhs.precision());
    format::rss::set_default_unit(rhs.unit());
    format::rss::set_default_width(rhs.width());
}

//============================================================================//

format::rss format::rss::get_default()
{
    format::rss obj;
    obj.set_format(format::rss::get_default_format());
    obj.set_precision(format::rss::get_default_precision());
    obj.set_unit(format::rss::get_default_unit());
    obj.set_width(format::rss::get_default_width());
    return obj;
}

//============================================================================//

format::rss* format::rss::copy_from(const rss* rhs)
{
    m_precision = rhs->precision();
    m_width = rhs->width();
    m_unit = rhs->unit();
    m_format = rhs->format();
    return this;
}

//============================================================================//


}
