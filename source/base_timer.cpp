// MIT License
//
// Copyright (c) 2018 Jonathan R. Madsen
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

#include "timemory/base_timer.hpp"
#include <cassert>
#include <algorithm>

//============================================================================//

CEREAL_CLASS_VERSION(tim::util::details::base_timer_data, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(tim::util::details::base_timer, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_clock_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_clock_data_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_duration_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_time_point_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_time_pair_t, TIMEMORY_TIMER_VERSION)

//============================================================================//

namespace tim
{
namespace util
{
namespace details
{

//============================================================================//

thread_local uint64_t base_timer::f_instance_count = 0;

//============================================================================//

thread_local uint64_t base_timer::f_instance_hash = 0;

//============================================================================//

thread_local base_timer::data_map_t* base_timer::f_data_map = nullptr;

//============================================================================//

base_timer::mutex_map_t base_timer::w_mutex_map;

//============================================================================//

base_timer::base_timer(uint16_t prec, const string_t& fmt, std::ostream* os)
: m_running(false),
  m_precision(prec),
  m_os(os),
  m_format_positions(poslist_t()),
  m_format_string(fmt),
  m_output_format("")
{ }

//============================================================================//

base_timer::~base_timer()
{
    if(m_running)
    {
        this->stop();
        if(m_os != &std::cout && *m_os)
            this->report();
    }
}

//============================================================================//

void base_timer::parse_format()
{
    m_format_positions.clear();

    this->compose();

    size_type npos = std::string::npos;

    strlist_t fmts;
    fmts.push_back(fieldstr_t("%w", timer_field::wall          ));
    fmts.push_back(fieldstr_t("%u", timer_field::user          ));
    fmts.push_back(fieldstr_t("%s", timer_field::system        ));
    fmts.push_back(fieldstr_t("%t", timer_field::cpu           ));
    fmts.push_back(fieldstr_t("%p", timer_field::percent       ));
    fmts.push_back(fieldstr_t("%c", timer_field::self_curr     ));
    fmts.push_back(fieldstr_t("%m", timer_field::self_peak     ));
    fmts.push_back(fieldstr_t("%C", timer_field::total_curr    ));
    fmts.push_back(fieldstr_t("%M", timer_field::total_peak    ));


    for(strlist_t::iterator itr = fmts.begin(); itr != fmts.end(); ++itr)
    {
        size_type pos = 0;
        // start at zero and look for all instances of string
        while((pos = m_format_string.find(itr->first, pos)) != npos)
        {
            // post-increment pos so we don't find same instance next
            // time around
            m_format_positions.push_back(fieldpos_t(pos++, itr->second));
        }
    }
    std::sort(m_format_positions.begin(), m_format_positions.end(),
              [] (const fieldpos_t& lhs, const fieldpos_t& rhs)
              { return lhs.first < rhs.first; });

}

//============================================================================//

void base_timer::report(std::ostream& os, bool endline, bool avg, bool always) const
{
    const_cast<base_timer*>(this)->parse_format();

    // stop, if not already stopped
    if(m_running)
        const_cast<base_timer*>(this)->stop();

    // for average reporting
    double div = 1.0;
    if(avg && this->laps() > 1)
        div = 1.0 / static_cast<double>(this->laps());

    double _real = real_elapsed();
    double _user = user_elapsed();
    double _system = system_elapsed();
    double _cpu = _user + _system;
    double _perc = (_cpu / _real) * 100.0;

    double tmin = 1.0 / (pow( (uint32_t) 10, (uint32_t) m_precision));
    // skip if it will be reported as all zeros
    // e.g. tmin = ( 1. / 10^3 ) = 0.001;
    if(!always && (_real < tmin || _cpu < tmin || _perc < 0.1))
        return;

    // timing spacing
    static uint16_t noff = 3;
    for( double _time : { _real, _user, _system, _cpu } )
        if(_time > 10.0)
            noff = std::max(noff, (uint16_t) (log10(_time) + 2));

    static uint16_t wrss = 3;
    for( double _mem : { m_rss_self.peak(),    m_rss_tot.peak(),
                        m_rss_self.current(), m_rss_tot.current() } )
        if(_mem > 10.0)
            wrss = std::max(wrss, (uint16_t) (log10(_mem) + 2));

    // use stringstream so precision and fixed don't directly affect
    // ostream
    std::stringstream ss;
    // set precision
    ss.precision(m_precision);
    // output fixed
    ss << std::fixed;
    size_type pos = 0;
    for(size_type i = 0; i < m_format_positions.size(); ++i)
    {
        // where to terminate the sub-string
        size_type ter = m_format_positions.at(i).first;
        assert(!(ter < pos));
        // length of substring
        size_type len = ter - pos;
        // create substring
        string_t substr = m_format_string.substr(pos, len);
        // add sub-string
        ss << substr;
        // print the appropriate timing mechanism
        switch (m_format_positions.at(i).second)
        {
            case timer_field::wall:
                // the real elapsed time
                ss << std::setw(noff+m_precision)
                   << (_real * div);
                break;
            case timer_field::user:
                // CPU time of non-system calls
                ss << std::setw(noff+m_precision)
                   << (_user * div);
                break;
            case timer_field::system:
                // thread specific CPU time, e.g. thread creation overhead
                ss << std::setw(noff+m_precision)
                   << (_system * div);
                break;
            case timer_field::cpu:
                // total CPU time
                ss << std::setw(noff+m_precision)
                   << (_cpu * div);
                break;
            case timer_field::percent:
                // percent CPU utilization
                ss.precision(1);
                ss << std::setw(5) << (_perc);
                break;
            case timer_field::total_curr:
                // total RSS (current)
                ss.precision(1);
                ss << std::setw(wrss)
                   << (m_rss_tot.current());
                break;
            case timer_field::total_peak:
                // total RSS (peak)
                ss.precision(1);
                ss << std::setw(wrss)
                   << (m_rss_tot.peak());
                break;
            case timer_field::self_curr:
                // self RSS (current)
                ss.precision(1);
                ss << std::setw(wrss)
                   << (m_rss_self.current());
                break;
            case timer_field::self_peak:
                // self RSS (peak)
                ss.precision(1);
                ss << std::setw(wrss)
                   << (m_rss_self.peak());
                break;

        }
        // skip over %{w,u,s,t,p} field
        pos = m_format_positions.at(i).first+2;
    }
    // write the end of the string
    size_type ter = m_format_string.length();
    size_type len = ter - pos;
    string_t substr = m_format_string.substr(pos, len);
    ss << substr;

    //std::cout << " [ total " << m_rss_tot << " ] [ self " << m_rss_self << " ]";
    if(avg && this->laps() > 1)
        ss << " (average of " << this->laps() << " laps)";
    else if(this->laps() > 1)
        ss << " (total # of laps: " << this->laps() << ")";

    if(endline)
        ss << std::endl;

    // ensure thread-safety
    recursive_lock_t lock(w_mutex_map[&os]);
    // output to ostream
    os << ss.str();
}

} // namespace details

} // namespace util

} // namespace tim

//============================================================================//



