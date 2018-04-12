//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California, 
// through Lawrence Berkeley National Laboratory (subject to receipt of any 
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "timemory/auto_timer.hpp"
#include "timemory/utility.hpp"
#include "timemory/macros.hpp"

namespace tim
{

//============================================================================//

auto_timer::counter_t& auto_timer::nhash()
{
    return manager::instance()->hash();
}

//============================================================================//

auto_timer::counter_t& auto_timer::ncount()
{
    return manager::instance()->count();
}

//============================================================================//

auto_timer::counter_t& auto_timer::phash()
{
    return manager::instance()->parent_hash();
}

//============================================================================//

auto_timer::counter_t& auto_timer::pcount()
{
    return manager::instance()->parent_count();
}

//============================================================================//

bool auto_timer::alloc_next()
{
    return manager::is_enabled() &&
            (uint64_t) manager::max_depth() > auto_timer::ncount();
}

//============================================================================//

auto_timer::auto_timer(const string_t& timer_tag,
                       const int32_t& lineno,
                       const string_t& code_tag,
                       bool report_at_exit)
: m_report_at_exit(report_at_exit),
  m_hash(10*lineno),
  m_timer(nullptr),
  m_temp_timer(nullptr)
{
    m_hash += std::hash<string_t>()(timer_tag);
    // for consistency, always increment hash keys
    ++auto_timer::ncount();
    auto_timer::nhash() += m_hash;

    if(manager::is_enabled() &&
       (uint64_t) manager::max_depth() > auto_timer::ncount() - 1)
    {
        m_timer = &manager::instance()->timer(timer_tag, code_tag,
                                              auto_timer::pcount() +
                                              auto_timer::ncount() - 1,
                                              auto_timer::phash() +
                                              auto_timer::nhash());

        m_temp_timer = new tim_timer_t();
        if(m_report_at_exit)
        {
            m_temp_timer->grab_metadata(*m_timer);
            m_temp_timer->set_begin("> [" + code_tag + "] " + timer_tag);
            m_temp_timer->set_use_static_width(false);
        }
        m_temp_timer->start();
    }
}

//============================================================================//

auto_timer::auto_timer(tim_timer_t& _atimer,
                       const int32_t& lineno,
                       const string_t& code_tag,
                       bool report_at_exit)
: m_report_at_exit(report_at_exit),
  m_hash(10*lineno),
  m_timer(nullptr),
  m_temp_timer(nullptr)
{
    string_t timer_tag = _atimer.begin();
    m_hash += std::hash<string_t>()(timer_tag);
    // for consistency, always increment hash keys
    ++auto_timer::ncount();
    auto_timer::nhash() += m_hash;

    if(manager::is_enabled() &&
       (uint64_t) manager::max_depth() > auto_timer::ncount() - 1)
    {
        m_timer = &manager::instance()->timer(timer_tag, code_tag,
                                              auto_timer::pcount() +
                                              auto_timer::ncount() - 1,
                                              auto_timer::phash() +
                                              auto_timer::nhash());
        m_temp_timer = m_timer;
        m_timer->sync(_atimer);
        if(m_report_at_exit)
        {
            m_temp_timer->grab_metadata(*m_timer);
            m_temp_timer->set_begin("> [" + code_tag + "] " + timer_tag);
            m_temp_timer->set_use_static_width(false);
        }
    }
    else
    {
        m_temp_timer = new tim_timer_t();
        m_temp_timer->sync(_atimer);
        if(m_report_at_exit)
        {
            m_temp_timer->set_begin("> [" + code_tag + "] " + timer_tag);
            m_temp_timer->set_use_static_width(false);
        }
    }
}

//============================================================================//

auto_timer::~auto_timer()
{
    // for consistency, always decrement hash keys
    --auto_timer::ncount();
    auto_timer::nhash() -= m_hash;

    if(m_timer)
    {
        m_temp_timer->stop();
        if(m_timer != m_temp_timer)
            *m_timer += *m_temp_timer;
        if(m_report_at_exit)
            m_temp_timer->report(std::cout, true, true);
    }
    else if(m_temp_timer)
    {
        m_temp_timer->stop();
        if(m_report_at_exit)
            m_temp_timer->report(std::cout, true, true);
    }

    if(m_temp_timer && m_temp_timer != m_timer)
        delete m_temp_timer;
}

//============================================================================//

} // namespace tim
