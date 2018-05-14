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

#include "timemory/macros.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/utility.hpp"
#include "timemory/timer.hpp"
#include "timemory/manager.hpp"

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
                       const string_t& lang_tag,
                       bool report_at_exit)
: m_enabled(auto_timer::alloc_next()),
  m_report_at_exit(report_at_exit),
  m_hash((m_enabled) ? (lineno + std::hash<string_t>()(timer_tag)) : 0),
  m_temp_timer(
      tim_timer_t(
          m_enabled,
          (m_enabled) ? &manager::instance()->timer(
                            timer_tag, lang_tag,
                            (m_enabled) ? (auto_timer::pcount() +
                                           (uint64_t) (auto_timer::ncount()++))
                                        : ((uint64_t) (0)),
                            (m_enabled) ? (auto_timer::phash() +
                                           (uint64_t) (auto_timer::nhash() += m_hash))
                                        : ((uint64_t) (0)))
                      : nullptr))
{ }

//============================================================================//

auto_timer::~auto_timer()
{
    if(m_enabled)
    {
        // will add itself to global when destroying m_temp_timer
        m_temp_timer.stop();

        assert(m_temp_timer.summation_timer() != nullptr);
        *m_temp_timer.summation_timer() += m_temp_timer;

        // report timer at exit
        if(m_report_at_exit)
        {
            m_temp_timer.grab_metadata(*(m_temp_timer.summation_timer()));

            // show number of laps in temporary timer
            auto _laps = m_temp_timer.summation_timer()->accum().size();
            m_temp_timer.accum().size() += _laps;

            // threadsafe output w.r.t. other timers
            m_temp_timer.report(std::cout, true, true);
        }

        // decrement hash keys
        --auto_timer::ncount();
        auto_timer::nhash() -= m_hash;
    }
}

//============================================================================//

} // namespace tim
