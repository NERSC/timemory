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
#include "timemory/namespace.hpp"

namespace NAME_TIM
{

//============================================================================//

uint64_t& auto_timer::nhash() { return timing_manager::instance()->hash(); }

//============================================================================//

uint64_t& auto_timer::ncount() { return timing_manager::instance()->count(); }

//============================================================================//

bool auto_timer::alloc_next()
{
    return timing_manager::is_enabled() &&
            (uint64_t) timing_manager::max_depth() > auto_timer::ncount();
}

//============================================================================//

auto_timer::auto_timer(const std::string& timer_tag,
                       const int32_t& lineno,
                       const std::string& code_tag,
                       bool report_at_exit)
: m_report_at_exit(report_at_exit),
  m_hash(10*lineno),
  m_timer(nullptr)
{
    m_hash += std::hash<std::string>()(timer_tag);
    // for consistency, always increment hash keys
    ++auto_timer::ncount();
    auto_timer::nhash() += m_hash;

    if(timing_manager::is_enabled() &&
       (uint64_t) timing_manager::max_depth() > auto_timer::ncount() - 1)
    {
        m_timer = &timing_manager::instance()->timer(timer_tag, code_tag,
                                                     auto_timer::ncount() - 1,
                                                     auto_timer::nhash());
        if(m_report_at_exit)
        {
            m_temp_timer.grab_metadata(*m_timer);
            m_temp_timer.set_begin("> [" + code_tag + "] " + timer_tag);
            m_temp_timer.set_use_static_width(false);
        }

        m_temp_timer.start();
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
        m_temp_timer.stop();
        *m_timer += m_temp_timer;
        if(m_report_at_exit)
            m_temp_timer.report(std::cout, true, true);
    }
}

//============================================================================//

} // namespace NAME_TIM
