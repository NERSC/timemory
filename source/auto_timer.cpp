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
            m_temp_timer->format()->set_prefix("> [" + code_tag + "] " + timer_tag);
            m_temp_timer->format()->set_use_align_width(false);
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
    string_t timer_tag = _atimer.format()->prefix();
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
        m_temp_timer = (m_report_at_exit)
                       ? new tim_timer_t()
                       : m_timer;
        m_timer->sync(_atimer);
        if(m_report_at_exit)
        {
            m_temp_timer->grab_metadata(*m_timer);
            m_temp_timer->format()->set_prefix("> [" + code_tag + "] " + timer_tag);
            m_temp_timer->format()->set_use_align_width(false);
        }
    }
    else
    {
        m_temp_timer = new tim_timer_t();
        m_temp_timer->sync(_atimer);
        if(m_report_at_exit)
        {
            m_temp_timer->format()->set_prefix("> [" + code_tag + "] " + timer_tag);
            m_temp_timer->format()->set_use_align_width(false);
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
        // report timer at exit
        if(m_report_at_exit)
            m_temp_timer->report(std::cout, true, true);

        // if same timer, don't add to itself
        if(m_timer != m_temp_timer)
            *m_timer += *m_temp_timer;
        else // ensure manager reporting uses align width
            m_temp_timer->format()->set_use_align_width(true);
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

//============================================================================//
//
//                      C interface
//
//============================================================================//

extern "C"
int cxx_timemory_enabled(void)
{
    return (int) tim::manager::instance()->is_enabled();
}

//============================================================================//

extern "C"
void* cxx_timemory_create_auto_timer(const char* timer_tag,
                                     int lineno,
                                     const char* code_tag,
                                     int report)
{
    return (void*) new auto_timer_t(timer_tag, lineno, code_tag, (bool) report);
}

//============================================================================//

extern "C"
void cxx_timemory_report(const char* fname)
{
    std::string _fname(fname);
    for(auto itr : {".txt", ".out", ".json"})
    {
        if(_fname.find(itr) != std::string::npos)
            _fname = _fname.substr(0, _fname.find(itr));
    }
    _fname = _fname.substr(0, _fname.find_last_of("."));

    tim::path_t _fpath_report = _fname + std::string(".out");
    tim::path_t _fpath_serial = _fname + std::string(".json");
    tim::manager::master_instance()->set_output_stream(_fpath_report);
    tim::makedir(tim::dirname(_fpath_report));
    std::ofstream ofs_report(_fpath_report);
    std::ofstream ofs_serial(_fpath_serial);
    if(ofs_report)
        tim::manager::master_instance()->report(ofs_report);
    if(ofs_serial)
        tim::manager::master_instance()->write_json(ofs_serial);
}

//============================================================================//

extern "C"
void* cxx_timemory_delete_auto_timer(void* ctimer)
{
    auto_timer_t* cxxtimer = static_cast<auto_timer_t*>(ctimer);
    delete cxxtimer;
    ctimer = NULL;
    return ctimer;
}

//============================================================================//

extern "C"
const char* cxx_timemory_string_combine(const char* _a, const char* _b)
{
    std::stringstream _ss;
    _ss << _a << _b;
    return _ss.str().c_str();
}

//============================================================================//

extern "C"
const char* cxx_timemory_auto_timer_str(const char* _a, const char* _b,
                                        const char* _c, const char* _d)
{
    std::stringstream _ss;
    _ss << std::string(_a)
        << std::string(_b)
        << "@'"
        << std::string(_c).substr(std::string(_c).find_last_of("/")+1)
        << "':" << std::string(_d);
    return _ss.str().c_str();
}

//============================================================================//
