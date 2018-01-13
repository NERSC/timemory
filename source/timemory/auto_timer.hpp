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

#ifndef auto_timer_hpp_
#define auto_timer_hpp_

#include "timemory/namespace.hpp"
#include "timemory/utility.hpp"
#include "timemory/timing_manager.hpp"

#include <string>
#include <cstdint>

namespace NAME_TIM
{
namespace util
{

class auto_timer
{
public:
    typedef NAME_TIM::util::timing_manager::tim_timer_t  tim_timer_t;

public:
    // Constructor and Destructors
    auto_timer(const std::string&, const int32_t& lineno,
               const std::string& = "cxx", bool temp_disable = false);
    virtual ~auto_timer();

private:
    static uint64_t& ncount()
    { return details::base_timer::get_instance_count(); }

    static uint64_t& nhash()
    { return details::base_timer::get_instance_hash(); }

private:
    bool m_temp_disable;
    uint64_t m_hash;
    tim_timer_t* m_timer;
};

//----------------------------------------------------------------------------//
inline auto_timer::auto_timer(const std::string& timer_tag,
                              const int32_t& lineno,
                              const std::string& code_tag,
                              bool temp_disable)
: m_temp_disable(temp_disable),
  m_hash(lineno),
  m_timer(nullptr)
{
    // for consistency, always increment hash keys
    ++auto_timer::ncount();
    auto_timer::nhash() += m_hash;

    if(timing_manager::is_enabled() &&
       (uint64_t) timing_manager::max_depth() > auto_timer::ncount() + 1)
        m_timer = &timing_manager::instance()->timer(timer_tag, code_tag,
                                                     auto_timer::ncount(),
                                                     auto_timer::nhash());
    if(m_timer)
        m_timer->start();

    if(m_temp_disable && timing_manager::instance()->is_enabled())
        timing_manager::instance()->enable(false);
}
//----------------------------------------------------------------------------//
inline auto_timer::~auto_timer()
{
    if(m_temp_disable && ! timing_manager::instance()->is_enabled())
        timing_manager::instance()->enable(true);

    if(m_timer)
        m_timer->stop();

    // for consistency, always decrement hash keys
    if(auto_timer::ncount() > 0)
        --auto_timer::ncount();
    auto_timer::nhash() -= m_hash;
}
//----------------------------------------------------------------------------//

} // namespace util

} // namespace NAME_TIM

//----------------------------------------------------------------------------//

typedef NAME_TIM::util::auto_timer                     auto_timer_t;
#if defined(DISABLE_TIMERS)
#   define TIMEMORY_AUTO_TIMER(str)
#else
#   define AUTO_TIMER_NAME_COMBINE(X, Y) X##Y
#   define AUTO_TIMER_NAME(Y) AUTO_TIMER_NAME_COMBINE(macro_auto_timer, Y)
#   define TIMEMORY_AUTO_TIMER(str) \
        auto_timer_t AUTO_TIMER_NAME(__LINE__)(std::string(__FUNCTION__) + std::string(str), __LINE__)
#endif

//----------------------------------------------------------------------------//

#endif

