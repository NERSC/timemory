// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#ifndef TIMEMORY_COMPONENTS_TIMING_COMPONENTS_CPP_
#define TIMEMORY_COMPONENTS_TIMING_COMPONENTS_CPP_ 1

#if defined(TIMEMORY_COMPONENT_SOURCE)
#    include "timemory/components/timing/components.hpp"
#    include "timemory/components/base.hpp"
#endif

#include "timemory/components/timing/backends.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#if !defined(TIMEMORY_COMPONENT_INLINE)
#    if defined(TIMEMORY_COMPONENT_SOURCE)
#        define TIMEMORY_COMPONENT_INLINE
#    else
#        define TIMEMORY_COMPONENT_INLINE inline
#    endif
#endif

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//              System Clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
system_clock::label()
{
    return "sys";
}

TIMEMORY_COMPONENT_INLINE
std::string
system_clock::description()
{
    return "CPU time spent in kernel-mode";
}

TIMEMORY_COMPONENT_INLINE
system_clock::value_type
system_clock::record() noexcept
{
    return tim::get_clock_system_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
system_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
system_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
system_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
system_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              User clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
user_clock::label()
{
    return "user";
}

TIMEMORY_COMPONENT_INLINE
std::string
user_clock::description()
{
    return "CPU time spent in user-mode";
}

TIMEMORY_COMPONENT_INLINE
user_clock::value_type
user_clock::record() noexcept
{
    return tim::get_clock_user_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
user_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
user_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
user_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
user_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              Cpu Clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
cpu_clock::label()
{
    return "cpu";
}

TIMEMORY_COMPONENT_INLINE
std::string
cpu_clock::description()
{
    return "Total CPU time spent in both user- and kernel-mode";
}

TIMEMORY_COMPONENT_INLINE
cpu_clock::value_type
cpu_clock::record() noexcept
{
    return tim::get_clock_cpu_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
cpu_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
cpu_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
cpu_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
cpu_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              Monotonic Clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
monotonic_clock::label()
{
    return "monotonic_clock";
}

TIMEMORY_COMPONENT_INLINE
std::string
monotonic_clock::description()
{
    return "Wall-clock timer which will continue to increment even while the system "
           "is asleep";
}

TIMEMORY_COMPONENT_INLINE
monotonic_clock::value_type
monotonic_clock::record() noexcept
{
    return tim::get_clock_monotonic_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
monotonic_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
monotonic_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
monotonic_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
monotonic_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              Monotonic Raw Clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
monotonic_raw_clock::label()
{
    return "monotonic_raw_clock";
}

TIMEMORY_COMPONENT_INLINE
std::string
monotonic_raw_clock::description()
{
    return "Wall-clock timer unaffected by frequency or time adjustments in system "
           "time-of-day clock";
}

TIMEMORY_COMPONENT_INLINE
monotonic_raw_clock::value_type
monotonic_raw_clock::record() noexcept
{
    return tim::get_clock_monotonic_raw_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
monotonic_raw_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
monotonic_raw_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
monotonic_raw_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
monotonic_raw_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              Thread Cpu Clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
thread_cpu_clock::label()
{
    return "thread_cpu";
}

TIMEMORY_COMPONENT_INLINE
std::string
thread_cpu_clock::description()
{
    return "CPU-clock timer for the calling thread";
}

TIMEMORY_COMPONENT_INLINE
thread_cpu_clock::value_type
thread_cpu_clock::record() noexcept
{
    return tim::get_clock_thread_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
thread_cpu_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
thread_cpu_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
thread_cpu_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
thread_cpu_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              Process Cpu Clock
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
process_cpu_clock::label()
{
    return "process_cpu";
}

TIMEMORY_COMPONENT_INLINE
std::string
process_cpu_clock::description()
{
    return "CPU-clock timer for the calling process (all threads)";
}

TIMEMORY_COMPONENT_INLINE
process_cpu_clock::value_type
process_cpu_clock::record() noexcept
{
    return tim::get_clock_process_now<int64_t, ratio_t>();
}

TIMEMORY_COMPONENT_INLINE
double
process_cpu_clock::get() const noexcept
{
    return static_cast<double>(load() / static_cast<double>(ratio_t::den) *
                               base_type::get_unit());
}

TIMEMORY_COMPONENT_INLINE
double
process_cpu_clock::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
process_cpu_clock::start() noexcept
{
    value = record();
}

TIMEMORY_COMPONENT_INLINE
void
process_cpu_clock::stop() noexcept
{
    value = (record() - value);
    accum += value;
}

//--------------------------------------------------------------------------------------//
//
//              Cpu Util
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
cpu_util::label()
{
    return "cpu_util";
}

TIMEMORY_COMPONENT_INLINE
std::string
cpu_util::description()
{
    return "Percentage of CPU-clock time divided by wall-clock time";
}

TIMEMORY_COMPONENT_INLINE
cpu_util::value_type
cpu_util::record()
{
    return value_type(cpu_clock::record(), wall_clock::record());
}

TIMEMORY_COMPONENT_INLINE
double
cpu_util::get() const noexcept
{
    const auto& _data = load();
    double      denom = (_data.second > 0) ? _data.second : 1;
    double      numer = (_data.second > 0) ? _data.first : 0;
    return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
}

TIMEMORY_COMPONENT_INLINE
double
cpu_util::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
cpu_util::start() noexcept
{
    if(!m_derive)
        value = record();
}

TIMEMORY_COMPONENT_INLINE
void
cpu_util::stop() noexcept
{
    using namespace tim::component::operators;
    if(!m_derive)
    {
        value = (record() - value);
        accum += value;
    }
}

TIMEMORY_COMPONENT_INLINE
cpu_util&
cpu_util::operator+=(const this_type& rhs) noexcept
{
    accum += rhs.accum;
    value += rhs.value;
    return *this;
}

TIMEMORY_COMPONENT_INLINE
cpu_util&
cpu_util::operator-=(const this_type& rhs) noexcept
{
    accum -= rhs.accum;
    value -= rhs.value;
    return *this;
}

TIMEMORY_COMPONENT_INLINE
bool
cpu_util::assemble(const wall_clock* wc, const cpu_clock* cc) noexcept
{
    if(wc && cc)
        m_derive = true;
    return m_derive;
}

TIMEMORY_COMPONENT_INLINE
bool
cpu_util::assemble(const wall_clock* wc, const user_clock* uc,
                   const system_clock* sc) noexcept
{
    if(wc && uc && sc)
        m_derive = true;
    return m_derive;
}

TIMEMORY_COMPONENT_INLINE
bool
cpu_util::derive(const wall_clock* wc, const cpu_clock* cc) noexcept
{
    if(m_derive && wc && cc)
    {
        value.first  = cc->get_value();
        value.second = wc->get_value();
        accum += value;
        return true;
    }
    return false;
}

TIMEMORY_COMPONENT_INLINE
bool
cpu_util::derive(const wall_clock* wc, const user_clock* uc,
                 const system_clock* sc) noexcept
{
    if(m_derive && wc && uc && sc)
    {
        value.first  = uc->get_value() + sc->get_value();
        value.second = wc->get_value();
        accum += value;
        return true;
    }
    return false;
}

//--------------------------------------------------------------------------------------//
//
//              Process Cpu Util
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
process_cpu_util::label()
{
    return "proc_cpu_util";
}

TIMEMORY_COMPONENT_INLINE
std::string
process_cpu_util::description()
{
    return "Percentage of CPU-clock time divided by wall-clock time for calling "
           "process (all threads)";
}

TIMEMORY_COMPONENT_INLINE
process_cpu_util::value_type
process_cpu_util::record()
{
    return value_type(process_cpu_clock::record(), wall_clock::record());
}

TIMEMORY_COMPONENT_INLINE
double
process_cpu_util::get() const noexcept
{
    const auto& _data = load();
    double      denom = (_data.second > 0) ? _data.second : 1;
    double      numer = (_data.second > 0) ? _data.first : 0;
    return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
}

TIMEMORY_COMPONENT_INLINE
double
process_cpu_util::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
process_cpu_util::start() noexcept
{
    if(!m_derive)
        value = record();
}

TIMEMORY_COMPONENT_INLINE
void
process_cpu_util::stop() noexcept
{
    using namespace tim::component::operators;
    if(!m_derive)
    {
        value = (record() - value);
        accum += value;
    }
}

TIMEMORY_COMPONENT_INLINE
process_cpu_util&
process_cpu_util::operator+=(const this_type& rhs) noexcept
{
    accum += rhs.accum;
    value += rhs.value;
    return *this;
}

TIMEMORY_COMPONENT_INLINE
process_cpu_util&
process_cpu_util::operator-=(const this_type& rhs) noexcept
{
    accum -= rhs.accum;
    value -= rhs.value;
    return *this;
}

TIMEMORY_COMPONENT_INLINE
bool
process_cpu_util::assemble(const wall_clock* wc, const process_cpu_clock* cc) noexcept
{
    if(wc && cc)
        m_derive = true;
    return m_derive;
}

TIMEMORY_COMPONENT_INLINE
bool
process_cpu_util::derive(const wall_clock* wc, const process_cpu_clock* cc) noexcept
{
    if(m_derive && wc && cc)
    {
        value.first  = cc->get_value();
        value.second = wc->get_value();
        accum += value;
        return true;
    }
    return false;
}

//--------------------------------------------------------------------------------------//
//
//              Thread Cpu Util
//
//--------------------------------------------------------------------------------------//

TIMEMORY_COMPONENT_INLINE
std::string
thread_cpu_util::label()
{
    return "thread_cpu_util";
}

TIMEMORY_COMPONENT_INLINE
std::string
thread_cpu_util::description()
{
    return "Percentage of CPU-clock time divided by wall-clock time for calling "
           "thread";
}

TIMEMORY_COMPONENT_INLINE
thread_cpu_util::value_type
thread_cpu_util::record()
{
    return value_type(thread_cpu_clock::record(), wall_clock::record());
}

TIMEMORY_COMPONENT_INLINE
double
thread_cpu_util::get() const noexcept
{
    const auto& _data = load();
    double      denom = (_data.second > 0) ? _data.second : 1;
    double      numer = (_data.second > 0) ? _data.first : 0;
    return 100.0 * static_cast<double>(numer) / static_cast<double>(denom);
}

TIMEMORY_COMPONENT_INLINE
double
thread_cpu_util::get_display() const noexcept
{
    return get();
}

TIMEMORY_COMPONENT_INLINE
void
thread_cpu_util::start() noexcept
{
    if(!m_derive)
        value = record();
}

TIMEMORY_COMPONENT_INLINE
void
thread_cpu_util::stop() noexcept
{
    using namespace tim::component::operators;
    if(!m_derive)
    {
        value = (record() - value);
        accum += value;
    }
}

TIMEMORY_COMPONENT_INLINE
thread_cpu_util&
thread_cpu_util::operator+=(const this_type& rhs) noexcept
{
    accum += rhs.accum;
    value += rhs.value;
    return *this;
}

TIMEMORY_COMPONENT_INLINE
thread_cpu_util&
thread_cpu_util::operator-=(const this_type& rhs) noexcept
{
    accum -= rhs.accum;
    value -= rhs.value;
    return *this;
}

TIMEMORY_COMPONENT_INLINE
bool
thread_cpu_util::assemble(const wall_clock* wc, const thread_cpu_clock* cc) noexcept
{
    if(wc && cc)
        m_derive = true;
    return m_derive;
}

TIMEMORY_COMPONENT_INLINE
bool
thread_cpu_util::derive(const wall_clock* wc, const thread_cpu_clock* cc) noexcept
{
    if(m_derive && wc && cc)
    {
        value.first  = cc->get_value();
        value.second = wc->get_value();
        accum += value;
        return true;
    }
    return false;
}

}  // namespace component
}  // namespace tim

#endif  // TIMEMORY_COMPONENTS_TIMING_COMPONENTS_CPP_
