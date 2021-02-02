//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

#pragma once

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/timing/types.hpp"
#include "timemory/components/timing/wall_clock.hpp"
#include "timemory/defines.h"

#include <utility>

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
/// \struct tim::component::system_clock
/// \brief this component extracts only the CPU time spent in kernel-mode.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct system_clock : public base<system_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<system_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::user_clock
/// \brief this component extracts only the CPU time spent in user-mode.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct user_clock : public base<user_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<user_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::cpu_clock
/// \brief this component extracts only the CPU time spent in both user- and kernel- mode.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct cpu_clock : public base<cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<cpu_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::monotonic_clock
/// \brief clock that increments monotonically, tracking the time since an arbitrary
/// point, and will continue to increment while the system is asleep.
struct monotonic_clock : public base<monotonic_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<monotonic_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::monotonic_raw_clock
/// \brief clock that increments monotonically, tracking the time since an arbitrary point
/// like CLOCK_MONOTONIC.  However, this clock is unaffected by frequency or time
/// adjustments. It should not be compared to other system time sources.
struct monotonic_raw_clock : public base<monotonic_raw_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<monotonic_raw_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::thread_cpu_clock
/// \brief this clock measures the CPU time within the current thread (excludes
/// sibling/child threads).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct thread_cpu_clock : public base<thread_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<thread_cpu_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::process_cpu_clock
/// \brief this clock measures the CPU time within the current process (excludes child
/// processes).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct process_cpu_clock : public base<process_cpu_clock>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<process_cpu_clock, value_type>;

    static std::string        label();
    static std::string        description();
    static value_type         record() noexcept;
    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;
    void                      start() noexcept;
    void                      stop() noexcept;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::cpu_util
/// \brief  this computes the CPU utilization percentage for the calling process and child
/// processes.
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct cpu_util : public base<cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<cpu_util, value_type>;
    using this_type  = cpu_util;

    static std::string label();
    static std::string description();
    static value_type  record();

    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;

    void start() noexcept;
    void stop() noexcept;

    this_type& operator+=(const this_type& rhs) noexcept;
    this_type& operator-=(const this_type& rhs) noexcept;

    bool assemble(const wall_clock* wc, const cpu_clock* cc) noexcept;
    bool assemble(const wall_clock* wc, const user_clock* uc,
                  const system_clock* sc) noexcept;
    bool derive(const wall_clock* wc, const cpu_clock* cc) noexcept;
    bool derive(const wall_clock* wc, const user_clock* uc,
                const system_clock* sc) noexcept;

    TIMEMORY_NODISCARD bool is_derived() const noexcept { return m_derive; }

private:
    bool m_derive = false;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::process_cpu_util
/// \brief this computes the CPU utilization percentage for ONLY the calling process
/// (excludes child processes).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct process_cpu_util : public base<process_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<process_cpu_util, value_type>;
    using this_type  = process_cpu_util;

    static std::string label();
    static std::string description();
    static value_type  record();

    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;

    void start() noexcept;
    void stop() noexcept;

    this_type& operator+=(const this_type& rhs) noexcept;
    this_type& operator-=(const this_type& rhs) noexcept;

    bool assemble(const wall_clock* wc, const process_cpu_clock* cc) noexcept;
    bool derive(const wall_clock* wc, const process_cpu_clock* cc) noexcept;

    TIMEMORY_NODISCARD bool is_derived() const noexcept { return m_derive; }

private:
    bool m_derive = false;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::component::thread_cpu_util
/// \brief this computes the CPU utilization percentage for ONLY the calling thread
/// (excludes sibling and child threads).
/// Only relevant as a time when a different is computed
/// Do not use a single CPU time as an amount of time; it doesn't work that way.
struct thread_cpu_util : public base<thread_cpu_util, std::pair<int64_t, int64_t>>
{
    using ratio_t    = std::nano;
    using value_type = std::pair<int64_t, int64_t>;
    using base_type  = base<thread_cpu_util, value_type>;
    using this_type  = thread_cpu_util;

    static std::string label();
    static std::string description();
    static value_type  record();

    TIMEMORY_NODISCARD double get() const noexcept;
    TIMEMORY_NODISCARD double get_display() const noexcept;

    void start() noexcept;
    void stop() noexcept;

    this_type& operator+=(const this_type& rhs) noexcept;
    this_type& operator-=(const this_type& rhs) noexcept;

    bool assemble(const wall_clock* wc, const thread_cpu_clock* cc) noexcept;
    bool derive(const wall_clock* wc, const thread_cpu_clock* cc) noexcept;

    TIMEMORY_NODISCARD bool is_derived() const noexcept { return m_derive; }

private:
    bool m_derive = false;
};

}  // namespace component
}  // namespace tim

#if !defined(TIMEMORY_COMPONENT_SOURCE) && !defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    include "timemory/components/timing/components.cpp"
#endif
