//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file timemory/timemory.hpp
 * \headerfile timemory/timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header
 *
 */

#pragma once

#if defined(DISABLE_TIMEMORY)

#    include <ostream>

namespace tim
{
template <typename... _Args>
void
timemory_init(_Args...)
{}
inline void
timemory_finalize()
{}
inline void
print_env()
{}

/// this provides "functionality" for *_HANDLE macros
/// and can be omitted if these macros are not utilized
struct dummy
{
    template <typename... _Types, typename... _Args>
    static void configure(_Args&&...)
    {}

    template <typename... _Args>
    dummy(_Args&&...)
    {}
    ~dummy()            = default;
    dummy(const dummy&) = default;
    dummy(dummy&&)      = default;
    dummy& operator=(const dummy&) = default;
    dummy& operator=(dummy&&) = default;

    void start() {}
    void stop() {}
    void report_at_exit(bool) {}
    template <typename... _Args>
    void mark_begin(_Args&&...)
    {}
    template <typename... _Args>
    void mark_end(_Args&&...)
    {}
    friend std::ostream& operator<<(std::ostream& os, const dummy&) { return os; }
};
}  // namespace tim

// startup/shutdown/configure
#    define TIMEMORY_INIT(...)
#    define TIMEMORY_FINALIZE()
#    define TIMEMORY_CONFIGURE(...)

// label creation
#    define TIMEMORY_BASIC_LABEL(...) std::string("")
#    define TIMEMORY_LABEL(...) std::string("")
#    define TIMEMORY_JOIN(...) std::string("")

// define an object
#    define TIMEMORY_BLANK_MARKER(...)
#    define TIMEMORY_BASIC_MARKER(...)
#    define TIMEMORY_MARKER(...)

// define an unique pointer object
#    define TIMEMORY_BLANK_POINTER(...)
#    define TIMEMORY_BASIC_POINTER(...)
#    define TIMEMORY_POINTER(...)

// define an object with a caliper reference
#    define TIMEMORY_BLANK_CALIPER(...)
#    define TIMEMORY_BASIC_CALIPER(...)
#    define TIMEMORY_CALIPER(...)

// define a static object with a caliper reference
#    define TIMEMORY_STATIC_BLANK_CALIPER(...)
#    define TIMEMORY_STATIC_BASIC_CALIPER(...)
#    define TIMEMORY_STATIC_CALIPER(...)

// invoke member function on caliper reference or type within reference
#    define TIMEMORY_CALIPER_APPLY(...)
#    define TIMEMORY_CALIPER_TYPE_APPLY(...)
#    define TIMEMORY_CALIPER_APPLY0(...)
#    define TIMEMORY_CALIPER_TYPE_APPLY0(...)
#    define TIMEMORY_CALIPER_LAMBDA(...)
#    define TIMEMORY_CALIPER_TYPE_LAMBDA(...)

// get an object
#    define TIMEMORY_BLANK_HANDLE(...) tim::dummy()
#    define TIMEMORY_BASIC_HANDLE(...) tim::dummy()
#    define TIMEMORY_HANDLE(...) tim::dummy()

// get a pointer to an object
#    define TIMEMORY_BLANK_RAW_POINTER(...) nullptr
#    define TIMEMORY_BASIC_RAW_POINTER(...) nullptr
#    define TIMEMORY_RAW_POINTER(...) nullptr

// debug only
#    define TIMEMORY_DEBUG_BLANK_MARKER(...)
#    define TIMEMORY_DEBUG_BASIC_MARKER(...)
#    define TIMEMORY_DEBUG_MARKER(...)

// auto-timers
#    define TIMEMORY_BLANK_AUTO_TIMER(...)
#    define TIMEMORY_BASIC_AUTO_TIMER(...)
#    define TIMEMORY_AUTO_TIMER(...)
#    define TIMEMORY_BLANK_AUTO_TIMER_HANDLE(...)
#    define TIMEMORY_BASIC_AUTO_TIMER_HANDLE(...)
#    define TIMEMORY_AUTO_TIMER_HANDLE(...)
#    define TIMEMORY_DEBUG_BASIC_AUTO_TIMER(...)
#    define TIMEMORY_DEBUG_AUTO_TIMER(...)

// auto-bundle (user-bundles)
#    define TIMEMORY_BLANK_AUTO_BUNDLE(...)
#    define TIMEMORY_BASIC_AUTO_BUNDLE(...)
#    define TIMEMORY_AUTO_BUNDLE(...)
#    define TIMEMORY_BLANK_AUTO_BUNDLE_HANDLE(...)
#    define TIMEMORY_BASIC_AUTO_BUNDLE_HANDLE(...)
#    define TIMEMORY_AUTO_BUNDLE_HANDLE(...)
#    define TIMEMORY_DEBUG_BASIC_AUTO_BUNDLE(...)
#    define TIMEMORY_DEBUG_AUTO_BUNDLE(...)

#else

// versioning header
#    include "timemory/version.h"

#    include "timemory/components.hpp"
#    include "timemory/manager.hpp"
#    include "timemory/settings.hpp"
#    include "timemory/units.hpp"
#    include "timemory/utility/macros.hpp"
#    include "timemory/utility/mangler.hpp"
#    include "timemory/utility/utility.hpp"
#    include "timemory/variadic/auto_hybrid.hpp"
#    include "timemory/variadic/auto_list.hpp"
#    include "timemory/variadic/auto_timer.hpp"
#    include "timemory/variadic/auto_user_bundle.hpp"
#    include "timemory/variadic/macros.hpp"

#    include "timemory/enum.h"

// definitions of types
#    include "timemory/types.hpp"
#    include "timemory/utility/bits/storage.hpp"

// allocator
#    include "timemory/ert/aligned_allocator.hpp"
#    include "timemory/ert/configuration.hpp"

//======================================================================================//

#    include "timemory/extern/auto_timer.hpp"
#    include "timemory/extern/auto_user_bundle.hpp"
#    include "timemory/extern/complete_list.hpp"
#    include "timemory/extern/ert.hpp"
#    include "timemory/extern/init.hpp"

//======================================================================================//

#    include "timemory/config.hpp"
#    include "timemory/plotting.hpp"
#    include "timemory/utility/conditional.hpp"
#    include "timemory/utility/storage.hpp"

//--------------------------------------------------------------------------------------//

#endif  // ! defined(DISABLE_TIMEMORY)
