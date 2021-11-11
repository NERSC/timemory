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

#pragma once

// perfetto forward decl types
namespace perfetto
{
struct TracingInitArgs;
struct TracingSession;
}  // namespace perfetto

#if defined(TIMEMORY_USE_PERFETTO)
#    include <perfetto.h>
#endif

namespace tim
{
namespace backend
{
namespace perfetto
{
#if defined(TIMEMORY_USE_PERFETTO)

using namespace ::perfetto;
using tracing_init_args = ::perfetto::TracingInitArgs;
using tracing_session   = ::perfetto::TracingSession;

#else

struct tracing_init_args
{};
struct tracing_session
{};

#endif

template <typename ApiT = TIMEMORY_PERFETTO_API, typename Tp>
void
trace_counter(const char*, Tp, enable_if_t<std::is_integral<Tp>::value, int> = 0);

template <typename ApiT = TIMEMORY_PERFETTO_API>
void
trace_event_start(const char*);

template <typename ApiT = TIMEMORY_PERFETTO_API>
void
trace_event_stop();

}  // namespace perfetto
}  // namespace backend
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                      perfetto backend functions (call macros)
//
//--------------------------------------------------------------------------------------//

template <typename ApiT, typename Tp>
void
tim::backend::perfetto::trace_counter(const char* _label, Tp _val,
                                      enable_if_t<std::is_integral<Tp>::value, int>)
{
#if defined(TIMEMORY_USE_PERFETTO)
    TRACE_COUNTER(::tim::trait::perfetto_category<ApiT>::value, _label, _val);
#else
    (void) _label;
    (void) _val;
#endif
}

template <typename ApiT>
void
tim::backend::perfetto::trace_event_start(const char* _label)
{
#if defined(TIMEMORY_USE_PERFETTO)
    TRACE_EVENT_BEGIN(::tim::trait::perfetto_category<ApiT>::value,
                      perfetto::StaticString(_label));
#else
    (void) _label;
#endif
}

template <typename ApiT>
void
tim::backend::perfetto::trace_event_stop()
{
#if defined(TIMEMORY_USE_PERFETTO)
    TRACE_EVENT_END(::tim::trait::perfetto_category<ApiT>::value);
#endif
}
