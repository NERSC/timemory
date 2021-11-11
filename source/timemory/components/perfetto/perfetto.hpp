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

#include "timemory/components/base.hpp"  // for base
#include "timemory/components/perfetto/backends.hpp"
#include "timemory/components/perfetto/types.hpp"
#include "timemory/mpl/concepts.hpp"  // for concepts::is_component
#include "timemory/mpl/types.hpp"     // for derivation_types

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//                      Component declaration
//
//--------------------------------------------------------------------------------------//

namespace component
{
/// \struct tim::component::perfetto_trace
/// \brief Component providing perfetto implementation
//
struct perfetto_trace : base<perfetto_trace, void>
{
    using TracingInitArgs = backend::perfetto::tracing_init_args;
    using TracingSession  = backend::perfetto::tracing_session;

    using value_type      = void;
    using base_type       = base<perfetto_trace, value_type>;
    using strset_t        = std::unordered_set<std::string>;
    using strset_iterator = typename strset_t::iterator;

    constexpr perfetto_trace() = default;

    struct config;

    static std::string label();
    static std::string description();
    static config&     get_config();
    static void        global_init();
    static void        global_finalize();

    template <typename Tp>
    void store(const char*, Tp, enable_if_t<std::is_integral<Tp>::value, int> = 0);

    template <typename Tp>
    void store(Tp, enable_if_t<std::is_integral<Tp>::value, int> = 0);

    void start(const char*);
    void start();
    void stop();
    void set_prefix(const char*);

    struct config
    {
        friend struct perfetto_trace;

        using session_t                = std::unique_ptr<TracingSession>;
        bool            in_process     = true;
        bool            system_backend = false;
        TracingInitArgs init_args      = {};

    private:
        session_t session{ nullptr };
    };

private:
    static TracingInitArgs& get_tracing_init_args();
    const char*             m_prefix = nullptr;
};

}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                      perfetto_trace component function defs
//
//--------------------------------------------------------------------------------------//

template <typename Tp>
void
tim::component::perfetto_trace::store(Tp _val,
                                      enable_if_t<std::is_integral<Tp>::value, int>)
{
    if(m_prefix)
        backend::perfetto::trace_counter(m_prefix, _val);
}

template <typename Tp>
void
tim::component::perfetto_trace::store(const char* _label, Tp _val,
                                      enable_if_t<std::is_integral<Tp>::value, int>)
{
    backend::perfetto::trace_counter<TIMEMORY_PERFETTO_API>(_label, _val);
}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_COMPONENT_PERFETTO_HEADER_ONLY_MODE) &&                             \
    TIMEMORY_COMPONENT_PERFETTO_HEADER_ONLY_MODE > 0
#    include "timemory/components/perfetto/perfetto.cpp"
#endif
