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

#ifndef TIMEMORY_COMPONENTS_PAPI_PAPI_COMMON_CPP_
#define TIMEMORY_COMPONENTS_PAPI_PAPI_COMMON_CPP_

#include "timemory/components/papi/macros.hpp"

#if !defined(TIMEMORY_PAPI_COMPONENT_HEADER_ONLY_MODE) ||                                \
    (defined(TIMEMORY_PAPI_COMPONENT_HEADER_ONLY_MODE) &&                                \
     TIMEMORY_PAPI_COMPONENT_HEADER_ONLY_MODE < 1)
#    include "timemory/components/papi/papi_common.hpp"
#endif

namespace tim
{
namespace component
{
TIMEMORY_PAPI_INLINE papi_common::state_data&
                     papi_common::state()
{
    static thread_local state_data _instance{};
    return _instance;
}

TIMEMORY_PAPI_INLINE void
papi_common::overflow_handler(int evt_set, void* address, long long overflow_vector,
                              void* context)
{
    TIMEMORY_PRINTF(stderr, "[papi_common%i]> Overflow at %p! bit=0x%llx \n", evt_set,
                    address, overflow_vector);
    consume_parameters(context);
}

TIMEMORY_PAPI_INLINE void
papi_common::add_event(int evt)
{
    auto& pevents = private_events();
    auto  fitr    = std::find(pevents.begin(), pevents.end(), evt);
    if(fitr == pevents.end())
        pevents.push_back(evt);
}

TIMEMORY_PAPI_INLINE bool
papi_common::initialize_papi()
{
    if(!state().is_initialized && !state().is_working)
    {
        if(!settings::papi_quiet() && (settings::debug() || settings::verbose() > 2))
        {
            TIMEMORY_PRINT_HERE("Initializing papi (initialized: %s, is working: %s)",
                                state().is_initialized ? "y" : "n",
                                state().is_working ? "y" : "n");
        }
        papi::init();
        papi::register_thread();
        state().is_working = papi::working();
        // prevent recursive re-entry via get_events<void>()
        state().is_initialized = true;
        if(!state().is_working)
        {
            if(!settings::papi_quiet() && !get_events<void>().empty())
            {
                std::cerr << "Warning! PAPI failed to initialized!\n";
                std::cerr << "The following PAPI events will not be reported: \n";
                for(const auto& itr : get_events<void>())
                    std::cerr << "    " << papi::get_event_info(itr).short_descr << "\n";
                std::cerr << std::flush;
            }
            // disable all the papi APIs with concrete instantiations
            tim::trait::apply<tim::trait::runtime_enabled>::set<
                tpls::papi, papi_array_t, papi_common, papi_vector, papi_array8_t,
                papi_array16_t, papi_array32_t>(false);
            state().is_initialized = false;
        }
    }
    return state().is_initialized && state().is_working;
}

TIMEMORY_PAPI_INLINE bool
papi_common::finalize_papi()
{
    if(!state().is_finalized && state().is_working)
    {
        papi::unregister_thread();
        state().is_working = papi::working();
        if(state().is_working)
            state().is_finalized = true;
    }
    return state().is_finalized && state().is_working;
}

}  // namespace component
}  // namespace tim

#endif  // TIMEMORY_COMPONENTS_PAPI_PAPI_COMMON_H_
