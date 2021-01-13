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

#include "timemory/components/user_bundle/types.hpp"

#if !defined(TIMEMORY_USER_BUNDLE_HEADER_MODE)
#    include "timemory/components/user_bundle/components.hpp"
#    include "timemory/runtime/properties.hpp"
#endif

namespace tim
{
namespace env
{
//
user_bundle_variables_t& get_user_bundle_variables(TIMEMORY_API)
{
    static user_bundle_variables_t _instance = {
        { component::global_bundle_idx,
          { []() { return settings::global_components(); } } },
        { component::ompt_bundle_idx,
          { []() { return settings::ompt_components(); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::mpip_bundle_idx,
          { []() { return settings::mpip_components(); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::ncclp_bundle_idx,
          { []() { return settings::ncclp_components(); },
            []() { return settings::mpip_components(); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::trace_bundle_idx,
          { []() { return settings::trace_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
        { component::profiler_bundle_idx,
          { []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } },
    };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
user_bundle_variables_t& get_user_bundle_variables(project::kokkosp)
{
    static user_bundle_variables_t _instance = {
        { component::kokkosp_bundle_idx,
          { []() { return settings::kokkos_components(); },
            []() { return get_env<std::string>("TIMEMORY_KOKKOSP_COMPONENTS", ""); },
            []() { return get_env<std::string>("KOKKOS_TIMEMORY_COMPONENTS", ""); },
            []() { return settings::trace_components(); },
            []() { return settings::profiler_components(); },
            []() { return settings::components(); },
            []() { return settings::global_components(); } } }
    };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
std::vector<TIMEMORY_COMPONENT>
get_bundle_components(const std::vector<user_bundle_spec_t>& _priority)
{
    std::string _custom{};
    bool        _fallthrough = false;
    auto        _replace     = [&_fallthrough](const std::string& _key) {
        const std::string _ft  = "fallthrough";
        auto              _pos = _key.find(_ft);
        if(_pos != std::string::npos)
        {
            _fallthrough = true;
            return _key.substr(0, _pos) + _key.substr(_pos + _ft.length() + 1);
        }
        return _key;
    };

    for(const auto& itr : _priority)
    {
        auto _spec = itr();
        if(_spec.length() > 0)
        {
            if(_spec != "none" && _spec != "NONE")
                _custom += _replace(_spec);
            else
                _fallthrough = false;
            if(!_fallthrough)
                break;
        }
    }
    return tim::enumerate_components(tim::delimit(_custom));
}
//
}  // namespace env
}  // namespace tim
