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
//

/**
 * \headerfile "timemory/components/user_bundle/overloads.hpp"
 * Defines the user_bundle component which can be used to inject components
 * at runtime. There are very useful for dynamically assembling collections
 * of tools at runtime
 *
 */

#pragma once

#include "timemory/components/user_bundle/components.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/variadic/component_tuple.hpp"

//======================================================================================//
//
#if defined(TIMEMORY_USER_BUNDLE_SOURCE)
//
#    define TIMEMORY_USER_BUNDLE_LINKAGE(...) __VA_ARGS__
//
#else
//
#    if !defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
//
#        define TIMEMORY_USER_BUNDLE_LINKAGE(...) inline __VA_ARGS__
//
#    else
//
#        define TIMEMORY_USER_BUNDLE_LINKAGE(...) __VA_ARGS__
//
#    endif
//
#endif
//
//======================================================================================//

namespace tim
{
namespace component
{
namespace env
{
//
//--------------------------------------------------------------------------------------//
//
template <typename StringT>
auto
get_bundle_components(const std::string& custom_env, const StringT& fallback_env)
{
    using string_t = std::string;
    auto parse_env = [](string_t _env, string_t _default) {
        if(_env.length() > 0)
            return get_env<string_t>(_env, _default);
        return _default;
    };
    auto env_tool = parse_env(custom_env, parse_env(fallback_env, ""));
    auto env_enum = tim::enumerate_components(tim::delimit(env_tool));
    return env_enum;
}
}  // namespace env
//
//--------------------------------------------------------------------------------------//
//
//   declare as extern when TIMEMORY_USE_EXTERN and not TIMEMORY_USE_BUNDLE_SOURCE
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USER_BUNDLE_SOURCE)
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<global_bundle_idx, api::native_tag>::global_init(storage_type*);
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<tuple_bundle_idx, api::native_tag>::global_init(storage_type*);
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<list_bundle_idx, api::native_tag>::global_init(storage_type*);
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<mpip_bundle_idx, api::native_tag>::global_init(storage_type*);
//
#endif
//
//--------------------------------------------------------------------------------------//
//
// user_global_bundle
//
//--------------------------------------------------------------------------------------//
//
#if !(defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)) ||       \
    defined(TIMEMORY_USER_BUNDLE_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<global_bundle_idx, api::native_tag>::global_init(storage_type*)
{
    auto env_enum = env::get_bundle_components("GLOBAL_COMPONENTS", "");
    tim::configure<this_type>(env_enum);
}
//
//--------------------------------------------------------------------------------------//
//
// user_tuple_bundle
//
//--------------------------------------------------------------------------------------//
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<tuple_bundle_idx, api::native_tag>::global_init(storage_type*)
{
    auto env_enum = env::get_bundle_components("TUPLE_COMPONENTS", "GLOBAL_COMPONENTS");
    tim::configure<this_type>(env_enum);
}
//
//--------------------------------------------------------------------------------------//
//
// user_list_bundle
//
//--------------------------------------------------------------------------------------//
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<list_bundle_idx, api::native_tag>::global_init(storage_type*)
{
    auto env_enum = env::get_bundle_components("LIST_COMPONENTS", "GLOBAL_COMPONENTS");
    tim::configure<this_type>(env_enum);
}
//
//--------------------------------------------------------------------------------------//
//
// user_mpip_bundle
//
//--------------------------------------------------------------------------------------//
//
template <>
TIMEMORY_USER_BUNDLE_LINKAGE(void)
user_bundle<mpip_bundle_idx, api::native_tag>::global_init(storage_type*)
{
    auto env_enum = env::get_bundle_components("MPIP_COMPONENTS", "GLOBAL_COMPONENTS");
    tim::configure<this_type>(env_enum);
}
//
//--------------------------------------------------------------------------------------//
//
#endif
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component

//--------------------------------------------------------------------------------------//
}  // namespace tim
