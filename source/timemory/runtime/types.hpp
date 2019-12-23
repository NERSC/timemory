// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

/** \file runtime/types.hpp
 * \headerfile runtime/types.hpp "timemory/runtime/types.hpp"
 * Declarations
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/utility/environment.hpp"

#include <unordered_map>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to generate an array of enumerations from a list of string
///      that can be subsequently used to initialize an auto_list or a component_list
///
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<wall_clock, cpu_clock, cpu_util, cuda_event>;
///
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, tim::enumerate_components({ "cpu_clock", "cpu_util"}));
///
template <typename _StringT, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
_Container<TIMEMORY_COMPONENT>
enumerate_components(const _Container<_StringT, _ExtraArgs...>& component_names);

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to initialize a auto_list or component_list from a list
///      of enumerations
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<wall_clock, cpu_clock, cpu_util, cuda_event>;
//
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, { CPU_CLOCK, CPU_UTIL });
//
///  typename... _ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <template <typename...> class _CompList, typename... _CompTypes,
          template <typename, typename...> class _Container, typename _Intp,
          typename... _ExtraArgs>
void
initialize(_CompList<_CompTypes...>&               obj,
           const _Container<_Intp, _ExtraArgs...>& components);

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to insert tools into a bundle
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_tuple<user_list_bundle>;
//
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::insert(obj.get<user_list_bundle>(), { CPU_CLOCK, CPU_UTIL });
//
///  typename... _ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle,
          template <typename, typename...> class _Container, typename _Intp,
          typename... _ExtraArgs>
void
insert(_Bundle<_Idx, _Type>& obj, const _Container<_Intp, _ExtraArgs...>& components);

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to insert tools into a bundle
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_tuple<user_list_bundle>;
///
///      tim::configure<user_list_bundle>({ CPU_CLOCK, CPU_UTIL });
///
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
//
///  typename... _ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <typename _Bundle_t, template <typename, typename...> class _Container,
          typename _Intp, typename... _ExtraArgs>
void
configure(const _Container<_Intp, _ExtraArgs...>& components);

//======================================================================================//

namespace env
{
//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          typename std::enable_if<(sizeof...(_CompTypes) > 0), int>::type = 0>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::string& env_var,
           const std::string& default_env)
{
    auto env_result = tim::get_env(env_var, default_env);
    initialize(obj, enumerate_components(tim::delimit(env_result)));
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          typename std::enable_if<(sizeof...(_CompTypes) == 0), int>::type = 0>
inline void
initialize(_CompList<_CompTypes...>&, const std::string&, const std::string&)
{}

//--------------------------------------------------------------------------------------//

template <size_t _Idx, typename _Type, template <size_t, typename> class _Bundle>
inline void
insert(_Bundle<_Idx, _Type>& obj, const std::string& env_var,
       const std::string& default_env)
{
    auto env_result = tim::get_env(env_var, default_env);
    insert(obj, enumerate_components(tim::delimit(env_result)));
}

//--------------------------------------------------------------------------------------//

template <typename _Bundle>
inline void
configure(const std::string& env_var, const std::string& default_env)
{
    auto env_result = tim::get_env(env_var, default_env);
    configure<_Bundle>(enumerate_components(tim::delimit(env_result)));
}

//--------------------------------------------------------------------------------------//

}  // namespace env

}  // namespace tim
