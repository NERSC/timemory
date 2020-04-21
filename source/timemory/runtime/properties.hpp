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

#include "timemory/components/placeholder.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/runtime/macros.hpp"
//
#include "timemory/components/factory.hpp"
#include "timemory/variadic/definition.hpp"
//

#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace runtime
{
//
//--------------------------------------------------------------------------------------//
//
template <int Idx>
using enumerator_t = typename component::enumerator<Idx>::type;
template <int I>
using make_int_sequence = std::make_integer_sequence<int, I>;
template <int... Ints>
using int_sequence         = std::integer_sequence<int, Ints...>;
using hash_type            = std::hash<std::string>;
using component_key_set_t  = std::set<std::string>;
using component_hash_map_t = std::unordered_map<std::string, int, hash_type>;
using opaque_pair_t        = std::pair<component::opaque, std::set<size_t>>;
//
//--------------------------------------------------------------------------------------//
//
static inline size_t
get_hash(std::string&& key)
{
    return ::tim::get_hash(std::forward<std::string>(key));
}
//
//--------------------------------------------------------------------------------------//
//
template <int I, typename... Args,
          enable_if_t<(component::enumerator<I>::value), int> = 0>
void
do_enumerator_generate(std::vector<opaque_pair_t>& opaque_array, int idx, Args&&... args)
{
    if(idx == I)
    {
        using type = enumerator_t<I>;
        if(!std::is_same<type, component::placeholder<component::nothing>>::value)
        {
            opaque_array.push_back(
                { component::factory::get_opaque<type>(std::forward<Args>(args)...),
                  component::factory::get_typeids<type>() });
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <int I, typename... Args,
          enable_if_t<!(component::enumerator<I>::value), int> = 0>
void
do_enumerator_generate(std::vector<opaque_pair_t>&, int, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
//                  The actual implementation of the function calls
//
//--------------------------------------------------------------------------------------//
//
template <int I, typename Tp, typename... Args,
          enable_if_t<(component::enumerator<I>::value), int> = 0>
void
do_enumerator_init(Tp& obj, int idx, Args&&... args)
{
    if(idx == I)
    {
        using type = enumerator_t<I>;
        if(!std::is_same<type, component::placeholder<component::nothing>>::value)
            obj.template initialize<type>(std::forward<Args>(args)...);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <int I, typename Tp, typename... Args,
          enable_if_t<!(component::enumerator<I>::value), int> = 0>
void
do_enumerator_init(Tp&, int, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
template <int I>
void
do_enumerator_enumerate(component_hash_map_t& _map, component_key_set_t& _set)
{
    using type = enumerator_t<I>;
    if(!std::is_same<type, component::placeholder<component::nothing>>::value)
    {
        std::string _id = component::properties<type>::id();
        if(_id != "TIMEMORY_COMPONENTS_END")
        {
            auto _add = [&](std::string _str) {
                if(_str.length() > 0)
                {
                    _str = settings::tolower(_str);
                    _map.insert({ _str, I });
                    _set.insert(_str);
                }
            };

            _add(_id);
            for(auto itr : component::properties<type>::ids())
                _add(itr);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
//                  The fold expressions to call the actual functions
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_init(Tp& obj, int idx, int_sequence<Ints...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(
        do_enumerator_init<Ints>(obj, idx, std::forward<Args>(args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_insert(Tp& obj, int idx, int_sequence<Ints...>, Args&&... args)
{
    std::vector<opaque_pair_t> opaque_array;
    TIMEMORY_FOLD_EXPRESSION(
        do_enumerator_generate<Ints>(opaque_array, idx, std::forward<Args>(args)...));
    for(auto&& itr : opaque_array)
        obj.insert(std::move(itr.first), std::move(itr.second));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_configure(int idx, int_sequence<Ints...>, Args&&... args)
{
    std::vector<opaque_pair_t> opaque_array;
    TIMEMORY_FOLD_EXPRESSION(
        do_enumerator_generate<Ints>(opaque_array, idx, std::forward<Args>(args)...));
    for(auto&& itr : opaque_array)
        Tp::configure(std::move(itr.first), std::move(itr.second));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, int... Ints, typename... Args>
void
enumerator_configure(Tp& obj, int idx, int_sequence<Ints...>, Args&&... args)
{
    std::vector<opaque_pair_t> opaque_array;
    TIMEMORY_FOLD_EXPRESSION(
        do_enumerator_generate<Ints>(opaque_array, idx, std::forward<Args>(args)...));
    for(auto&& itr : opaque_array)
        obj.configure(std::move(itr.first), std::move(itr.second));
}
//
//--------------------------------------------------------------------------------------//
//
template <int... Ints>
void
enumerator_enumerate(component_hash_map_t& _map, component_key_set_t& _set,
                     int_sequence<Ints...>)
{
    TIMEMORY_FOLD_EXPRESSION(do_enumerator_enumerate<Ints>(_map, _set));
}
//
//--------------------------------------------------------------------------------------//
//
//                      The forward declared functions
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Args>
void
initialize(Tp& obj, int idx, Args&&... args)
{
    enumerator_init(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                    std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Args>
void
insert(Tp& obj, int idx, Args&&... args)
{
    enumerator_insert(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                      std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Args>
void
configure(int idx, Args&&... args)
{
    enumerator_configure<Tp>(idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                             std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename... Args>
void
configure(Tp& obj, int idx, Args&&... args)
{
    enumerator_configure(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{},
                         std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
inline int
enumerate(std::string key)
{
    using data_t = std::tuple<component_hash_map_t, std::function<void(const char*)>>;
    static auto _data = []() {
        component_hash_map_t _map;
        component_key_set_t  _set;
        enumerator_enumerate(_map, _set, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
        std::stringstream ss;
        ss << "Valid choices are: [";
        for(auto itr = _set.begin(); itr != _set.end(); ++itr)
        {
            ss << "'" << (*itr) << "'";
            size_t _dist = std::distance(_set.begin(), itr);
            if(_dist + 1 < _set.size())
                ss << ", ";
        }
        ss << ']';
        auto _choices = ss.str();
        auto _msg     = [_choices](const char* itr) {
            fprintf(stderr, "Unknown component: '%s'. %s\n", itr, _choices.c_str());
        };
        return data_t(_map, _msg);
    }();

    auto itr = std::get<0>(_data).find(settings::tolower(key));
    if(itr != std::get<0>(_data).end())
        return itr->second;

    std::get<1>(_data)(key.c_str());
    return TIMEMORY_COMPONENTS_END;
}
//
//--------------------------------------------------------------------------------------//
//
inline int
enumerate(const char* key)
{
    return enumerate(std::string(key));
}
//
//--------------------------------------------------------------------------------------//
//
//                      Non-variadic specializations
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
initialize(Tp& obj, int idx)
{
    enumerator_init(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{});
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
insert(Tp& obj, int idx, scope::config _scope)
{
    enumerator_insert(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{}, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
configure(int idx, scope::config _scope)
{
    enumerator_configure<Tp>(idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{}, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
configure(Tp& obj, int idx, scope::config _scope)
{
    enumerator_configure(obj, idx, make_int_sequence<TIMEMORY_COMPONENTS_END>{}, _scope);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace runtime
}  // namespace tim
