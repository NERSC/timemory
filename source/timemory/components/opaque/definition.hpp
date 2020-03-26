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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
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
 * \file timemory/components/opaque.hpp
 * \brief Implementation of the opaque
 */

#pragma once

#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
namespace factory
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<(concepts::is_comp_wrapper<Tp>::value), int> = 0>
static auto
create_heap_variadic(Label&& _label, bool flat, Args&&... args)
{
    return new Tp(std::forward<Label>(_label), true, flat, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<(concepts::is_auto_wrapper<Tp>::value), int> = 0>
static auto
create_heap_variadic(Label&& _label, bool flat, Args&&... args)
{
    return new Tp(std::forward<Label>(_label), flat, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<!(concepts::is_auto_wrapper<Tp>::value ||
                        concepts::is_auto_wrapper<Tp>::value),
                      int> = 0>
static auto
create_heap_variadic(Label&& _label, bool, Args&&... args)
{
    return new Tp(std::forward<Label>(_label), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
namespace hidden
{
//
//--------------------------------------------------------------------------------------//
//
static inline size_t
get_opaque_hash(const std::string& key)
{
    auto ret = std::hash<std::string>()(key);
    return ret;
}
//
//--------------------------------------------------------------------------------------//
//
//      simplify forward declaration
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific component
//
template <typename Toolset, typename... Args,
          enable_if_t<(trait::is_available<Toolset>::value &&
                       !concepts::is_wrapper<Toolset>::value),
                      int> = 0>
static auto
get_opaque(bool flat, Args&&... args)
{
    auto _typeid_hash = get_opaque_hash(demangle<Toolset>());

    auto _init = []() {
        static bool _inited = []() {
            operation::init_storage<Toolset>{};
            return true;
        }();
        consume_parameters(_inited);
    };

    auto _start = [=, &args...](const string_t& _prefix, bool argflat) {
        auto                            _hash   = add_hash_id(_prefix);
        Toolset*                        _result = new Toolset{};
        operation::set_prefix<Toolset>  _opprefix(*_result, _prefix);
        operation::reset<Toolset>       _opreset(*_result);
        operation::insert_node<Toolset> _opinsert(*_result, _hash, flat || argflat);
        operation::start<Toolset>       _opstart(*_result);
        consume_parameters(_opprefix, _opreset, _opinsert, _opstart);
        return (void*) _result;
    };

    auto _stop = [=](void* v_result) {
        Toolset*                     _result = static_cast<Toolset*>(v_result);
        operation::stop<Toolset>     _opstop(*_result);
        operation::pop_node<Toolset> _oppop(*_result);
        consume_parameters(_opstop, _oppop);
    };

    auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
        if(_hash == _typeid_hash && v_result && !ptr)
        {
            Toolset* _result = static_cast<Toolset*>(v_result);
            operation::get<Toolset>(*_result, ptr, _hash);
        }
    };

    auto _del = [=](void* v_result) {
        if(v_result)
        {
            Toolset* _result = static_cast<Toolset*>(v_result);
            delete _result;
        }
    };

    return opaque(true, _typeid_hash, _init, _start, _stop, _get, _del);
}
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific set of tools
//
template <typename Toolset, typename... Args,
          enable_if_t<(concepts::is_wrapper<Toolset>::value), int> = 0>
static auto
get_opaque(bool flat, Args&&... args)
{
    using Toolset_t = Toolset;

    if(Toolset::size() == 0)
    {
        DEBUG_PRINT_HERE("returning! %s is empty", demangle<Toolset>().c_str());
        return opaque{};
    }

    auto _typeid_hash = get_opaque_hash(demangle<Toolset>());

    auto _init = []() {};

    auto _start = [=, &args...](const string_t& _prefix, bool argflat) {
        Toolset_t* _result = create_heap_variadic<Toolset_t>(_prefix, flat || argflat,
                                                             std::forward<Args>(args)...);
        _result->start();
        return (void*) _result;
    };

    auto _stop = [=](void* v_result) {
        Toolset_t* _result = static_cast<Toolset_t*>(v_result);
        _result->stop();
    };

    auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
        if(v_result && !ptr)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->get(ptr, _hash);
        }
    };

    auto _del = [=](void* v_result) {
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            delete _result;
        }
    };

    return opaque(true, _typeid_hash, _init, _start, _stop, _get, _del);
}
//
//--------------------------------------------------------------------------------------//
//
//  If a tool is not avail or has no contents return empty opaque
//
template <typename Toolset, typename... Args,
          enable_if_t<(!trait::is_available<Toolset>::value), int> = 0>
static auto
get_opaque(bool, Args&&...)
{
    return opaque{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct opaque_typeids
{
    using result_type = std::set<size_t>;

    template <typename U = T, enable_if_t<(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        return result_type({ get_opaque_hash(demangle<T>()) });
    }

    template <typename U = T, enable_if_t<!(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        return result_type({ 0 });
    }

    template <typename U = T, enable_if_t<(trait::is_available<U>::value), int> = 0>
    static auto hash()
    {
        return get_opaque_hash(demangle<T>());
    }

    template <typename U = T, enable_if_t<!(trait::is_available<U>::value), int> = 0>
    static auto hash() -> size_t
    {
        return 0;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class Tuple, typename... T>
struct opaque_typeids<Tuple<T...>>
{
    using result_type = std::set<size_t>;

    template <typename U>
    static auto get(result_type& ret)
    {
        ret.insert(get_opaque_hash(demangle<U>()));
    }

    template <typename U                                        = Tuple<T...>,
              enable_if_t<(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        result_type ret;
        get<Tuple<T...>>(ret);
        TIMEMORY_FOLD_EXPRESSION(get<T>(ret));
        return ret;
    }

    template <typename U                                         = Tuple<T...>,
              enable_if_t<!(trait::is_available<U>::value), int> = 0>
    static auto get()
    {
        return result_type({ 0 });
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace hidden
//
//--------------------------------------------------------------------------------------//
//
//  Generic handler
//
template <typename Toolset, typename Arg, typename... Args>
static opaque
get_opaque(Arg&& arg, Args&&... args)
{
    return hidden::get_opaque<Toolset>(std::forward<Arg>(arg),
                                       std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
static opaque
get_opaque()
{
    return hidden::get_opaque<Toolset>(settings::flat_profile());
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
static std::set<size_t>
get_typeids()
{
    return hidden::opaque_typeids<Toolset>::get();
}
//
}  // namespace factory
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim