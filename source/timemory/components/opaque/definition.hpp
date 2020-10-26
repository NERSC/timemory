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

#pragma once

#include "timemory/components/opaque/declaration.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"

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
          enable_if_t<concepts::is_comp_wrapper<Tp>::value, int> = 0>
static auto
create_heap_variadic(Label&& _label, scope::config _scope, Args&&... args)
{
    return new Tp(std::forward<Label>(_label), true, _scope, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Label, typename... Args,
          enable_if_t<concepts::is_auto_wrapper<Tp>::value, int> = 0>
static auto
create_heap_variadic(Label&& _label, scope::config _scope, Args&&... args)
{
    return new Tp(std::forward<Label>(_label), _scope, std::forward<Args>(args)...);
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
template <typename Toolset, enable_if_t<(trait::is_available<Toolset>::value &&
                                         !concepts::is_wrapper<Toolset>::value),
                                        int> = 0>
auto
get_opaque(scope::config _scope)
{
    auto _typeid_hash = get_opaque_hash(demangle<Toolset>());

    auto _init = []() { operation::init_storage<Toolset>(); };

    auto _setup = [=](void* v_result, const string_t& _prefix, scope::config arg_scope) {
        Toolset* _result = static_cast<Toolset*>(v_result);
        if(!_result)
            _result = new Toolset{};
        operation::set_prefix<Toolset> _opprefix(*_result, _prefix);
        operation::set_scope<Toolset>  _opscope(*_result, arg_scope);
        consume_parameters(_opprefix, _opscope);
        return (void*) _result;
    };

    auto _push = [=](void*& v_result, const string_t& _prefix, scope::config arg_scope) {
        if(v_result)
        {
            auto                          _hash   = add_hash_id(_prefix);
            Toolset*                      _result = static_cast<Toolset*>(v_result);
            operation::push_node<Toolset> _opinsert(*_result, _scope + arg_scope, _hash);
            consume_parameters(_opinsert);
        }
    };

    auto _start = [=](void* v_result) {
        if(v_result)
        {
            Toolset*                  _result = static_cast<Toolset*>(v_result);
            operation::start<Toolset> _opstart(*_result);
            consume_parameters(_opstart);
        }
    };

    auto _stop = [=](void* v_result) {
        if(v_result)
        {
            Toolset*                 _result = static_cast<Toolset*>(v_result);
            operation::stop<Toolset> _opstop(*_result);
            consume_parameters(_opstop);
        }
    };

    auto _pop = [=](void* v_result) {
        if(v_result)
        {
            Toolset*                     _result = static_cast<Toolset*>(v_result);
            operation::pop_node<Toolset> _oppop(*_result);
            consume_parameters(_oppop);
        }
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

    return opaque(true, _typeid_hash, _init, _start, _stop, _get, _del, _setup, _push,
                  _pop);
}
//
//--------------------------------------------------------------------------------------//
//
//  Configure the tool for a specific set of tools
//
template <typename Toolset, typename... Args,
          enable_if_t<concepts::is_wrapper<Toolset>::value, int> = 0>
auto
get_opaque(scope::config _scope, Args&&... args)
{
    using Toolset_t = Toolset;

    if(Toolset::size() == 0)
    {
        DEBUG_PRINT_HERE("returning! %s is empty", demangle<Toolset>().c_str());
        return opaque{};
    }

    auto _typeid_hash = get_opaque_hash(demangle<Toolset>());

    auto _init = []() {};

    auto _setup = [=, &args...](void* v_result, const string_t& _prefix,
                                scope::config arg_scope) {
        Toolset_t* _result = static_cast<Toolset_t*>(v_result);
        if(!_result)
            _result = create_heap_variadic<Toolset_t>(_prefix, _scope + arg_scope,
                                                      std::forward<Args>(args)...);
        else
            _result->rekey(_prefix);
        return (void*) _result;
    };

    auto _push = [=](void*& v_result, const string_t& _prefix, scope::config arg_scope) {
        v_result = _setup(v_result, _prefix, arg_scope);
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->push();
        }
    };

    auto _start = [=](void* v_result) {
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->start();
        }
    };

    auto _stop = [=](void* v_result) {
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->stop();
        }
    };

    auto _pop = [=](void* v_result) {
        if(v_result)
        {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->pop();
        }
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

    return opaque(true, _typeid_hash, _init, _start, _stop, _get, _del, _setup, _push,
                  _pop);
}
//
//--------------------------------------------------------------------------------------//
//
//  If a tool is not avail or has no contents return empty opaque
//
template <typename Toolset, typename... Args,
          enable_if_t<!trait::is_available<Toolset>::value, int> = 0>
auto
get_opaque(scope::config, Args&&...)
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

    template <typename U = T, enable_if_t<trait::is_available<U>::value, int> = 0>
    static auto get()
    {
        return result_type({ get_opaque_hash(demangle<T>()) });
    }

    template <typename U = T, enable_if_t<trait::is_available<U>::value, int> = 0>
    static auto hash()
    {
        return get_opaque_hash(demangle<T>());
    }

    template <typename U = T, enable_if_t<!trait::is_available<U>::value, int> = 0>
    static auto get()
    {
        return result_type({ 0 });
    }

    template <typename U = T, enable_if_t<!trait::is_available<U>::value, int> = 0>
    static auto hash() -> size_t
    {
        return 0;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... T>
struct opaque_typeids<TupleT<T...>>
{
    using result_type = std::set<size_t>;
    using this_type   = opaque_typeids<TupleT<T...>>;

#if !defined(_WINDOWS)
    template <typename U>
    static void get(result_type& ret)
    {
        ret.insert(get_opaque_hash(demangle<U>()));
    }

    template <typename U       = TupleT<T...>,
              enable_if_t<(trait::is_available<U>::value &&
                           concepts::is_wrapper<TupleT<T...>>::value),
                          int> = 0>
    static result_type get()
    {
        result_type ret;
        this_type::get<TupleT<T...>>(ret);
        TIMEMORY_FOLD_EXPRESSION(get<T>(ret));
        return ret;
    }

    template <typename U       = TupleT<T...>,
              enable_if_t<(trait::is_available<U>::value &&
                           !concepts::is_wrapper<TupleT<T...>>::value),
                          int> = 0>
    static result_type get()
    {
        result_type ret;
        this_type::get<TupleT<T...>>(ret);
        return ret;
    }
#else
    template <typename U, enable_if_t<trait::is_available<U>::value, int> = 0>
    static void get(result_type& ret)
    {
        ret.insert(get_opaque_hash(demangle<U>()));
    }

    template <typename U, enable_if_t<!trait::is_available<U>::value, int> = 0>
    static void get(result_type&)
    {}

    static result_type get()
    {
        result_type ret;
        this_type::get<TupleT<T...>>(ret);
        return ret;
    }
#endif

    template <typename U                                       = TupleT<T...>,
              enable_if_t<!trait::is_available<U>::value, int> = 0>
    static result_type get()
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
opaque
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
opaque
get_opaque()
{
    return hidden::get_opaque<Toolset>(tim::scope::get_default());
}
//
//--------------------------------------------------------------------------------------//
//
//  With scope arguments
//
template <typename Toolset>
opaque
get_opaque(scope::config _scope)
{
    return hidden::get_opaque<Toolset>(_scope);
}
//
//--------------------------------------------------------------------------------------//
//
//  With bool arguments
//
template <typename Toolset>
opaque
get_opaque(bool _flat)
{
    return hidden::get_opaque<Toolset>(scope::config{ _flat });
}
//
//--------------------------------------------------------------------------------------//
//
//  Default with no args
//
template <typename Toolset>
std::set<size_t>
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
