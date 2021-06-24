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

/**
 * \file timemory/operations/types/push_node.hpp
 * \brief Definition for various functions for push_node in operations
 */

#pragma once

#include "timemory/mpl/type_traits.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/add_secondary.hpp"
#include "timemory/operations/types/add_statistics.hpp"
#include "timemory/operations/types/math.hpp"

namespace tim
{
namespace operation
{
//
template <typename Up>
static constexpr bool
storage_is_nullptr_t()
{
    return std::is_same<
        decay_t<decltype(operation::get_storage<Up>{}(std::declval<Up&>()))>,
        std::nullptr_t>::value;
}
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct push_node
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(push_node)

    TIMEMORY_HOT push_node(type& obj, scope::config _scope, hash_value_t _hash)
    {
        (*this)(obj, _scope, _hash);
    }

    TIMEMORY_HOT push_node(type& obj, scope::config _scope, const string_view_t& _key)
    : push_node(obj, _scope, get_hash_id(_key))
    {}

    TIMEMORY_HOT auto operator()(type& obj, scope::config _scope,
                                 hash_value_t _hash) const
    {
        init_storage<Tp>::init();
        return sfinae(obj, 0, 0, 0, _scope, _hash);
    }

    TIMEMORY_HOT auto operator()(type& obj, scope::config _scope,
                                 const string_view_t& _key) const
    {
        return (*this)(obj, _scope, get_hash_id(_key));
    }

private:
    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, int, Args&&... args) const
        -> decltype(obj.push(std::forward<Args>(args)...))
    {
        return obj.push(std::forward<Args>(args)...);
    }

    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, long, Args&&...) const -> decltype(obj.push())
    {
        return obj.push();
    }

    //  typical resolution: component inheriting from component::base<T, V>
    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_HOT auto sfinae(Up& _obj, int, long, long, scope::config _scope,
                             hash_value_t _hash,
                             enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(_obj.get_iterator())
    {
        using storage_type          = StorageT;
        constexpr bool force_flat_v = trait::flat_storage<Tp>::value;
        constexpr bool force_time_v = trait::timeline_storage<Tp>::value;
        if(!operation::get_is_on_stack<type, false>{}(_obj))
        {
            // reset some state
            operation::set_is_on_stack<type>{}(_obj, true);
            operation::set_is_flat<type>{}(_obj, _scope.is_flat() || force_flat_v);

            auto _storage = operation::get_storage<type>{}(_obj);
            // if storage is a nullptr, iterator is stale
            if(!_storage)
                return nullptr;

            // get the current depth
            auto _beg_depth = operation::get_depth<storage_type>{}(*_storage);
            // check against max depth when not flat
            if(!operation::get_is_flat<type, false>{}(_obj))
            {
                auto* _settings = settings::instance();
                if(_settings && _beg_depth + 1 > _settings->get_max_depth())
                {
                    operation::set_is_on_stack<type>{}(_obj, false);
                    return nullptr;
                }
            }
            // assign iterator to the insertion point in storage
            operation::set_iterator<type>{}(
                _obj, operation::insert<storage_type>{}(*_storage, _scope, _obj, _hash));
            // get the new depth
            auto _end_depth = operation::get_depth<storage_type>{}(*_storage);
            // configure the depth change state
            operation::set_depth_change<type>{}(
                _obj,
                (_beg_depth < _end_depth) || (_scope.is_timeline() || force_time_v));
            // add to stack
            operation::stack_push<storage_type>{}(*_storage, &_obj);
        }
        return _obj.get_iterator();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct pop_node
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(pop_node)

    TIMEMORY_HOT explicit pop_node(type& obj) { (*this)(obj); }

    template <typename Arg, typename... Args>
    TIMEMORY_HOT explicit pop_node(type& obj, Arg&& arg, Args&&... args)
    {
        (*this)(obj, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    TIMEMORY_HOT auto operator()(type& obj) const { return sfinae(obj, 0, 0, 0); }

    template <typename Arg, typename... Args>
    TIMEMORY_HOT auto operator()(type& obj, Arg&& arg, Args&&... args) const
    {
        return sfinae(obj, 0, 0, 0, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

private:
    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, int, Args&&... args) const
        -> decltype(obj.pop(std::forward<Args>(args)...))
    {
        return obj.pop(std::forward<Args>(args)...);
    }

    //  typical resolution: variadic bundle of components, component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, long, Args&&...) const -> decltype(obj.pop())
    {
        return obj.pop();
    }

    //  typical resolution: component inheriting from component::base<T, V>
    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    TIMEMORY_HOT auto sfinae(Up& _obj, int, long, long,
                             enable_if_t<!storage_is_nullptr_t<Up>(), int> = 0) const
        -> decltype(_obj.get_iterator())
    {
        using storage_type = StorageT;

        // return
        if(operation::get_is_invalid<Up, false>{}(_obj))
            return nullptr;

        if(operation::get_is_on_stack<type, true>{}(_obj) && _obj.get_iterator())
        {
            auto _storage = operation::get_storage<type>{}(_obj);
            assert(_storage != nullptr);

            // if storage is null, iterator is stale which means targ and stats are too
            if(!_storage)
                return nullptr;

            operation::set_is_on_stack<type>{}(_obj, false);
            auto&& itr   = _obj.get_iterator();
            type&  targ  = itr->obj();
            auto&  stats = itr->stats();

            // reset depth change and set valid state on target
            operation::set_depth_change<type>{}(_obj, false);
            operation::set_is_invalid<type>{}(targ, false);
            // add measurement to target in storage
            operation::plus<type>(targ, _obj);
            // add the secondary data
            operation::add_secondary<type>(_storage, itr, _obj);
            // update the statistics
            operation::add_statistics<type>(_obj, stats);
            // if not finalizing manipulate call-stack hierarchy
            if(!storage_type::is_finalizing())
            {
                if(operation::get_is_flat<type, false>{}(_obj))
                {
                    // just pop if flat
                    operation::stack_pop<storage_type>{}(*_storage, &_obj);
                }
                else
                {
                    // get depth and then pop
                    auto _beg_depth = operation::get_depth<storage_type>{}(_storage);
                    operation::pop<storage_type>{}(*_storage);
                    operation::stack_pop<storage_type>{}(*_storage, &_obj);
                    // get depth and determine if depth change occurred
                    auto _end_depth = operation::get_depth<storage_type>{}(_storage);
                    operation::set_depth_change<type>{}(_obj, _beg_depth > _end_depth);
                }
            }
            operation::set_is_running<type>{}(targ, false);
        }
        return _obj.get_iterator();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
