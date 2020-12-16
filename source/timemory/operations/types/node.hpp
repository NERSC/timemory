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

    TIMEMORY_DELETED_OBJECT(push_node)

    push_node(type& obj, scope::config _scope, int64_t _hash)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        sfinae(obj, 0, 0, 0, _scope, _hash);
    }

private:
    //  typical resolution: component
    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    auto sfinae(Up& _obj, int, int, int, scope::config _scope, int64_t _hash)
        -> decltype(_obj.get_is_on_stack() && _obj.get_is_flat() && _obj.get_storage() &&
                        _obj.get_depth_change(),
                    void())
    {
        using storage_type          = StorageT;
        constexpr bool force_flat_v = trait::flat_storage<Tp>::value;
        constexpr bool force_time_v = trait::timeline_storage<Tp>::value;
        if(!_obj.get_is_on_stack())
        {
            _obj.set_is_on_stack(true);
            _obj.get_is_flat() = _scope.is_flat() || force_flat_v;
            auto _storage      = static_cast<storage_type*>(_obj.get_storage());
            assert(_storage != nullptr);
            auto _beg_depth = _storage->depth();
            _obj.set_iterator(_storage->insert(_scope, _obj, _hash));
            auto _end_depth = _storage->depth();
            _obj.set_depth_change((_beg_depth < _end_depth) ||
                                  (_scope.is_timeline() || force_time_v));
            _storage->stack_push(&_obj);
        }
    }

    //  typical resolution: variadic bundle of components
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, long, Args&&... args)
        -> decltype(obj.push(std::forward<Args>(args)...), void())
    {
        obj.push(std::forward<Args>(args)...);
    }

    //  typical resolution: variadic bundle of components
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, long, Args&&...) -> decltype(obj.push(), void())
    {
        obj.push();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct pop_node
{
    using type = Tp;

    TIMEMORY_DELETED_OBJECT(pop_node)

    template <typename... Args>
    explicit pop_node(type& obj, Args&&... args)
    {
        sfinae(obj, 0, 0, 0, 0, std::forward<Args>(args)...);
    }

private:
    //  typical resolution: component
    template <typename Up, typename Vp = typename Up::value_type,
              typename StorageT = storage<Up, Vp>,
              enable_if_t<trait::uses_value_storage<Up, Vp>::value, int> = 0>
    auto sfinae(Up& _obj, int, int, int, int)
        -> decltype(_obj.get_is_on_stack() && _obj.get_depth_change() &&
                        _obj.get_storage() && _obj.get_iterator(),
                    void())
    {
        using storage_type = StorageT;
        // obj.pop_node(std::forward<Args>(args)...);
        if(_obj.get_is_on_stack() && _obj.get_iterator())
        {
            _obj.set_is_on_stack(false);
            type& targ  = _obj.get_iterator()->obj();
            auto& stats = _obj.get_iterator()->stats();
            _obj.set_depth_change(false);
            auto _storage = static_cast<storage_type*>(_obj.get_storage());
            assert(_storage != nullptr);

            if(storage_type::is_finalizing())
            {
                operation::plus<type>(targ, _obj);
                if(_storage)
                    operation::add_secondary<type>(_storage, _obj.get_iterator(), _obj);
                operation::add_statistics<type>(_obj, stats);
            }
            else if(_obj.get_is_flat())
            {
                operation::plus<type>(targ, _obj);
                if(_storage)
                    operation::add_secondary<type>(_storage, _obj.get_iterator(), _obj);
                operation::add_statistics<type>(_obj, stats);
                if(_storage)
                    _storage->stack_pop(&_obj);
            }
            else
            {
                auto _beg_depth = (_storage) ? _storage->depth() : 0;
                operation::plus<type>(targ, _obj);
                if(_storage)
                    operation::add_secondary<type>(_storage, _obj.get_iterator(), _obj);
                operation::add_statistics<type>(_obj, stats);
                if(_storage)
                {
                    _storage->pop();
                    _storage->stack_pop(&_obj);
                    auto _end_depth = _storage->depth();
                    _obj.set_depth_change(_beg_depth > _end_depth);
                }
            }
            targ.set_is_running(false);
        }
    }

    //  typical resolution: component
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, int, long, Args&&...)
        -> decltype(obj.pop_node(), void())
    {
        obj.pop_node();
    }

    //  typical resolution: variadic bundle of components
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, long, long, Args&&... args)
        -> decltype(obj.pop(std::forward<Args>(args)...), void())
    {
        obj.pop(std::forward<Args>(args)...);
    }

    //  typical resolution: variadic bundle of components
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, long, long, Args&&...) -> decltype(obj.pop(), void())
    {
        obj.pop();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, long, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
