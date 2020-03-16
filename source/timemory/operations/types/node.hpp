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
 * \file timemory/operations/types/insert_node.hpp
 * \brief Definition for various functions for insert_node in operations
 */

#pragma once

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//

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
struct insert_node
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(insert_node)

    //  has run-time optional flat storage implementation
    template <typename Up = base_type, typename T = type,
              enable_if_t<!(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(Up::implements_storage_v), int>        = 0>
    explicit insert_node(base_type& obj, const uint64_t& nhash, bool flat);

    //  has compile-time fixed flat storage implementation
    template <typename Up = base_type, typename T = type,
              enable_if_t<(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(Up::implements_storage_v), int>       = 0>
    explicit insert_node(base_type& obj, const uint64_t& nhash, bool);

    //  no storage implementation
    template <typename Up = base_type, enable_if_t<!(Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type&, const uint64_t&, bool);
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
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(pop_node)

    //  has storage implementation
    template <typename Up = base_type, enable_if_t<(Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type& obj);

    //  no storage implementation
    template <typename Up = base_type, enable_if_t<!(Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type&);
};
//
//--------------------------------------------------------------------------------------//
//
//                                  INSERT NODE
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, typename T, enable_if_t<!(trait::flat_storage<T>::value), char>,
          enable_if_t<(Up::implements_storage_v), int>>
insert_node<Tp>::insert_node(base_type& obj, const uint64_t& nhash, bool flat)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    init_storage<Tp>::init();
    if(flat)
        obj.insert_node(scope::flat{}, nhash);
    else
        obj.insert_node(scope::tree{}, nhash);
}

//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, typename T, enable_if_t<(trait::flat_storage<T>::value), char>,
          enable_if_t<(Up::implements_storage_v), int>>
insert_node<Tp>::insert_node(base_type& obj, const uint64_t& nhash, bool)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    init_storage<Tp>::init();
    obj.insert_node(scope::flat{}, nhash);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, enable_if_t<!(Up::implements_storage_v), int>>
insert_node<Tp>::insert_node(base_type&, const uint64_t&, bool)
{}
//
//--------------------------------------------------------------------------------------//
//
//                                  POP NODE
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, enable_if_t<(Up::implements_storage_v), int>>
pop_node<Tp>::pop_node(base_type& obj)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    obj.pop_node();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, enable_if_t<!(Up::implements_storage_v), int>>
pop_node<Tp>::pop_node(base_type&)
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
