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
 * \file timemory/operations/types/pointer.hpp
 * \brief Definition for various functions for pointer in operations
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
///
/// \class operation::pointer_operator
/// \brief This operation class enables pointer-safety for the components created
/// on the heap (e.g. within a component_list) by ensuring other operation
/// classes are not invoked on a null pointer
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Op>
struct pointer_operator
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(pointer_operator)

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(type* obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            Op(*obj, std::forward<Args>(args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(type* obj, type* rhs, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(args)...);
    }

    // if the type is not available, never do anything
    template <typename Up                                         = Tp, typename... Args,
              enable_if_t<!(trait::is_available<Up>::value), int> = 0>
    pointer_operator(Args&&...)
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
struct pointer_deleter
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(pointer_deleter)

    explicit pointer_deleter(type*& obj) { delete obj; }
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct pointer_counter
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(pointer_counter)

    explicit pointer_counter(type* obj, uint64_t& count)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            ++count;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
