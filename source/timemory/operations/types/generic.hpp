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
 * \file timemory/operations/types/generic.hpp
 * \brief Definition for various functions for generic in operations
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
/// \class operation::generic_operator
/// \brief This operation class is similar to pointer_operator but can handle non-pointer
/// types
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Op>
struct generic_operator
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(generic_operator)

    template <typename Up>
    static void check()
    {
        using U = std::decay_t<std::remove_pointer_t<Up>>;
        static_assert(std::is_same<U, type>::value, "Error! Up != type");
    }

    //----------------------------------------------------------------------------------//
    //
    //      Pointers
    //
    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Args&&... args)
    {
        check<Up>();
        if(obj)
        {
            Op tmp(*obj, std::forward<Args>(args)...);
            consume_parameters(tmp);
        }
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Up& rhs, Args&&... args)
    {
        check<Up>();
        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      References
    //
    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && !std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Args&&... args)
    {
        check<Up>();
        Op tmp(obj, std::forward<Args>(args)...);
        consume_parameters(tmp);
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && !std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Up& rhs, Args&&... args)
    {
        check<Up>();
        Op(obj, rhs, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      Not available
    //
    //----------------------------------------------------------------------------------//

    // if the type is not available, never do anything
    template <typename Up, typename... Args,
              enable_if_t<!(trait::is_available<Up>::value), int> = 0>
    generic_operator(Up&, Args&&...)
    {
        check<Up>();
    }
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct generic_deleter
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(generic_deleter)

    template <typename Up, enable_if_t<(std::is_pointer<Up>::value), int> = 0>
    explicit generic_deleter(Up& obj)
    {
        static_assert(std::is_same<Up, type>::value, "Error! Up != type");
        delete static_cast<type*&>(obj);
    }

    template <typename Up, enable_if_t<!(std::is_pointer<Up>::value), int> = 0>
    explicit generic_deleter(Up&)
    {
        static_assert(std::is_same<Up, type>::value, "Error! Up != type");
    }
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct generic_counter
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(generic_counter)

    template <typename Up, enable_if_t<(std::is_pointer<Up>::value), int> = 0>
    explicit generic_counter(const Up& obj, uint64_t& count)
    {
        static_assert(std::is_same<Up, type>::value, "Error! Up != type");
        count += (trait::runtime_enabled<type>::get() && obj) ? 1 : 0;
    }

    template <typename Up, enable_if_t<!(std::is_pointer<Up>::value), int> = 0>
    explicit generic_counter(const Up&, uint64_t& count)
    {
        static_assert(std::is_same<Up, type>::value, "Error! Up != type");
        count += (trait::runtime_enabled<type>::get()) ? 1 : 0;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
