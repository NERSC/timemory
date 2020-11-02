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
 * \file timemory/operations/types/copy.hpp
 * \brief Definition for various functions for copy in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::copy
/// \brief This operation class is used for copying the object generically
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct copy
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(copy)

    template <typename Up = Tp, enable_if_t<trait::is_available<Up>::value, char> = 0>
    copy(Up& obj, const Up& rhs)
    {
        obj = Up(rhs);
    }

    template <typename Up = Tp, enable_if_t<trait::is_available<Up>::value, char> = 0>
    copy(Up*& obj, const Up* rhs)
    {
        if(rhs)
        {
            if(!obj)
                obj = new type(*rhs);
            else
                *obj = type(*rhs);
        }
    }

    template <typename Up = Tp, enable_if_t<!trait::is_available<Up>::value, char> = 0>
    copy(Up&, const Up&)
    {}

    template <typename Up = Tp, enable_if_t<!trait::is_available<Up>::value, char> = 0>
    copy(Up*&, const Up*)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
