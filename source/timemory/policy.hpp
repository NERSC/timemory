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

/** \file policy.hpp
 * \headerfile policy.hpp "timemory/policy.hpp"
 * Provides the template meta-programming policy types
 *
 */

#pragma once

#include "timemory/apply.hpp"

namespace tim
{
namespace policy
{

// these are policy classes
struct serialization;
// struct type_addition;
struct initialization;
struct finalization;

template <typename... _Policies>
class PolicyWrapper : _Policies...
{
public:
    template <typename _Tp,
              enable_if_t<(is_one_of_v<serialization, _Policies...> == true), int> = 0>
    void apply_serialization(_Tp&& obj)
    {
        serialization::apply(std::forward<_Tp>(obj));
    }

    template <typename _Tp,
              enable_if_t<(is_one_of_v<serialization, _Policies...> == false), int> = 0>
    void apply_serialization(_Tp&&)
    {

    }

    template <typename _Tp,
              enable_if_t<(is_one_of_v<initialization, _Policies...> == true), int> = 0>
    void apply_initialization(_Tp&& obj)
    {
        initialization::apply(std::forward<_Tp>(obj));
    }

    template <typename _Tp,
              enable_if_t<(is_one_of_v<initialization, _Policies...> == false), int> = 0>
    void apply_initialization(_Tp&&)
    {

    }

    template <typename _Tp,
              enable_if_t<(is_one_of_v<finalization, _Policies...> == true), int> = 0>
    void apply_finalization(_Tp&& obj)
    {
        finalization::apply(std::forward<_Tp>(obj));
    }

    template <typename _Tp,
              enable_if_t<(is_one_of_v<finalization, _Policies...> == false), int> = 0>
    void apply_finalization(_Tp&&)
    {

    }

};

}
}
