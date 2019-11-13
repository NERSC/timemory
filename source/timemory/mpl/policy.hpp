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
 * \headerfile policy.hpp "timemory/mpl/policy.hpp"
 * Provides the template meta-programming policy types
 *
 */

#pragma once

#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/types.hpp"

namespace tim
{
namespace policy
{
// these are policy classes
struct serialization
{};
struct global_init
{};
struct global_finalize
{};
struct thread_init
{};
struct thread_finalize
{};

template <typename... _Policies>
struct wrapper
{
    using type = std::tuple<_Policies...>;

    //----------------------------------------------------------------------------------//
    //  Policy is specified
    //----------------------------------------------------------------------------------//
    template <typename _Tp, typename _Archive, typename _Polp = typename _Tp::policy_type,
              enable_if_t<(is_one_of<serialization, typename _Polp::type>::value == true),
                          int> = 0>
    static void invoke_serialize(_Archive& ar, const unsigned int ver)
    {
        _Tp::invoke_serialize(ar, ver);
    }

    template <typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
              enable_if_t<(is_one_of<global_init, typename _Polp::type>::value == true),
                          int> = 0>
    static void invoke_global_init(_Store* _store)
    {
        _Tp::invoke_global_init(_store);
    }

    template <
        typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
        enable_if_t<(is_one_of<global_finalize, typename _Polp::type>::value == true),
                    int> = 0>
    static void invoke_global_finalize(_Store* _store)
    {
        _Tp::invoke_global_finalize(_store);
    }

    template <typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
              enable_if_t<(is_one_of<thread_init, typename _Polp::type>::value == true),
                          int> = 0>
    static void invoke_thread_init(_Store* _store)
    {
        _Tp::invoke_thread_init(_store);
    }

    template <
        typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
        enable_if_t<(is_one_of<thread_finalize, typename _Polp::type>::value == true),
                    int> = 0>
    static void invoke_thread_finalize(_Store* _store)
    {
        _Tp::invoke_thread_finalize(_store);
    }

    //----------------------------------------------------------------------------------//
    //  Policy is NOT specified
    //----------------------------------------------------------------------------------//
    template <
        typename _Tp, typename _Archive, typename _Polp = typename _Tp::policy_type,
        enable_if_t<(is_one_of<serialization, typename _Polp::type>::value == false),
                    int> = 0>
    static void invoke_serialize(_Archive&, const unsigned int)
    {}

    template <typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
              enable_if_t<(is_one_of<global_init, typename _Polp::type>::value == false),
                          int> = 0>
    static void invoke_global_init(_Store*)
    {}

    template <
        typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
        enable_if_t<(is_one_of<global_finalize, typename _Polp::type>::value == false),
                    int> = 0>
    static void invoke_global_finalize(_Store*)
    {}

    template <typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
              enable_if_t<(is_one_of<thread_init, typename _Polp::type>::value == false),
                          int> = 0>
    static void invoke_thread_init(_Store*)
    {}

    template <
        typename _Tp, typename _Store, typename _Polp = typename _Tp::policy_type,
        enable_if_t<(is_one_of<thread_finalize, typename _Polp::type>::value == false),
                    int> = 0>
    static void invoke_thread_finalize(_Store*)
    {}
};

}  // namespace policy
}  // namespace tim
