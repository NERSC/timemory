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
 * \file timemory/operations/types/add_statistics.hpp
 * \brief Definition for various functions for add_statistics in operations
 */

#pragma once

#include "timemory/mpl/policy.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace policy
{
//
//--------------------------------------------------------------------------------------//
//
//          Configure the default method for policy::record_statistics
//
//--------------------------------------------------------------------------------------//
//
template <typename CompT, typename Tp>
inline void
record_statistics<CompT, Tp>::apply(statistics<Tp>& _stat, const CompT& _obj)
{
    using result_type = decltype(std::declval<CompT>().get());
    static_assert(std::is_same<result_type, Tp>::value,
                  "Error! The default implementation of "
                  "'policy::record_statistics<Component, T>::apply' requires 'T' to be "
                  "the same type as the return type from 'Component::get()'");

    _stat += _obj.get();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace policy
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::add_statistics
/// \brief
///     Enabling statistics in timemory has two parts:
///     1. tim::trait::record_statistics must be set to true for component
///     2. tim::trait::statistics must set the data type of the statistics
///         - this is usually set to the data type returned from get()
///         - tuple<> is the default and will fully disable statistics unless changed
///
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_statistics
{
    using type = T;

    //----------------------------------------------------------------------------------//
    // if statistics is enabled
    //
    template <typename StatsT, typename U = type,
              enable_if_t<enabled_statistics<U, StatsT>::value, int> = 0>
    add_statistics(const U& rhs, StatsT& stats)
    {
        // for type comparison
        using incoming_t = decay_t<typename StatsT::value_type>;
        using expected_t = decay_t<typename trait::statistics<U>::type>;
        // check the incomming stat type against declared stat type
        // but allow for permissive_statistics when there is an acceptable
        // implicit conversion
        static_assert((!trait::permissive_statistics<U>::value &&
                       std::is_same<incoming_t, expected_t>::value),
                      "add_statistics was passed a data type different than declared "
                      "trait::statistics type. To disable this error, e.g. permit "
                      "implicit conversion, set trait::permissive_statistics "
                      "to true_type for component");
        using stats_policy_type = policy::record_statistics<U>;
        stats_policy_type::apply(stats, rhs);
    }

    //----------------------------------------------------------------------------------//
    // if statistics is not enabled
    //
    template <typename StatsT, typename U,
              enable_if_t<!enabled_statistics<U, StatsT>::value, int> = 0>
    add_statistics(const U&, StatsT&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
