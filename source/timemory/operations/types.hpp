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
 * \file timemory/operations/types.hpp
 * \brief Declare the operations types
 */

#pragma once

#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/macros.hpp"

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct statistics;
//
//--------------------------------------------------------------------------------------//
//
//                              operations
//
//--------------------------------------------------------------------------------------//
//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Up>
struct is_enabled
{
    // shorthand for available + non-void
    using Vp = typename Up::value_type;
    static constexpr bool value =
        (trait::is_available<Up>::value && !(std::is_same<Vp, void>::value));
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
using is_enabled_t = typename is_enabled<U>::type;
//
//--------------------------------------------------------------------------------------//
//
template <typename Up>
struct has_data
{
    // shorthand for non-void
    using Vp                    = typename Up::value_type;
    static constexpr bool value = (!std::is_same<Vp, void>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type>
struct check_record_type
{
    static constexpr bool value =
        (!std::is_same<V, void>::value && is_enabled<T>::value &&
         std::is_same<
             V, typename function_traits<decltype(&T::record)>::result_type>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Up, typename Vp>
struct stats_enabled
{
    using EmptyT = std::tuple<>;

    static constexpr bool value =
        (trait::record_statistics<Up>::value && !(std::is_same<Vp, void>::value) &&
         !(std::is_same<Vp, EmptyT>::value) &&
         !(std::is_same<Vp, statistics<void>>::value) &&
         !(std::is_same<Vp, statistics<EmptyT>>::value));
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U, typename StatsT>
struct enabled_statistics
{
    using EmptyT = std::tuple<>;

    static constexpr bool value =
        (trait::record_statistics<U>::value && !std::is_same<StatsT, EmptyT>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
using has_data_t = typename has_data<U>::type;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct init_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct construct;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_prefix;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_flat_profile;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_timeline_profile;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct insert_node;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pop_node;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct record;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct reset;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct measure;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct sample;
//
//--------------------------------------------------------------------------------------//
//
template <typename Ret, typename Lhs, typename Rhs>
struct compose;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct priority_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct standard_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct delayed_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct priority_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct standard_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct delayed_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark_begin;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark_end;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct store;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct audit;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct plus;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct minus;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct multiply;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct divide;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_labeled_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct base_printer;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_header;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_statistics;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_secondary;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_statistics;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct serialization;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, bool Enabled = trait::echo_enabled<T>::value>
struct echo_measurement;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct copy;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename Op>
struct pointer_operator;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pointer_deleter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pointer_counter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename Op>
struct generic_operator;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct generic_deleter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct generic_counter;
//
//--------------------------------------------------------------------------------------//
//
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct mpi_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct upc_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct dmp_get;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
