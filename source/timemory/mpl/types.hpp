//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file mpl/types.hpp
 * \headerfile mpl/types.hpp "timemory/mpl/types.hpp"
 *
 * This is a pre-declaration of all the operation structs.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers.
 *
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

//======================================================================================//
//
namespace tim
{
//======================================================================================//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace operation
{
// operators
template <typename _Tp>
struct init_storage;

template <typename _Tp>
struct live_count;

template <typename _Tp>
struct set_prefix;

template <typename _Tp, typename _Scope>
struct insert_node;

template <typename _Tp>
struct pop_node;

template <typename _Tp>
struct record;

template <typename _Tp>
struct reset;

template <typename _Tp>
struct measure;

template <typename _Ret, typename _Lhs, typename _Rhs>
struct compose;

template <typename _Tp>
struct start;

template <typename _Tp>
struct priority_start;

template <typename _Tp>
struct standard_start;

template <typename _Tp>
struct stop;

template <typename _Tp>
struct priority_stop;

template <typename _Tp>
struct standard_stop;

template <typename _Tp>
struct mark_begin;

template <typename _Tp>
struct mark_end;

template <typename RetType, typename LhsType, typename RhsType>
struct compose;

template <typename _Tp>
struct plus;

template <typename _Tp>
struct minus;

template <typename _Tp>
struct multiply;

template <typename _Tp>
struct divide;

template <typename _Tp>
struct get_data;

template <typename _Tp>
struct base_printer;

template <typename _Tp>
struct print;

template <typename _Tp>
struct print_storage;

template <typename _Tp, typename _Archive>
struct serialization;

template <typename _Tp>
struct echo_measurement;

template <typename _Tp>
struct copy;

template <typename _Tp, typename _Op>
struct pointer_operator;

template <typename _Tp>
struct pointer_deleter;

template <typename _Tp>
struct pointer_counter;

template <typename _Tp>
struct set_width;

template <typename _Tp>
struct set_precision;

template <typename _Tp>
struct set_format_flags;

template <typename _Tp>
struct set_units;

}  // namespace operation

namespace policy
{
struct serialization;
struct global_init;
struct global_finalize;
struct thread_init;
struct thread_finalize;

template <typename... _Policies>
struct wrapper;

}  // namespace policy

namespace trait
{
template <typename _Tp>
struct is_available;

template <typename _Tp>
struct record_max;

template <typename _Tp>
struct array_serialization;

template <typename _Tp>
struct external_output_handling;

template <typename _Tp>
struct requires_prefix;

template <typename _Tp>
struct custom_label_printing;

template <typename _Tp>
struct custom_unit_printing;

template <typename _Tp>
struct custom_laps_printing;

template <typename _Tp>
struct start_priority;

template <typename _Tp>
struct stop_priority;

template <typename _Tp>
struct is_timing_category;

template <typename _Tp>
struct is_memory_category;

template <typename _Tp>
struct uses_timing_units;

template <typename _Tp>
struct uses_memory_units;

template <typename _Tp>
struct requires_json;

template <typename _Tp>
struct is_gotcha;

template <typename _Tp, typename _Tuple>
struct supports_args;

template <typename _Tp>
struct supports_custom_record;

template <typename _Tp>
struct iterable_measurement;

template <typename _Tp>
struct secondary_data;
}  // namespace trait

}  // namespace tim
