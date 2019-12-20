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

/** \file timemory/utility/types.hpp
 * \headerfile timemory/utility/types.hpp "timemory/utility/types.hpp"
 * Declaration of types for utility directory
 *
 */

#pragma once

#include <memory>

namespace tim
{
template <typename _Tp, typename _Deleter>
class singleton;

//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//
//
template <typename StorageType>
struct storage_deleter;

//--------------------------------------------------------------------------------------//

template <typename _Tp>
using storage_singleton_t =
    singleton<_Tp, std::unique_ptr<_Tp, impl::storage_deleter<_Tp>>>;

//--------------------------------------------------------------------------------------//

template <typename Type, bool ImplementsStorage>
class storage
{};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

namespace base
{
class storage;
}

//--------------------------------------------------------------------------------------//

template <typename Type>
class storage;

//--------------------------------------------------------------------------------------//

template <typename _Tp>
impl::storage_singleton_t<_Tp>*
get_storage_singleton();

//--------------------------------------------------------------------------------------//
// clang-format off
namespace cupti { struct result; }
// clang-format on
//--------------------------------------------------------------------------------------//

namespace scope
{
// flat-scope storage
struct flat
{};

// thread-scoped storage
struct thread
{};

// process-scoped storage
struct process
{};

}  // namespace scope

//--------------------------------------------------------------------------------------//

}  // namespace tim
