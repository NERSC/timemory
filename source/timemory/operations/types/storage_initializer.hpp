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
 * \file timemory/operations/types/storage_initializer.hpp
 * \brief Definition for various functions for storage_initializer in operations
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
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct storage_initializer
/// \brief This operation class is used for generic storage initalization
///
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
auto
invoke_preinit(int) -> decltype(std::declval<T>().preinit(), void())
{
    T::preinit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
auto
invoke_preinit(long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
storage_initializer
storage_initializer::get()
{
    auto library_ctor = tim::get_env<bool>("TIMEMORY_LIBRARY_CTOR", true);
    if(!library_ctor)
        return storage_initializer{};

    if(!trait::runtime_enabled<T>::get())
        return storage_initializer{};

    invoke_preinit<T>(0);

    using storage_type = storage<T, typename T::value_type>;

    static auto _master = []() {
        consume_parameters(storage_type::master_instance());
        return storage_initializer{};
    }();

    static thread_local auto _worker = []() {
        consume_parameters(storage_type::instance());
        return storage_initializer{};
    }();

    consume_parameters(_master);
    return _worker;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
