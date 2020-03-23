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
 * \file timemory/hash/declaration.hpp
 * \brief The declaration for the types for hash without definitions
 */

#pragma once

#include "timemory/hash/macros.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/utility/macros.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              hash
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag, typename Ptr, typename Pair>
Pair&
get_shared_ptr_pair()
{
    static auto                 _master = std::make_shared<Tp>();
    static std::atomic<int64_t> _count(0);
    static thread_local auto    _inst =
        Pair(_master, Ptr((_count++ == 0) ? nullptr : new Tp()));
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag, typename Ptr, typename Pair>
Ptr
get_shared_ptr_pair_instance()
{
    static thread_local auto& _pinst = get_shared_ptr_pair<Tp, Tag>();
    static thread_local auto& _inst  = _pinst.second.get() ? _pinst.second : _pinst.first;
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag, typename Ptr, typename Pair>
Ptr
get_shared_ptr_pair_master_instance()
{
    static auto& _pinst = get_shared_ptr_pair<Tp, Tag>();
    static auto  _inst  = _pinst.first;
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
