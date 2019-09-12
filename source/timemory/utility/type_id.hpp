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

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"

//--------------------------------------------------------------------------------------//

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tim
{
//--------------------------------------------------------------------------------------//
//  base overload
//
template <typename _Tp>
struct type_id
{
    using string_t = std::string;
    static std::string name() { return typeid(_Tp).name(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct type_id<const _Tp>
{
    using string_t = std::string;
    static std::string name() { return string_t("K") + typeid(_Tp).name(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct type_id<const _Tp&>
{
    using string_t = std::string;
    static std::string name() { return string_t("RK") + typeid(_Tp).name(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct type_id<_Tp&>
{
    using string_t = std::string;
    static std::string name() { return string_t("R") + typeid(_Tp).name(); }
};

//--------------------------------------------------------------------------------------//

}  // namespace tim
