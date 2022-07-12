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

#pragma once

#include "timemory/math/fwd.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace tim
{
namespace math
{
template <typename Tp, typename Up>
Tp&
assign(Tp& _lhs, Up&& _rhs)
{
    return (_lhs = std::forward<Up>(_rhs));
}

template <typename Up, typename... Tp>
std::variant<Tp...>&
assign(std::variant<Tp...>& _lhs, Up&& _rhs)
{
    using value_t = std::remove_cv_t<std::remove_reference_t<decay_t<Up>>>;
    if constexpr(std::is_same<value_t, std::variant<Tp...>>::value)
    {
        _lhs = _rhs;
    }
    else
    {
        static_assert(is_one_of<value_t, type_list<Tp...>>::value,
                      "Error! rhs type is not one of variants");

        std::visit([&_rhs](auto& _lhs_v) { assign(_lhs_v, std::forward<Up>(_rhs)); },
                   _lhs);
    }
    return _lhs;
}
}  // namespace math
}  // namespace tim
