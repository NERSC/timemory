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

#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/type_traits.hpp"

#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace tim
{
namespace math
{
//--------------------------------------------------------------------------------------//
//
//      Combining daughter data
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
combine(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    apply<void>::plus(lhs, rhs);
}

//--------------------------------------------------------------------------------------//

template <
    typename _Tp, typename... _ExtraArgs,
    template <typename, typename...> class _Container,
    typename _TupleA = _Container<_Tp, _ExtraArgs...>,
    typename _TupleB = std::tuple<_Tp, _ExtraArgs...>,
    typename std::enable_if<!(std::is_same<_TupleA, _TupleB>::value), int>::type = 0>
void
combine(_Container<_Tp, _ExtraArgs...>& lhs, const _Container<_Tp, _ExtraArgs...>& rhs)
{
    auto len = std::min(lhs.size(), rhs.size());
    for(decltype(len) i = 0; i < len; ++i)
        lhs[i] += rhs[i];
}

//--------------------------------------------------------------------------------------//

template <typename _Key, typename _Mapped, typename... _ExtraArgs>
void
combine(std::map<_Key, _Mapped, _ExtraArgs...>&       lhs,
        const std::map<_Key, _Mapped, _ExtraArgs...>& rhs)
{
    for(auto itr : rhs)
    {
        if(lhs.find(itr.first) != lhs.end())
            lhs.find(itr.first)->second += itr.second;
        else
            lhs[itr.first] = itr.second;
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(!std::is_class<_Tp>::value), int>::type = 0>
void
combine(_Tp& lhs, const _Tp& rhs)
{
    lhs += rhs;
}

//--------------------------------------------------------------------------------------//
//
//      Computing percentage that excludes daughters
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>
compute_percentage(const std::tuple<_Types...>& _lhs, const std::tuple<_Types...>& _rhs)
{
    std::tuple<_Types...> _ret;
    apply<void>::percent_diff(_ret, _lhs, _rhs);
    return _ret;
}

//--------------------------------------------------------------------------------------//

template <
    typename _Tp, typename... _ExtraArgs,
    template <typename, typename...> class _Container, typename _Ret = _Tp,
    typename _TupleA = _Container<_Tp, _ExtraArgs...>,
    typename _TupleB = std::tuple<_Tp, _ExtraArgs...>,
    typename std::enable_if<!(std::is_same<_TupleA, _TupleB>::value), int>::type = 0>
_Container<_Ret>
compute_percentage(const _Container<_Tp, _ExtraArgs...>& lhs,
                   const _Container<_Tp, _ExtraArgs...>& rhs)
{
    auto             len = std::min(lhs.size(), rhs.size());
    _Container<_Ret> perc(len, 0.0);
    for(decltype(len) i = 0; i < len; ++i)
    {
        if(rhs[i] > 0)
            perc[i] = (1.0 - (lhs[i] / rhs[i])) * 100.0;
    }
    return perc;
}

//--------------------------------------------------------------------------------------//

template <typename _Key, typename _Mapped, typename... _ExtraArgs>
std::map<_Key, _Mapped, _ExtraArgs...>
compute_percentage(const std::map<_Key, _Mapped, _ExtraArgs...>& lhs,
                   const std::map<_Key, _Mapped, _ExtraArgs...>& rhs)
{
    std::map<_Key, _Mapped, _ExtraArgs...> perc;
    for(auto itr : lhs)
    {
        if(rhs.find(itr.first) != rhs.end())
        {
            auto ritr = rhs.find(itr.first)->second;
            if(itr.second > 0)
                perc[itr.first] = (1.0 - (itr.second / ritr)) * 100.0;
        }
    }
    return perc;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Ret = _Tp,
          typename std::enable_if<(!std::is_class<_Tp>::value), int>::type = 0>
_Ret
compute_percentage(_Tp& lhs, const _Tp& rhs)
{
    return (rhs > 0) ? ((1.0 - (lhs / rhs)) * 100.0) : 0.0;
}

//--------------------------------------------------------------------------------------//
//
//      Printing percentage that excludes daughters
//
//--------------------------------------------------------------------------------------//

template <
    typename _Tp, typename... _ExtraArgs,
    template <typename, typename...> class _Container,
    typename _TupleA = _Container<_Tp, _ExtraArgs...>,
    typename _TupleB = std::tuple<_Tp, _ExtraArgs...>,
    typename std::enable_if<!(std::is_same<_TupleA, _TupleB>::value), int>::type = 0>
void
print_percentage(std::ostream& os, const _Container<_Tp, _ExtraArgs...>& obj)
{
    // negative values appear when multiple threads are involved.
    // This needs to be addressed
    for(size_t i = 0; i < obj.size(); ++i)
        if(obj[i] < 0.0 || !is_finite(obj[i]))
            return;

    std::stringstream ss;
    ss << "(exclusive: ";
    for(size_t i = 0; i < obj.size(); ++i)
    {
        ss << std::setprecision(1) << std::fixed << std::setw(5) << obj[i] << "%";
        if(i + 1 < obj.size())
            ss << ", ";
    }
    ss << ")";
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename _Key, typename _Mapped, typename... _ExtraArgs>
void
print_percentage(std::ostream& os, const std::map<_Key, _Mapped, _ExtraArgs...>& obj)
{
    // negative values appear when multiple threads are involved.
    // This needs to be addressed
    for(auto itr = obj.begin(); itr != obj.end(); ++itr)
        if(itr->second < 0.0 || !is_finite(itr->second))
            return;

    std::stringstream ss;
    ss << "(exclusive: ";
    for(auto itr = obj.begin(); itr != obj.end(); ++itr)
    {
        size_t i = std::distance(obj.begin(), itr);
        ss << std::setprecision(1) << std::fixed << std::setw(5) << itr->second << "%";
        if(i + 1 < obj.size())
            ss << ", ";
    }
    ss << ")";
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
print_percentage(std::ostream&, const std::tuple<_Types...>&)
{}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(!std::is_class<_Tp>::value), int>::type = 0>
void
print_percentage(std::ostream& os, const _Tp& obj)
{
    // negative values appear when multiple threads are involved.
    // This needs to be addressed
    if(obj < 0.0 || !is_finite(obj))
        return;

    std::stringstream ss;
    ss << "(exclusive: ";
    ss << std::setprecision(1) << std::fixed << std::setw(5) << obj;
    ss << "%)";
    os << ss.str();
}

}  // namespace math
}  // namespace tim
