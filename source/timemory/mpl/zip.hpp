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

/** \file mpl/zip.hpp
 * \headerfile mpl/zip.hpp "timemory/mpl/zip.hpp"
 * Provides routines for combining two structures into one
 *
 */

#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
namespace mpl
{
namespace zip_impl
{
//======================================================================================//

template <typename _Func, template <typename...> class _Tuple, typename... _Types,
          size_t... _Idx>
_Tuple<_Types...>
zipper(_Func&& _func, const _Tuple<_Types...>& lhs, const _Tuple<_Types...>& rhs,
       index_sequence<_Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    using result_type    = _Tuple<_Types...>;

    result_type ret{};
    auto&&      tmp = init_list_type{ (
        std::get<_Idx>(ret) = _func(std::get<_Idx>(lhs), std::get<_Idx>(rhs)), 0)... };
    consume_parameters(tmp);
    return ret;
}

//======================================================================================//

template <typename _Tp>
struct zip
{
    using type = _Tp;

    zip(const _Tp& _v)
    : m_value(_v)
    {}

    template <typename _Func>
    zip(_Func&& _func, const type& _lhs, const type& _rhs)
    {
        m_value = _func(_lhs, _rhs);
    }

    operator type() const { return m_value; }
    operator type&() { return m_value; }

    const type& get() const { return m_value; }
    type&       get() { return m_value; }

private:
    type m_value;
};

//--------------------------------------------------------------------------------------//

template <>
struct zip<std::string>
{
    using type = std::string;

    zip(const type& _v)
    : m_value(_v)
    {}

    template <typename _Func>
    zip(_Func&& _func, const type& _lhs, const type& _rhs)
    {
        m_value = _func(_lhs, _rhs);
    }

    operator type() const { return m_value; }
    operator type&() { return m_value; }

    const type& get() const { return m_value; }
    type&       get() { return m_value; }

private:
    type m_value;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
struct zip<std::array<_Tp, _N>> : std::array<_Tp, _N>
{
    using type = std::array<_Tp, _N>;

    template <typename _Func>
    zip(_Func&& _func, const type& _lhs, const type& _rhs)
    {
        for(size_t i = 0; i < _N; ++i)
            (*this)[i] = _func(_lhs.at(i), _rhs.at(i));
    }

    const type& get() const { return static_cast<const type&>(*this); }
    type&       get() { return static_cast<type&>(*this); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Alloc>
struct zip<std::vector<_Tp, _Alloc...>> : std::vector<_Tp, _Alloc...>
{
    using type = std::vector<_Tp, _Alloc...>;

    using type::push_back;

    template <typename _Func>
    zip(_Func&& _func, const type& _lhs, const type& _rhs)
    {
        for(size_t i = 0; i < std::min(_lhs.size(), _rhs.size()); ++i)
            push_back(_func(_lhs.at(i), _rhs.at(i)));
    }

    const type& get() const { return static_cast<const type&>(*this); }
    type&       get() { return static_cast<type&>(*this); }
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class _Tuple, typename... _Types>
struct zip<_Tuple<_Types...>> : _Tuple<_Types...>
{
    using type                   = _Tuple<_Types...>;
    static constexpr size_t size = sizeof...(_Types);

    template <typename _Func>
    zip(_Func&& _func, const type& _lhs, const type& _rhs)
    : type(zipper(std::forward<_Func>(_func), _lhs, _rhs, make_index_sequence<size>{}))
    {}

    const type& get() const { return static_cast<const type&>(*this); }
    type&       get() { return static_cast<type&>(*this); }
};

}  // namespace zip_impl

//======================================================================================//

template <typename _Func, typename _Tp, typename... _Types>
zip_impl::zip<decay_t<_Tp>>
zip(_Func&& _func, _Tp&& _arg0, _Types&&... _args)
{
    return zip_impl::zip<decay_t<_Tp>>(std::forward<_Func>(_func),
                                       std::forward<_Tp>(_arg0),
                                       std::forward<_Types>(_args)...);
}

}  // namespace mpl

}  // namespace tim
