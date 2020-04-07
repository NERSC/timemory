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
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
namespace mpl
{
namespace zip_impl
{
//======================================================================================//

template <typename FuncT, template <typename...> class _Tuple, typename... Types,
          size_t... Idx>
_Tuple<Types...>
zipper(FuncT&& _func, const _Tuple<Types...>& lhs, const _Tuple<Types...>& rhs,
       index_sequence<Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    using result_type    = _Tuple<Types...>;

    result_type ret{};
    auto&&      tmp = init_list_type{ (
        std::get<Idx>(ret) = _func(std::get<Idx>(lhs), std::get<Idx>(rhs)), 0)... };
    consume_parameters(tmp);
    return ret;
}

//======================================================================================//

template <typename Tp>
struct zip
{
    using type = Tp;

    zip(const Tp& _v)
    : m_value(_v)
    {}

    template <typename FuncT>
    zip(FuncT&& _func, const type& _lhs, const type& _rhs)
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

    template <typename FuncT>
    zip(FuncT&& _func, const type& _lhs, const type& _rhs)
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

template <typename Tp, size_t N>
struct zip<std::array<Tp, N>> : std::array<Tp, N>
{
    using type = std::array<Tp, N>;

    template <typename FuncT>
    zip(FuncT&& _func, const type& _lhs, const type& _rhs)
    {
        for(size_t i = 0; i < N; ++i)
            (*this)[i] = _func(_lhs.at(i), _rhs.at(i));
    }

    const type& get() const { return static_cast<const type&>(*this); }
    type&       get() { return static_cast<type&>(*this); }
};

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... _Alloc>
struct zip<std::vector<Tp, _Alloc...>> : std::vector<Tp, _Alloc...>
{
    using type = std::vector<Tp, _Alloc...>;

    using type::push_back;

    template <typename FuncT>
    zip(FuncT&& _func, const type& _lhs, const type& _rhs)
    {
        for(size_t i = 0; i < std::min(_lhs.size(), _rhs.size()); ++i)
            push_back(_func(_lhs.at(i), _rhs.at(i)));
    }

    const type& get() const { return static_cast<const type&>(*this); }
    type&       get() { return static_cast<type&>(*this); }
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class _Tuple, typename... Types>
struct zip<_Tuple<Types...>> : _Tuple<Types...>
{
    using type                   = _Tuple<Types...>;
    static constexpr size_t size = sizeof...(Types);

    template <typename FuncT>
    zip(FuncT&& _func, const type& _lhs, const type& _rhs)
    : type(zipper(std::forward<FuncT>(_func), _lhs, _rhs, make_index_sequence<size>{}))
    {}

    const type& get() const { return static_cast<const type&>(*this); }
    type&       get() { return static_cast<type&>(*this); }
};

}  // namespace zip_impl

//======================================================================================//

template <typename FuncT, typename Tp, typename... Types>
zip_impl::zip<decay_t<Tp>>
zip(FuncT&& _func, Tp&& _arg0, Types&&... _args)
{
    return zip_impl::zip<decay_t<Tp>>(std::forward<FuncT>(_func), std::forward<Tp>(_arg0),
                                      std::forward<Types>(_args)...);
}

}  // namespace mpl

}  // namespace tim
