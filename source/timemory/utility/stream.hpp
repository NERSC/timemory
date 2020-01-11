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

/** \file utility/stream.hpp
 * \headerfile utility/stream.hpp "timemory/utility/stream.hpp"
 * Provides a simple stream type that generates a vector of strings for column alignment
 *
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

namespace tim
{
namespace zip_impl
{
//======================================================================================//

template <template <typename...> class _Tuple, typename... _Types, size_t... _Idx>
_Tuple<_Tuple<_Types, _Types>...>
zipper(const _Tuple<_Types...>& lhs, const _Tuple<_Types...>& rhs,
       index_sequence<_Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    using result_type    = _Tuple<_Tuple<_Types, _Types>...>;

    result_type ret{};
    auto&&      tmp = init_list_type{ (
        std::get<_Idx>(ret) = { std::get<_Idx>(lhs), std::get<_Idx>(rhs) }, 0)... };
    consume_parameters(tmp);
    return ret;
}

//======================================================================================//

template <typename... _Types>
struct zip : std::tuple<_Types...>
{
    using type = std::tuple<_Types...>;

    zip(_Types&&... _args)
    : type{ std::forward<_Types>(_args)... }
    {}
};
/*
template <typename _Lhs, typename... _LhsA, typename _Rhs, typename... _RhsA>
struct zip<std::vector<_Lhs, _LhsA...>, std::vector<_Rhs, _RhsA...>>
: std::vector<std::tuple<_Lhs, _Rhs>>
{
    using tuple_type = std::tuple<_Lhs, _Rhs>;
    using type      = std::vector<std::tuple<_Lhs, _Rhs>>;

    zip(const std::vector<_Lhs, _LhsA...>& _lhs,
        const std::vector<_Rhs, _RhsA...>& _rhs)
    {
        for(size_t i = 0; i < std::min(_lhs.size(), _rhs.size()); ++i)
            push_back(tuple_type{ _lhs.at(i), _rhs.at(i) });
    }
};
*/
template <template <typename...> class _Tuple, typename... _Types>
struct zip<_Tuple<_Types...>, _Tuple<_Types...>> : _Tuple<_Tuple<_Types, _Types>...>
{
    using type                   = _Tuple<_Tuple<_Types, _Types>...>;
    static constexpr size_t size = sizeof...(_Types);

    zip(const _Tuple<_Types...>& _lhs, const _Tuple<_Types...>& _rhs)
    : type(zipper(_lhs, _rhs, make_index_sequence<size>{}))
    {}
};

}  // namespace zip_impl

//======================================================================================//

template <typename... _Types>
zip_impl::zip<_Types...>
zip(_Types&&... _args)
{
    return zip_impl::zip<_Types...>(std::forward<_Types>(_args)...);
}

//======================================================================================//

struct stream
{
    using row_type     = std::vector<std::string>;
    using value_type   = std::vector<row_type>;
    using width_type   = std::vector<int>;
    using format_flags = std::ios_base::fmtflags;

public:
    explicit stream(char _delim = '|', char _fill = '-', format_flags _fmt = {},
                    int _width = 0, int _prec = 0, bool _center = false)
    : m_center(_center)
    , m_fill(_fill)
    , m_delim(_delim)
    , m_default_width(_width)
    , m_precision(_prec)
    , m_fmt(_fmt)
    {
        m_value.resize(1, row_type{});
    }

    bool center() const { return m_center; }
    int  precision() const { return m_precision; }
    int  width() const
    {
        return (m_width.empty()) ? 0 : m_width.at(m_value.back().size() - 1);
    }
    char         delim() const { return m_delim; }
    format_flags getf() const { return m_fmt; }

    void center(bool v) { m_center = v; }
    void precision(int v) { m_precision = v; }
    void width(int v)
    {
        if(!m_width.empty())
            m_width.at(m_value.back().size() - 1) = v;
    }
    void delim(char v) { m_delim = v; }
    void setf(format_flags v) { m_fmt = v; }

    const value_type& get() const { return m_value; }

    void clear()
    {
        m_width.clear();
        m_value.clear();
        m_value.resize(1, row_type());
    }

    void push(const std::string& v, size_t _N = 0)
    {
        auto _idx = m_index + (_N * m_scale);

        if(!(m_width.size() > _idx))
            m_width.resize(_idx + 1, m_default_width);

        if(!(m_value.back().size() > _idx))
            m_value.back().resize(_idx + 1, "");

        auto _len = v.length() + 2;
        if(_len % 2 == 0)
            _len += 1;
        m_width.at(_idx)        = std::max<int>(m_width.at(_idx), _len);
        m_value.back().at(_idx) = v;
        m_index += 1;
    }

    void add_row()
    {
        m_index = 0;
        m_value.push_back(row_type{});
    }

    template <int _Idx, typename _Tp>
    struct offset
    {
        offset(_Tp&& _val)
        : m_value(std::forward<_Tp>(_val))
        {}
        operator _Tp() const { return m_value; }

    private:
        _Tp m_value;
    };

    template <typename _Tp>
    void operator()(const _Tp& val, size_t _idx = 0)
    {
        std::stringstream ss;
        ss.setf(m_fmt);
        ss << std::setprecision(m_precision) << val;
        push(ss.str(), _idx);
    }

    template <int _Idx, typename _Tp>
    void operator()(const offset<_Idx, _Tp>& val)
    {
        std::stringstream ss;
        ss.setf(m_fmt);
        ss << std::setprecision(m_precision) << static_cast<_Tp>(val);
        push(ss.str(), _Idx);
    }

    void operator()(const char* val) { push(std::string(val)); }

    void operator()(const std::string& val) { push(val); }

    friend std::ostream& operator<<(std::ostream& os, const stream& obj)
    {
        // return if completely empty
        if(obj.m_value.empty())
            return os;

        // return if not entries
        if(obj.m_value.size() == 1 && obj.m_value.front().size() == 0)
            return os;

        std::stringstream ss;
        ss.setf(obj.getf());

        size_t n = 0;
        for(const auto& row : obj.m_value)
        {
            if(row.size() == 0)
                continue;

            auto _n = n++;
            if(_n == 1)
                obj.write_separator(ss);

            size_t row_idx = 0;
            for(const auto& itr : row)
            {
                auto idx       = row_idx++;
                auto itr_width = obj.m_width.at(idx % obj.m_width.size());

                ss << obj.m_delim << ' ';
                if(_n != 0 && !obj.m_center)
                {
                    ss << std::setw(itr_width - 2)
                       << ((idx == 0) ? std::left : std::right) << itr << ' ';
                }
                else
                {
                    int _w    = itr_width;
                    int _wbeg = (_w / 2) - (itr.length() / 2) - 2;
                    _wbeg     = std::max<int>(_wbeg, 1);
                    int _lhs  = itr.length() + _wbeg;
                    int _rhs  = (_w - 2);
                    if(_lhs >= _rhs)
                    {
                        _wbeg = ((_w - 2) - itr.length()) / 2;
                        _wbeg -= _wbeg % 2;
                    }
                    std::stringstream ssbeg;
                    ssbeg << std::setw(_wbeg) << "" << itr;
                    ss << std::left << std::setw(_w - 2) << ssbeg.str() << ' ';
                }
            }

            ss << obj.m_delim << '\n';
        }
        os << ss.str();
        return os;
    }

    void write_separator(std::ostream& os) const
    {
        std::stringstream ss;
        ss.fill(m_fill);

        for(size_t j = 0; j < m_value.size(); ++j)
        {
            for(size_t i = 0; i < m_value.at(j).size(); ++i)
            {
                auto itr_width = m_width.at(i % m_width.size());
                ss << m_delim << std::setw(itr_width) << "";
            }
            ss << m_delim << '\n';
            break;
        }
        os << ss.str();
    }

    template <typename... _Tp, template <typename...> class _Tuple, size_t... _Idx>
    static void write(stream&, const _Tuple<_Tp...>&, index_sequence<_Idx...>);

private:
    bool         m_center        = false;
    char         m_fill          = '-';
    char         m_delim         = '|';
    int          m_default_width = 0;
    int          m_precision     = 0;
    int          m_index         = 0;
    int          m_scale         = 1;
    format_flags m_fmt           = {};
    width_type   m_width         = {};
    value_type   m_value         = {};
};

//--------------------------------------------------------------------------------------//

inline stream&
operator<<(stream& os, const std::string& obj)
{
    os(obj);
    return os;
}

template <typename _Tp>
stream&
operator<<(stream& os, const _Tp& obj)
{
    os(obj);
    return os;
}

template <typename _Tp, typename... _Alloc>
stream&
operator<<(stream& os, const std::vector<_Tp, _Alloc...>& obj)
{
    for(const auto& itr : obj)
        os << itr;
    return os;
}

template <typename _Tp, size_t _N>
stream&
operator<<(stream& os, const std::array<_Tp, _N>& obj)
{
    for(const auto& itr : obj)
        os << itr;
    return os;
}

template <typename _Tp, typename _Up>
stream&
operator<<(stream& os, const std::pair<_Tp, _Up>& obj)
{
    os << obj.first;
    os << obj.second;
    return os;
}

template <typename... _Tp, template <typename...> class _Tuple,
          size_t _N = sizeof...(_Tp)>
stream&
operator<<(stream& os, const _Tuple<_Tp...>& obj)
{
    stream::write(os, obj, make_index_sequence<_N>{});
    return os;
}

template <typename... _Tp, template <typename...> class _Tuple, size_t... _Idx>
inline void
stream::write(stream& os, const _Tuple<_Tp...>& val, index_sequence<_Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    auto&& ret           = init_list_type{ (os << std::get<_Idx>(val), 0)... };
    consume_parameters(ret);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
