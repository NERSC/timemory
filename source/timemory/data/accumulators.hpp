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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file accumulators.hpp
 * \headerfile accumulators.hpp "timemory/utility/accumulators.hpp"
 * This provides accumulation capabilities
 *
 */

#pragma once

//----------------------------------------------------------------------------//

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>

#include "timemory/data/functional.hpp"
#include "timemory/utility/macros.hpp"

namespace tim
{
namespace impl
{
template <typename _Tp>
struct statistics
{
public:
    using value_type = _Tp;

public:
    inline statistics();
    inline ~statistics()                 = default;
    inline statistics(const statistics&) = default;
    inline statistics(statistics&&)      = default;

    statistics& operator=(const statistics&) = default;
    statistics& operator=(statistics&&) = default;

public:
    // Accumulated values
    inline const value_type& get_min() const;
    inline const value_type& get_max() const;
    inline const value_type& get_sum() const;

    // Conversion
    inline operator value_type() const { return m_sum; }

    // Modifications
    inline void reset();

    // Operators
    inline statistics& operator+=(const value_type& val)
    {
        m_sum += val;
        m_min = std::min(m_min, val);
        m_max = std::max(m_max, val);
        return *this;
    }

    inline statistics& operator+=(const statistics& rhs)
    {
        m_sum += rhs.m_sum;
        m_min = std::min(m_min, rhs.m_min);
        m_max = std::max(m_max, rhs.m_max);
        return *this;
    }

private:
    // summation of each history^1
    value_type m_min = value_type();
    value_type m_max = value_type();
    value_type m_sum = value_type();

public:
    // friend operator for output
    friend std::ostream& operator<<(std::ostream& os, const statistics& obj)
    {
        os << obj.get_sum() << " " << obj.get_min() << " " << obj.get_max();
        return os;
    }

    // friend operator for addition
    friend const statistics operator+(const statistics& lhs, const statistics& rhs)
    {
        return statistics(lhs) += rhs;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline const _Tp&
statistics<_Tp>::get_min() const
{
    return m_min;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline const _Tp&
statistics<_Tp>::get_max() const
{
    return m_max;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline const _Tp&
statistics<_Tp>::get_sum() const
{
    return m_sum;
}

//--------------------------------------------------------------------------------------//

}  // namespace impl

//======================================================================================//

template <typename _Tp>
struct statistics : public impl::statistics<_Tp>
{};

template <typename _Tp, size_t _N>
struct statistics<std::array<_Tp, _N>> : public std::array<impl::statistics<_Tp>, _N>
{};

template <typename _Tp>
struct statistics<std::vector<_Tp>> : public std::vector<impl::statistics<_Tp>>
{};

//======================================================================================//

}  // namespace tim
