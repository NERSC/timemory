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

/** \file timemory/data/statistics.hpp
 * \headerfile timemory/data/statistics.hpp "timemory/data/statistics.hpp"
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
#include "timemory/mpl/math.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"

namespace tim
{
//======================================================================================//

template <typename _Tp>
struct statistics
{
public:
    using value_type   = _Tp;
    using compute_type = math::compute<_Tp>;

public:
    inline statistics()                  = default;
    inline ~statistics()                 = default;
    inline statistics(const statistics&) = default;
    inline statistics(statistics&&)      = default;

    inline statistics(const value_type& val)
    : m_cnt(0)
    , m_sum(val)
    , m_min(val)
    , m_max(val)
    {}

    inline statistics(value_type&& val)
    : m_cnt(0)
    , m_sum(std::move(val))
    , m_min(m_sum)
    , m_max(m_sum)
    {}

    statistics& operator=(const statistics&) = default;
    statistics& operator=(statistics&&) = default;

    statistics& operator=(const value_type& val)
    {
        m_sum = val;
        if(m_cnt == 0)
            m_min = val;
        else
            m_min = compute_type::min(m_min, val);
        m_max = compute_type::max(m_max, val);
        return *this;
    }

public:
    // Accumulated values
    inline const value_type& get_min() const { return m_min; }
    inline const value_type& get_max() const { return m_max; }
    inline const value_type& get_sum() const { return m_sum; }

    // Conversion
    inline operator const value_type&() const { return m_sum; }
    inline operator value_type&() { return m_sum; }

    // Modifications
    inline void reset();

    inline statistics& get_min(const value_type& val)
    {
        m_sum = compute_type::min(m_sum, val);
        if(m_cnt == 0)
            m_min = val;
        else
            m_min = compute_type::min(m_min, val);
        return *this;
    }

    inline statistics& get_max(const value_type& val)
    {
        m_sum = compute_type::max(m_sum, val);
        m_max = compute_type::max(m_max, val);
        return *this;
    }

public:
    // Operators (value_type)
    inline statistics& operator+=(const value_type& val)
    {
        compute_type::plus(m_sum, val);
        if(m_cnt == 0)
            m_min = val;
        else
            m_min = compute_type::min(m_min, val);
        m_max = compute_type::max(m_max, val);
        ++m_cnt;
        return *this;
    }

    inline statistics& operator-=(const value_type& val)
    {
        compute_type::minus(m_sum, val);
        compute_type::minus(m_min, val);
        compute_type::minus(m_max, val);
        return *this;
    }

    inline statistics& operator*=(const value_type& val)
    {
        compute_type::multiply(m_sum, val);
        compute_type::multiply(m_min, val);
        compute_type::multiply(m_max, val);
        return *this;
    }

    inline statistics& operator/=(const value_type& val)
    {
        compute_type::divide(m_sum, val);
        compute_type::divide(m_min, val);
        compute_type::divide(m_max, val);
        return *this;
    }

public:
    // Operators (this_type)
    inline statistics& operator+=(const statistics& rhs)
    {
        m_sum += rhs.m_sum;
        if(m_cnt == 0)
            m_min = rhs.m_min;
        else
            m_min = compute_type::min(m_min, rhs.m_min);
        m_max = compute_type::max(m_max, rhs.m_max);
        m_cnt += rhs.m_cnt;
        return *this;
    }

private:
    // summation of each history^1
    int64_t    m_cnt = 0;
    value_type m_sum = value_type();
    value_type m_min = value_type();
    value_type m_max = value_type();

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

    friend const statistics operator-(const statistics& lhs, const statistics& rhs)
    {
        return statistics(lhs) -= rhs;
    }

    template <typename _Archive>
    void serialize(_Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("sum", m_sum), cereal::make_nvp("min", m_min),
           cereal::make_nvp("max", m_max));
    }
};

//======================================================================================//

}  // namespace tim

namespace std
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
::tim::statistics<_Tp>
max(::tim::statistics<_Tp> lhs, const _Tp& rhs)
{
    return lhs.get_max(rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
::tim::statistics<_Tp>
min(::tim::statistics<_Tp> lhs, const _Tp& rhs)
{
    return lhs.get_min(rhs);
}

//--------------------------------------------------------------------------------------//
}  // namespace std
