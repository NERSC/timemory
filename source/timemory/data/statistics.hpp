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

#include "timemory/data/functional.hpp"
#include "timemory/data/stream.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/stl.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>

namespace tim
{
//======================================================================================//

template <typename Tp>
struct statistics
{
public:
    using value_type   = Tp;
    using compute_type = math::compute<Tp>;
    template <typename Vp>
    using compute_value_t = math::compute<Tp, Vp>;

public:
    inline statistics()                      = default;
    inline ~statistics()                     = default;
    inline statistics(const statistics&)     = default;
    inline statistics(statistics&&) noexcept = default;
    inline statistics& operator=(const statistics&) = default;
    inline statistics& operator=(statistics&&) noexcept = default;

    inline explicit statistics(const value_type& val)
    : m_cnt(1)
    , m_sum(val)
    , m_sqr(compute_type::sqr(val))
    , m_min(val)
    , m_max(val)
    {}

    inline explicit statistics(value_type&& val)
    : m_cnt(1)
    , m_sum(std::move(val))
    , m_sqr(compute_type::sqr(m_sum))
    , m_min(m_sum)
    , m_max(m_sum)
    {}

    statistics& operator=(const value_type& val)
    {
        m_cnt = 1;
        m_sum = val;
        m_min = val;
        m_max = val;
        m_sqr = compute_type::sqr(val);
        return *this;
    }

public:
    // Accumulated values
    inline int64_t           get_count() const { return m_cnt; }
    inline const value_type& get_min() const { return m_min; }
    inline const value_type& get_max() const { return m_max; }
    inline const value_type& get_sum() const { return m_sum; }
    inline const value_type& get_sqr() const { return m_sqr; }
    inline value_type        get_mean() const { return m_sum / m_cnt; }
    inline value_type        get_variance() const
    {
        if(m_cnt < 2)
        {
            auto ret = m_sum;
            compute_type::minus(ret, m_sum);
            return ret;
        }

        auto _sum = m_sum;
        auto _sqr = m_sqr;

        // lambda for equation clarity (will be inlined)
        auto compute_variance = [&]() {
            compute_type::multiply(_sum, m_sum);
            _sum /= m_cnt;
            compute_type::minus(_sqr, _sum);
            _sqr /= (m_cnt - 1);
            return _sqr;
        };
        return compute_variance();
    }

    inline value_type get_stddev() const
    {
        return compute_type::sqrt(compute_type::abs(get_variance()));
    }

    // Modifications
    inline void reset();

public:
    // Operators (value_type)
    inline statistics& operator+=(const value_type& val)
    {
        if(m_cnt == 0)
        {
            m_sum = val;
            m_sqr = compute_type::sqr(val);
            m_min = val;
            m_max = val;
        }
        else
        {
            compute_type::plus(m_sum, val);
            compute_type::plus(m_sqr, compute_type::sqr(val));
            m_min = compute_type::min(m_min, val);
            m_max = compute_type::max(m_max, val);
        }
        ++m_cnt;

        return *this;
    }

    inline statistics& operator-=(const value_type& val)
    {
        if(m_cnt > 1)
            --m_cnt;
        compute_type::minus(m_sum, val);
        compute_type::minus(m_sqr, compute_type::sqr(val));
        compute_type::minus(m_min, val);
        compute_type::minus(m_max, val);
        return *this;
    }

    inline statistics& operator*=(const value_type& val)
    {
        compute_type::multiply(m_sum, val);
        compute_type::multiply(m_sqr, compute_type::sqr(val));
        compute_type::multiply(m_min, val);
        compute_type::multiply(m_max, val);
        return *this;
    }

    inline statistics& operator/=(const value_type& val)
    {
        compute_type::divide(m_sum, val);
        compute_type::divide(m_sqr, compute_type::sqr(val));
        compute_type::divide(m_min, val);
        compute_type::divide(m_max, val);
        return *this;
    }

public:
    // Operators (this_type)
    inline statistics& operator+=(const statistics& rhs)
    {
        if(m_cnt == 0)
        {
            m_sum = rhs.m_sum;
            m_sqr = rhs.m_sqr;
            m_min = rhs.m_min;
            m_max = rhs.m_max;
        }
        else
        {
            compute_type::plus(m_sum, rhs.m_sum);
            compute_type::plus(m_sqr, rhs.m_sqr);
            m_min = compute_type::min(m_min, rhs.m_min);
            m_max = compute_type::max(m_max, rhs.m_max);
        }
        m_cnt += rhs.m_cnt;
        return *this;
    }

    // Operators (this_type)
    inline statistics& operator-=(const statistics& rhs)
    {
        if(m_cnt > 0)
        {
            compute_type::minus(m_sum, rhs.m_sum);
            compute_type::minus(m_sqr, rhs.m_sqr);
            m_min = compute_type::min(m_min, rhs.m_min);
            m_max = compute_type::max(m_max, rhs.m_max);
            // m_cnt += std::abs(m_cnt - rhs.m_cnt);
        }
        return *this;
    }

private:
    // summation of each history^1
    int64_t    m_cnt = 0;
    value_type m_sum = value_type{};
    value_type m_sqr = value_type{};
    value_type m_min = value_type{};
    value_type m_max = value_type{};

public:
    // friend operator for output
    friend std::ostream& operator<<(std::ostream& os, const statistics& obj)
    {
        using namespace tim::stl::ostream;
        os << "[sum: " << obj.get_sum() << "] [min: " << obj.get_min()
           << "] [max: " << obj.get_max() << "] [sqr: " << obj.get_sqr()
           << "] [count: " << obj.get_count() << "]";
        return os;
    }

    // friend operator for addition
    friend statistics operator+(const statistics& lhs, const statistics& rhs)
    {
        return statistics(lhs) += rhs;
    }

    friend statistics operator-(const statistics& lhs, const statistics& rhs)
    {
        return statistics(lhs) -= rhs;
    }

    template <typename Archive>
    void save(Archive& ar, const unsigned int) const
    {
        auto _mean = (m_cnt > 0) ? get_mean() : value_type{};
        ar(cereal::make_nvp("sum", m_sum), cereal::make_nvp("count", m_cnt),
           cereal::make_nvp("min", m_min), cereal::make_nvp("max", m_max),
           cereal::make_nvp("sqr", m_sqr), cereal::make_nvp("mean", _mean),
           cereal::make_nvp("stddev", get_stddev()));
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("sum", m_sum), cereal::make_nvp("min", m_min),
           cereal::make_nvp("max", m_max), cereal::make_nvp("sqr", m_sqr),
           cereal::make_nvp("count", m_cnt));
    }
};

//======================================================================================//

}  // namespace tim

namespace std
{
//--------------------------------------------------------------------------------------//

template <typename Tp>
::tim::statistics<Tp>
max(::tim::statistics<Tp> lhs, const Tp& rhs)
{
    return lhs.get_max(rhs);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
::tim::statistics<Tp>
min(::tim::statistics<Tp> lhs, const Tp& rhs)
{
    return lhs.get_min(rhs);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
::tim::statistics<tuple<>>&
operator+=(::tim::statistics<tuple<>>& _lhs, const Tp&)
{
    return _lhs;
}

//--------------------------------------------------------------------------------------//
}  // namespace std

#if !defined(_TIMEMORY_INTEL)
CEREAL_CLASS_VERSION(tim::statistics<int64_t>, 0)
CEREAL_CLASS_VERSION(tim::statistics<float>, 0)
CEREAL_CLASS_VERSION(tim::statistics<double>, 0)
#endif
