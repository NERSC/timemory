// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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

/** \file usage.hpp
 * \headerfile usage.hpp "timemory/usage.hpp"
 * Resident set size handler
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>

#include "timemory/apply.hpp"
#include "timemory/data_types.hpp"
#include "timemory/formatters.hpp"
#include "timemory/macros.hpp"
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/string.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//

template <typename... Types>
class base_usage
{
public:
    typedef base_usage<Types...> this_type;
    typedef intmax_t             size_type;

    template <typename _Tp>
    using base_delta = internal::base_delta<_Tp>;
    template <typename _Tp>
    using base_data = internal::base_data<_Tp>;
    //
    typedef std::tuple<base_delta<Types>...> data_accum_t;
    typedef std::tuple<Types...>             data_t;
    //

public:
    explicit base_usage() {}

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    base_usage(const base_usage& rhs) = default;
    base_usage& operator=(const base_usage& rhs) = default;
    base_usage(base_usage&&)                     = default;
    base_usage& operator=(base_usage&&) = default;

public:
    this_type& record()
    {
        // everything is bytes
        apply<void>::once(m_data);
        return *this;
    }
    this_type& record(const this_type& rhs)
    {
        // everything is bytes
        apply<void>::once(m_data, rhs.m_data);
        return *this;
    }
    this_type record() const
    {
        auto _data = m_data;
        apply<void>::once(_data);
        return _data;
    }
    this_type record(const this_type& rhs) const
    {
        auto _data = m_data;
        apply<void>::once(_data, rhs);
        return _data;
    }
    void reset()
    {
        typedef std::tuple<rusage::reset<Types...>> reset_t;
        apply<void>::access<reset_t>(m_data);
    }

    //--------------------------------------------------------------------------------------//
    inline void report(std::ostream& os, bool endline, bool ign_cutoff) const
    {
        // stop, if not already stopped
        // if(m_data.running())
        //    const_cast<base_usage*>(this)->stop();

        std::stringstream ss;
        // ss << (*m_format)(this);

        if(endline)
            ss << std::endl;

        // ensure thread-safety
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        // output to ostream
        os << ss.str();
    }

public:
    /*
    //------------------------------------------------------------------------//
    //          operator <
    //------------------------------------------------------------------------//
    friend bool operator<(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.m_peak == rhs.m_peak) ? (lhs.m_curr < rhs.m_curr)
                                          : (lhs.m_peak < rhs.m_peak);
    }
    //------------------------------------------------------------------------//
    //          operator ==
    //------------------------------------------------------------------------//
    friend bool operator==(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.m_peak == rhs.m_peak) && (lhs.m_curr == rhs.m_curr);
    }
    //------------------------------------------------------------------------//
    //          operator !=
    //------------------------------------------------------------------------//
    friend bool operator!=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs == rhs);
    }
    //------------------------------------------------------------------------//
    //          operator >
    //------------------------------------------------------------------------//
    friend bool operator>(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.m_peak == rhs.m_peak) ? (lhs.m_curr > rhs.m_curr)
                                          : (lhs.m_peak > rhs.m_peak);
    }
    //------------------------------------------------------------------------//
    //          operator <=
    //------------------------------------------------------------------------//
    friend bool operator<=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs > rhs);
    }
    //------------------------------------------------------------------------//
    //          operator >=
    //------------------------------------------------------------------------//
    friend bool operator>=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs < rhs);
    }
    //------------------------------------------------------------------------//
    //          operator ()
    //------------------------------------------------------------------------//
    bool operator()(const this_type& rhs) const { return (*this < rhs); }
    //------------------------------------------------------------------------//
    //          operator +
    //------------------------------------------------------------------------//
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type r = lhs;
        r.m_curr += rhs.m_curr;
        r.m_peak += rhs.m_peak;
        return r;
    }
    //------------------------------------------------------------------------//
    //          operator -
    //------------------------------------------------------------------------//
    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type r = lhs;
        r.m_curr -= rhs.m_curr;
        r.m_peak -= rhs.m_peak;
        return r;
    }
    //------------------------------------------------------------------------//
    //          operator +=
    //------------------------------------------------------------------------//
    this_type& operator+=(const this_type& rhs)
    {
        m_curr += rhs.m_curr;
        m_peak += rhs.m_peak;
        return *this;
    }
    //------------------------------------------------------------------------//
    //          operator -=
    //------------------------------------------------------------------------//
    this_type& operator-=(const this_type& rhs)
    {
        m_curr -= rhs.m_curr;
        m_peak -= rhs.m_peak;
        return *this;
    }

    //------------------------------------------------------------------------//
    //          operator <<
    //------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const this_type& m)
    {
        format_type _format = (m.format().get()) ? (*(m.format().get())) : format_type();
        os << _format(&m);
        return os;
    }
    */
protected:
    // objects
    mutex_t              m_mutex;
    mutable data_t       m_data;
    mutable data_accum_t m_accum;
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
