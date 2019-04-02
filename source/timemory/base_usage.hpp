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
    base_usage(const base_usage& rhs)
    : m_data(rhs.m_data)
    , m_accum(rhs.m_accum)
    , m_laps(rhs.m_laps)
    {
    }
    base_usage& operator=(const base_usage& rhs)
    {
        if(this == &rhs)
            return *this;
        m_data  = rhs.m_data;
        m_accum = rhs.m_accum;
        m_laps  = rhs.m_laps;
        return *this;
    }

    base_usage(base_usage&&) = default;
    base_usage& operator=(base_usage&&) = default;

public:
    this_type& record()
    {
        apply<void>::once(m_data);
        return *this;
    }
    this_type& record(const this_type& rhs)
    {
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
    // operations
    this_type& operator-=(const this_type& rhs)
    {
        typedef std::tuple<rusage::minus<Types>...> apply_types;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        return *this;
    }
    this_type& operator-=(uintmax_t&& rhs)
    {
        typedef std::tuple<rusage::minus<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<uintmax_t>(rhs));
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        typedef std::tuple<rusage::plus<Types>...> apply_types;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        return *this;
    }
    this_type& operator+=(uintmax_t&& rhs)
    {
        typedef std::tuple<rusage::plus<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<uintmax_t>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        typedef std::tuple<rusage::multiply<Types...>> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }
    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        typedef std::tuple<rusage::divide<Types...>> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
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
    inline data_accum_t&       accum() { return m_accum; }
    inline const data_accum_t& accum() const { return m_accum; }
    inline intmax_t            laps() const { return m_laps; }

protected:
    // protected member functions
    data_accum_t&       get_accum() { return m_accum; }
    const data_accum_t& get_accum() const { return m_accum; }

protected:
    // objects
    mutex_t              m_mutex;
    mutable data_t       m_data;
    mutable data_accum_t m_accum;
    intmax_t             m_laps;
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
