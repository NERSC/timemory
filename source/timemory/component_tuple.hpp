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

/** \file component_tuple.hpp
 * \headerfile component_tuple.hpp "timemory/component_tuple.hpp"
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
#include "timemory/components.hpp"
#include "timemory/data_types.hpp"
#include "timemory/macros.hpp"
#include "timemory/serializer.hpp"
#include "timemory/string.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//

template <typename... Types>
class component_tuple
{
public:
    using size_type = intmax_t;
    using this_type = component_tuple<Types...>;
    using data_t    = std::tuple<Types...>;

public:
    explicit component_tuple()
    : m_laps(0)
    {
    }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_tuple(const component_tuple& rhs)
    : m_data(rhs.m_data)
    , m_accum(rhs.m_accum)
    , m_laps(rhs.m_laps)
    {
    }
    component_tuple& operator=(const component_tuple& rhs)
    {
        if(this == &rhs)
            return *this;
        m_data  = rhs.m_data;
        m_accum = rhs.m_accum;
        m_laps  = rhs.m_laps;
        return *this;
    }

    component_tuple(component_tuple&&) = default;
    component_tuple& operator=(component_tuple&&) = default;

public:
    //----------------------------------------------------------------------------------//
    this_type& record()
    {
        typedef std::tuple<timing::record<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data);
        ++m_laps;
        return *this;
    }

    this_type& record(const this_type& rhs)
    {
        if(this != &rhs)
            ++m_laps;
        auto c_data = std::move(rhs.m_data);
        {
            typedef std::tuple<timing::record<Types>...> apply_types;
            apply<void>::access<apply_types>(m_data);
        }
        {
            typedef std::tuple<timing::minus<Types>...> apply_types;
            apply<void>::access2<apply_types>(m_data, c_data);
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//
    this_type record() const
    {
        this_type                                    tmp(*this);
        typedef std::tuple<timing::record<Types>...> apply_types;
        apply<void>::access<apply_types>(tmp.m_data);
        tmp.m_laps += 1;
        return tmp;
    }

    this_type record(const this_type& rhs) const
    {
        this_type tmp(*this);
        {
            typedef std::tuple<timing::record<Types>...> apply_types;
            apply<void>::access<apply_types>(tmp.m_data);
        }
        {
            typedef std::tuple<timing::minus<Types>...> apply_types;
            apply<void>::access2<apply_types>(tmp.m_data, rhs.m_data);
        }
        tmp.m_laps += 1;
        return tmp;
    }

    //----------------------------------------------------------------------------------//
    void reset()
    {
        typedef std::tuple<timing::reset<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data);
        m_laps = 0;
    }

    //----------------------------------------------------------------------------------//
    // operators
    this_type& operator-=(const this_type& rhs)
    {
        typedef std::tuple<timing::minus<Types>...> apply_types;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator-=(uintmax_t&& rhs)
    {
        typedef std::tuple<timing::minus<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<uintmax_t>(rhs));
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        typedef std::tuple<timing::plus<Types>...> apply_types;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    this_type& operator+=(uintmax_t&& rhs)
    {
        typedef std::tuple<timing::plus<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<uintmax_t>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        typedef std::tuple<timing::multiply<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        typedef std::tuple<timing::divide<Types>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    //--------------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const this_type& _timing)
    {
        typedef std::tuple<timing::print<Types>...> apply_types;
        data_t                                      _data = _timing.m_data;
        apply<void>::access<apply_types>(_timing.m_data, std::ref(os));
        return os;
    }

    //--------------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        typedef std::tuple<timing::serial<Types, Archive>...> apply_types;
        apply<void>::access<apply_types>(m_data, std::ref(ar), version);
        ar(serializer::make_nvp("laps", m_laps));
    }

    //--------------------------------------------------------------------------------------//
    inline void report(std::ostream& os, bool endline, bool ign_cutoff) const
    {
        // stop, if not already stopped
        // if(m_data.running())
        //    const_cast<component_tuple*>(this)->stop();

        std::stringstream ss;
        ss << *this;
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
    mutex_t        m_mutex;
    mutable data_t m_data;
    mutable data_t m_accum;
    intmax_t       m_laps;
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
