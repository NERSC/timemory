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
#include "timemory/component_operations.hpp"
#include "timemory/components.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"
#include "timemory/serializer.hpp"

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
    : m_identifier("")
    , m_laps(0)
    {
    }

    component_tuple(const string_t& key, const string_t& tag, const int32_t& ncount,
                    const int32_t& nhash)
    : m_identifier("")
    , m_laps(0)
    {
        auto string_hash = [](const string_t& str) { return std::hash<string_t>()(str); };

        auto get_prefix = []() {
            if(!mpi_is_initialized())
                return string_t("> ");

            static string_t* _prefix = nullptr;
            if(!_prefix)
            {
                // prefix spacing
                static uint16_t width = 1;
                if(mpi_size() > 9)
                    width = std::max(width, (uint16_t)(log10(mpi_size()) + 1));
                std::stringstream ss;
                ss.fill('0');
                ss << "|" << std::setw(width) << mpi_rank() << "> ";
                _prefix = new string_t(ss.str());
            }
            return *_prefix;
        };

        uintmax_t ref =
            (string_hash(key) + string_hash(tag)) * (ncount + 2) * (nhash + 2);

        std::stringstream ss;

        // designated as [cxx], [pyc], etc.
        ss << get_prefix() << "[" << tag << "] ";
        // indent
        for(intmax_t i = 0; i < ncount; ++i)
        {
            if(i + 1 == ncount)
                ss << "|_";
            else
                ss << "  ";
        }
        ss << std::left << key;
        m_identifier = ss.str();
        output_width(m_identifier.length());
    }
    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_tuple(const component_tuple& rhs)
    : m_identifier(rhs.m_identifier)
    , m_data(rhs.m_data)
    , m_accum(rhs.m_accum)
    , m_laps(rhs.m_laps)
    {
    }
    component_tuple& operator=(const component_tuple& rhs)
    {
        if(this == &rhs)
            return *this;
        m_identifier = rhs.m_identifier;
        m_data       = rhs.m_data;
        m_accum      = rhs.m_accum;
        m_laps       = rhs.m_laps;
        return *this;
    }

    component_tuple(component_tuple&&) = default;
    component_tuple& operator=(component_tuple&&) = default;

public:
    //----------------------------------------------------------------------------------//
    // start/stop functions
    void start()
    {
        ++m_laps;
        using apply_types = std::tuple<component::start<Types>...>;
        apply<void>::access<apply_types>(m_data);
    }

    void stop()
    {
        using apply_types = std::tuple<component::stop<Types>...>;
        apply<void>::access<apply_types>(m_data);
    }

    //----------------------------------------------------------------------------------//
    // conditional start/stop functions
    void conditional_start()
    {
        auto increment    = [&](bool did_start) { ++m_laps; };
        using apply_types = std::tuple<component::conditional_start<Types>...>;
        apply<void>::access<apply_types>(m_data, increment);
    }

    void conditional_stop()
    {
        using apply_types = std::tuple<component::conditional_stop<Types>...>;
        apply<void>::access<apply_types>(m_data);
    }

    //----------------------------------------------------------------------------------//
    // recording
    //
    this_type& record()
    {
        ++m_laps;
        {
            using apply_types = std::tuple<component::record<Types>...>;
            apply<void>::access<apply_types>(m_data);
        }
        return *this;
    }

    this_type& record(const this_type& rhs)
    {
        if(this != &rhs)
            ++m_laps;
        auto c_data = std::move(rhs.m_data);
        {
            using apply_types = std::tuple<component::record<Types>...>;
            apply<void>::access<apply_types>(m_data);
        }
        {
            using apply_types = std::tuple<component::minus<Types>...>;
            apply<void>::access2<apply_types>(m_data, c_data);
        }
        {
            using apply_types = std::tuple<component::plus<Types>...>;
            apply<void>::access2<apply_types>(m_accum, m_data);
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//
    this_type record() const
    {
        this_type tmp(*this);
        return tmp.record();
    }

    this_type record(const this_type& rhs) const
    {
        this_type tmp(*this);
        return tmp.record(rhs);
    }

    //----------------------------------------------------------------------------------//
    void reset()
    {
        using apply_types = std::tuple<component::reset<Types>...>;
        apply<void>::access<apply_types>(m_data);
        m_laps = 0;
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        using apply_types = std::tuple<component::minus<Types>...>;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        using apply_types = std::tuple<component::plus<Types>...>;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename _Op>
    this_type& operator-=(_Op&& rhs)
    {
        using apply_types = std::tuple<component::minus<Types>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        using apply_types = std::tuple<component::plus<Types>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        using apply_types = std::tuple<component::multiply<Types>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        using apply_types = std::tuple<component::divide<Types>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    //----------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        {
            // stop, if not already stopped
            using apply_types = std::tuple<component::conditional_stop<Types>...>;
            apply<void>::access<apply_types>(obj.m_data);
        }
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        {
            using apply_types = std::tuple<component::print<Types>...>;
            apply<void>::access_with_indices<apply_types>(obj.m_data, std::ref(ss_data),
                                                          false);
        }
        ss_prefix << std::setw(output_width()) << std::left << obj.m_identifier << " : ";
        os << ss_prefix.str() << ss_data.str() << " [laps: " << obj.m_laps << "]";
        return os;
    }

    //----------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        using apply_types = std::tuple<component::serial<Types, Archive>...>;
        ar(serializer::make_nvp("identifier", m_identifier),
           serializer::make_nvp("laps", m_laps));
        ar.setNextName("data");
        ar.startNode();
        apply<void>::access<apply_types>(m_data, std::ref(ar), version);
        ar.finishNode();
        ar.setNextName("accum");
        ar.startNode();
        apply<void>::access<apply_types>(m_accum, std::ref(ar), version);
        ar.finishNode();
    }

    //----------------------------------------------------------------------------------//
    inline void report(std::ostream& os, bool endline, bool ign_cutoff) const
    {
        std::stringstream ss;
        ss << *this;

        if(endline)
            ss << std::endl;

        // ensure thread-safety
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        // output to ostream
        os << ss.str();
    }

public:
    inline data_t&       accum() { return m_accum; }
    inline const data_t& accum() const { return m_accum; }
    inline intmax_t      laps() const { return m_laps; }

protected:
    // protected member functions
    data_t&       get_accum() { return m_accum; }
    const data_t& get_accum() const { return m_accum; }

protected:
    // objects
    mutex_t        m_mutex;
    string_t       m_identifier;
    mutable data_t m_data;
    mutable data_t m_accum;
    intmax_t       m_laps;

protected:
    static intmax_t output_width(intmax_t width = 0)
    {
        static std::atomic_intmax_t _instance;
        if(width > 0)
        {
            auto current_width = _instance.load(std::memory_order_relaxed);
            auto compute       = [&]() {
                current_width = _instance.load(std::memory_order_relaxed);
                return std::max(_instance.load(), width);
            };
            intmax_t propose_width = compute();
            do
            {
                if(propose_width > current_width)
                {
                    auto ret = _instance.compare_exchange_strong(
                        current_width, propose_width, std::memory_order_relaxed);
                    if(!ret)
                        compute();
                }
            } while(propose_width > current_width);
        }
        return _instance.load();
    }
};

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
class custom_component_tuple : public component_tuple<Types...>
{
public:
    custom_component_tuple(const string_t& key, const string_t& tag)
    : component_tuple<Types...>(key, tag, 0, 0)
    {
    }

    //----------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&                           os,
                                    const custom_component_tuple<Types...>& obj)
    {
        {
            // stop, if not already stopped
            using apply_types = std::tuple<component::conditional_stop<Types>...>;
            apply<void>::access<apply_types>(obj.m_data);
        }
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        {
            using apply_types = std::tuple<custom_print<Types>...>;
            apply<void>::access_with_indices<apply_types>(obj.m_data, std::ref(ss_data),
                                                          false);
        }
        ss_prefix << std::setw(obj.output_width()) << std::left << obj.m_identifier
                  << " : ";
        os << ss_prefix.str() << ss_data.str();
        return os;
    }

protected:
    //----------------------------------------------------------------------------------//
    template <typename _Tp>
    struct custom_print
    {
        using value_type = typename _Tp::value_type;
        using base_type  = tim::component::base<_Tp, value_type>;

        custom_print(std::size_t _N, std::size_t _Ntot, base_type& obj, std::ostream& os,
                     bool endline)
        {
            std::stringstream ss;
            if(_N == 0)
                ss << std::endl;
            ss << "    " << obj << std::endl;
            os << ss.str();
        }
    };
};

//--------------------------------------------------------------------------------------//

}  // namespace details

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
