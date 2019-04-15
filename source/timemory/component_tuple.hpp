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
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"
#include "timemory/serializer.hpp"
#include "timemory/storage.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//

template <typename... Types>
class component_tuple
{
    static const std::size_t num_elements = sizeof...(Types);

public:
    using size_type   = intmax_t;
    using this_type   = component_tuple<Types...>;
    using data_t      = std::tuple<Types...>;
    using string_hash = std::hash<string_t>;
    using bool_array  = std::array<bool, num_elements>;

public:
    explicit component_tuple()
    : m_store(false)
    , m_laps(0)
    , m_count(0)
    , m_hash(0)
    , m_identifier(get_prefix())
    {
        init_manager();
        push();
    }

    explicit component_tuple(const string_t& key, const bool& store,
                             const string_t& tag = "cxx", const int32_t& ncount = 0,
                             const int32_t& nhash = 0)
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash(nhash)
    , m_identifier("")
    {
        compute_identifier(key, tag);
        init_manager();
        push();
    }

    component_tuple(const string_t& key, const string_t& tag = "cxx",
                    const int32_t& ncount = 0, const int32_t& nhash = 0,
                    bool store = true)
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash(nhash)
    , m_identifier("")
    {
        compute_identifier(key, tag);
        init_manager();
        push();
    }

    ~component_tuple() { pop(); }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_tuple(const component_tuple& rhs)
    : m_store(rhs.m_store)
    , m_laps(rhs.m_laps)
    , m_count(rhs.m_count)
    , m_hash(rhs.m_hash)
    , m_data(rhs.m_data)
    , m_identifier(rhs.m_identifier)
    {
    }

    component_tuple& operator=(const component_tuple& rhs)
    {
        if(this == &rhs)
            return *this;
        m_store      = rhs.m_store;
        m_laps       = rhs.m_laps;
        m_count      = rhs.m_count;
        m_hash       = rhs.m_hash;
        m_data       = rhs.m_data;
        m_identifier = rhs.m_identifier;
        return *this;
    }

    component_tuple(component_tuple&&) = default;
    component_tuple& operator=(component_tuple&&) = default;

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return num_elements; }

    //----------------------------------------------------------------------------------//
    // insert into graph
    inline void push()
    {
        if(m_store)
        {
            {
                apply<void>::set_value(m_exists, false);
                using apply_types = std::tuple<component::insert_node<Types>...>;
                apply<void>::access_with_indices<apply_types>(m_data, m_exists.data(),
                                                              m_hash);
            }
            {
                using apply_types = std::tuple<component::set_prefix<Types>...>;
                apply<void>::access2<apply_types>(m_data, m_exists, m_identifier);
            }
        }
    }

    //----------------------------------------------------------------------------------//
    // pop out of grapsh
    inline void pop()
    {
        if(m_store)
        {
            using apply_types = std::tuple<component::pop_node<Types>...>;
            apply<void>::access<apply_types>(m_data);
        }
    }

    //----------------------------------------------------------------------------------//
    // measure functions
    void measure()
    {
        using apply_types = std::tuple<component::measure<Types>...>;
        apply<void>::access<apply_types>(m_data);
    }

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
        auto increment = [&](bool did_start) {
            if(did_start)
            {
                ++m_laps;
            }
        };
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

    this_type& operator-=(this_type& rhs)
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

    this_type& operator+=(this_type& rhs)
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
    // friend operators
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        return tmp += rhs;
    }

    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        return tmp -= rhs;
    }

    template <typename _Op>
    friend this_type operator*(const this_type& lhs, _Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp *= std::forward<_Op>(rhs);
    }

    template <typename _Op>
    friend this_type operator/(const this_type& lhs, _Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp /= std::forward<_Op>(rhs);
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
    }

    //----------------------------------------------------------------------------------//
    inline void report(std::ostream& os, bool endline, bool ign_cutoff) const
    {
        consume_parameters(std::move(ign_cutoff));
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
    inline data_t&       data() { return m_data; }
    inline const data_t& data() const { return m_data; }
    inline intmax_t      laps() const { return m_laps; }

protected:
    // protected member functions
    data_t&       get_data() { return m_data; }
    const data_t& get_data() const { return m_data; }

protected:
    // objects
    bool           m_store = false;
    mutex_t        m_mutex;
    intmax_t       m_laps  = 0;
    intmax_t       m_count = 0;
    intmax_t       m_hash  = 0;
    mutable data_t m_data;
    string_t       m_identifier = "";
    bool_array     m_exists;

protected:
    string_t get_prefix()
    {
        auto _get_prefix = []() {
            if(!mpi_is_initialized())
                return string_t("> ");

            // prefix spacing
            static uint16_t width = 1;
            if(mpi_size() > 9)
                width = std::max(width, (uint16_t)(log10(mpi_size()) + 1));
            std::stringstream ss;
            ss.fill('0');
            ss << "|" << std::setw(width) << mpi_rank() << "> ";
            return ss.str();
        };
        static string_t _prefix = _get_prefix();
        return _prefix;
    }

    void compute_identifier(const string_t& key, const string_t& tag)
    {
        static string_t   _prefix = get_prefix();
        std::stringstream ss;

        // designated as [cxx], [pyc], etc.
        ss << _prefix << "[" << tag << "] ";

        // indent
        for(intmax_t i = 0; i < m_count; ++i)
        {
            if(i + 1 == m_count)
                ss << "|_";
            else
                ss << "  ";
        }
        ss << std::left << key;
        m_identifier = ss.str();
        output_width(m_identifier.length());
    }

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

private:
    void init_manager();
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

        custom_print(std::size_t _N, std::size_t /*_Ntot*/, base_type& obj,
                     std::ostream& os, bool /*endline*/)
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

#if defined(TIMEMORY_USE_FILTERING)

//--------------------------------------------------------------------------------------//

template <typename...>
struct component_concat
{
};

template <>
struct component_concat<>
{
    using type = component_tuple<>;
};

template <typename... Ts>
struct component_concat<component_tuple<Ts...>>
{
    using type = component_tuple<Ts...>;
};

template <typename... Ts0, typename... Ts1, typename... Rest>
struct component_concat<component_tuple<Ts0...>, component_tuple<Ts1...>, Rest...>
: component_concat<component_tuple<Ts0..., Ts1...>, Rest...>
{
};

template <typename... Ts>
using component_concat_t = typename component_concat<Ts...>::type;

//--------------------------------------------------------------------------------------//

template <bool>
struct component_filter_if_result
{
    template <typename T>
    using type = component_tuple<T>;
};

template <>
struct component_filter_if_result<false>
{
    template <typename T>
    using type = component_tuple<>;
};

template <template <typename> class Predicate, typename Sequence>
struct component_filter_if;

template <template <typename> class Predicate, typename... Ts>
struct component_filter_if<Predicate, component_tuple<Ts...>>
{
    using type = component_concat_t<
        typename component_filter_if_result<Predicate<Ts>::value>::template type<Ts>...>;
};

//--------------------------------------------------------------------------------------//

#endif

//--------------------------------------------------------------------------------------//

}  // namespace details

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_FILTERING)
template <template <typename> class Predicate, typename Sequence>
using type_filter = typename details::component_filter_if<Predicate, Sequence>::type;
#endif

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#include "timemory/manager.hpp"

//======================================================================================//

template <typename... Types>
void
tim::component_tuple<Types...>::init_manager()
{
    tim::manager::instance();
}

//======================================================================================//
//
//      std::get operator
//
namespace std
{
//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(tim::component_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::component_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::component_tuple<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::component_tuple<Types...>>(obj).data()))
{
    using obj_type = tim::component_tuple<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}

//--------------------------------------------------------------------------------------//
