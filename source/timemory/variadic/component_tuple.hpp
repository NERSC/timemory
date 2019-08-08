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
 * \headerfile component_tuple.hpp "timemory/variadic/component_tuple.hpp"
 * This is the C++ class that bundles together components and enables
 * operation on the components as a single entity
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "timemory/backends/mpi.hpp"
#include "timemory/components.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/storage.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//
// forward declaration
//
template <typename... Types>
class auto_tuple;

//======================================================================================//
// variadic list of components
//
template <typename... Types>
class component_tuple
{
    static const std::size_t num_elements = sizeof...(Types);
    // empty init for friends
    explicit component_tuple() {}
    // manager is friend so can use above
    friend class manager;

public:
    using size_type   = int64_t;
    using this_type   = component_tuple<Types...>;
    using data_type   = tim::implemented_tuple<Types...>;
    using string_hash = std::hash<string_t>;
    using language_t  = tim::language;

public:
    // operation types
    using op_insert_node_t = tim::operation_tuple<operation::insert_node, Types...>;
    using op_pop_node_t    = tim::operation_tuple<operation::pop_node, Types...>;
    using op_measure_t     = tim::operation_tuple<operation::measure, Types...>;
    using op_start_t       = tim::operation_tuple<operation::start, Types...>;
    using op_stop_t        = tim::operation_tuple<operation::stop, Types...>;
    using op_cond_start_t  = tim::operation_tuple<operation::conditional_start, Types...>;
    using op_cond_stop_t   = tim::operation_tuple<operation::conditional_stop, Types...>;
    using op_record_t      = tim::operation_tuple<operation::record, Types...>;
    using op_reset_t       = tim::operation_tuple<operation::reset, Types...>;
    using op_plus_t        = tim::operation_tuple<operation::plus, Types...>;
    using op_minus_t       = tim::operation_tuple<operation::minus, Types...>;
    using op_multiply_t    = tim::operation_tuple<operation::multiply, Types...>;
    using op_divide_t      = tim::operation_tuple<operation::divide, Types...>;
    using op_print_t       = tim::operation_tuple<operation::print, Types...>;

public:
    using auto_type = auto_tuple<Types...>;

public:
    explicit component_tuple(const string_t& key, const bool& store,
                             const int64_t& ncount = 0, const int64_t& nhash = 0,
                             const language_t& lang = language_t::cxx())
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash((nhash == 0) ? string_hash()(key) : nhash)
    , m_lang(lang)
    , m_key(key)
    , m_identifier("")
    {
        compute_identifier(key, lang);
        init_manager();
        init_storage();
        push();
    }

    explicit component_tuple(const string_t& key, const bool& store,
                             const language_t& lang, const int64_t& ncount = 0,
                             const int64_t& nhash = 0)
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash((nhash == 0) ? string_hash()(key) : nhash)
    , m_lang(lang)
    , m_key(key)
    , m_identifier("")
    {
        compute_identifier(key, lang);
        init_manager();
        init_storage();
        push();
    }

    component_tuple(const string_t& key, const language_t& lang = language_t::cxx(),
                    const int64_t& ncount = 0, const int64_t& nhash = 0,
                    bool store = false)
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash((nhash == 0) ? string_hash()(key) : nhash)
    , m_lang(lang)
    , m_key(key)
    , m_identifier("")
    {
        compute_identifier(key, lang);
        init_manager();
        init_storage();
        push();
    }

    ~component_tuple() { pop(); }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_tuple(const component_tuple&) = default;
    component_tuple(component_tuple&&)      = default;

    component_tuple& operator=(const component_tuple& rhs) = default;
    component_tuple& operator=(component_tuple&&) = default;

    component_tuple clone(const int64_t& nhash, bool store)
    {
        component_tuple tmp(*this);
        tmp.m_hash  = nhash;
        tmp.m_store = store;
        return tmp;
    }

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return num_elements; }

    //----------------------------------------------------------------------------------//
    // insert into graph
    inline void push()
    {
        if(m_store && !m_is_pushed)
        {
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = true;
            // insert node or find existing node
            apply<void>::access<op_insert_node_t>(m_data, m_identifier, m_hash);
        }
    }

    //----------------------------------------------------------------------------------//
    // pop out of graph
    inline void pop()
    {
        if(m_store && m_is_pushed)
        {
            // set the current node to the parent node
            apply<void>::access<op_pop_node_t>(m_data);
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = false;
        }
    }

    //----------------------------------------------------------------------------------//
    // measure functions
    void measure() { apply<void>::access<op_measure_t>(m_data); }

    //----------------------------------------------------------------------------------//
    // start/stop functions
    void start()
    {
        // increment laps
        ++m_laps;
        // start components
        apply<void>::access<op_start_t>(m_data);
    }

    void stop()
    {
        // stop components
        apply<void>::access<op_stop_t>(m_data);
        // pop them off the running stack
        pop();
    }

    void conditional_start()
    {
        // start, if not already started
        apply<void>::access<op_cond_start_t>(m_data);
    }

    void conditional_stop()
    {
        // stop, if not already stopped
        apply<void>::access<op_cond_stop_t>(m_data);
        // pop them off the running stack
        pop();
    }

    //----------------------------------------------------------------------------------//
    // recording
    //
    this_type& record()
    {
        ++m_laps;
        apply<void>::access<op_record_t>(m_data);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    void reset()
    {
        apply<void>::access<op_reset_t>(m_data);
        m_laps = 0;
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        apply<void>::access2<op_minus_t>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        apply<void>::access2<op_minus_t>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        apply<void>::access2<op_plus_t>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        apply<void>::access2<op_plus_t>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename _Op>
    this_type& operator-=(_Op&& rhs)
    {
        apply<void>::access<op_minus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        apply<void>::access<op_plus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        apply<void>::access<op_multiply_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        apply<void>::access<op_divide_t>(m_data, std::forward<_Op>(rhs));
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
        // stop, if not already stopped
        apply<void>::access<op_cond_stop_t>(obj.m_data);
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        apply<void>::access_with_indices<op_print_t>(obj.m_data, std::ref(ss_data),
                                                     false);
        obj.update_identifier();
        ss_prefix << std::setw(output_width()) << std::left << obj.m_identifier << " : ";
        os << ss_prefix.str() << ss_data.str();
        if(obj.m_laps > 0)
            os << " [laps: " << obj.m_laps << "]";
        return os;
    }

    //----------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        using apply_types = std::tuple<operation::serialization<Types, Archive>...>;
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

    //----------------------------------------------------------------------------------//
    static void print_storage()
    {
        apply<void>::type_access<operation::print_storage, data_type>();
    }

public:
    inline data_type&       data() { return m_data; }
    inline const data_type& data() const { return m_data; }
    inline int64_t          laps() const { return m_laps; }

    int64_t&  hash() { return m_hash; }
    string_t& key() { return m_key; }
    string_t& identifier() { return m_identifier; }

    const int64_t&    hash() const { return m_hash; }
    const string_t&   key() const { return m_key; }
    const language_t& lang() const { return m_lang; }
    const string_t&   identifier() const { return m_identifier; }

    bool&       store() { return m_store; }
    const bool& store() const { return m_store; }

public:
    // get member functions taking either an integer or a type
    template <std::size_t _N>
    typename std::tuple_element<_N, data_type>::type& get()
    {
        return std::get<_N>(m_data);
    }

    template <std::size_t _N>
    const typename std::tuple_element<_N, data_type>::type& get() const
    {
        return std::get<_N>(m_data);
    }

    template <typename _Tp>
    _Tp& get()
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    template <typename _Tp>
    const _Tp& get() const
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

protected:
    // protected member functions
    data_type&       get_data() { return m_data; }
    const data_type& get_data() const { return m_data; }

protected:
    // objects
    bool              m_store      = false;
    bool              m_is_pushed  = false;
    int64_t           m_laps       = 0;
    int64_t           m_count      = 0;
    int64_t           m_hash       = 0;
    const language_t  m_lang       = language_t::cxx();
    string_t          m_key        = "";
    string_t          m_identifier = "";
    mutable data_type m_data;

protected:
    string_t get_prefix()
    {
        auto _get_prefix = []() {
            if(!mpi::is_initialized())
                return string_t("> ");

            // prefix spacing
            static uint16_t width = 1;
            if(mpi::size() > 9)
                width = std::max(width, (uint16_t)(log10(mpi::size()) + 1));
            std::stringstream ss;
            ss.fill('0');
            ss << "|" << std::setw(width) << mpi::rank() << "> ";
            return ss.str();
        };
        static string_t _prefix = _get_prefix();
        return _prefix;
    }

    void compute_identifier(const string_t& key, const language_t& lang)
    {
        static string_t   _prefix = get_prefix();
        std::stringstream ss;

        // designated as [cxx], [pyc], etc.
        ss << _prefix << lang << " ";

        // indent
        for(int64_t i = 0; i < m_count; ++i)
        {
            if(i + 1 == m_count)
                ss << "|_";
            else
                ss << "  ";
        }
        ss << std::left << key;
        m_identifier = ss.str();
        output_width(m_identifier.length());
        compute_identifier_extra<data_type>(key, lang);
    }

    void update_identifier() const
    {
        const_cast<this_type&>(*this).compute_identifier(m_key, m_lang);
    }

    static int64_t output_width(int64_t width = 0)
    {
        static std::atomic<int64_t> _instance;
        if(width > 0)
        {
            auto current_width = _instance.load(std::memory_order_relaxed);
            auto compute       = [&]() {
                current_width = _instance.load(std::memory_order_relaxed);
                return std::max(_instance.load(), width);
            };
            int64_t propose_width = compute();
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

    template <
        typename _Tuple = data_type,
        tim::enable_if_t<(is_one_of<component::caliper, _Tuple>::value == true), int> = 0>
    void compute_identifier_extra(const string_t& key, const language_t& lang)
    {
        constexpr auto idx           = index_of<component::caliper, _Tuple>::value;
        std::get<idx>(m_data).prefix = key;
        consume_parameters(lang);
    }

    template <typename _Tuple       = data_type,
              tim::enable_if_t<(is_one_of<component::caliper, _Tuple>::value == false),
                               int> = 0>
    void compute_identifier_extra(const string_t&, const language_t&)
    {
    }

private:
    void init_manager();
    void init_storage()
    {
        apply<void>::type_access<operation::init_storage, data_type>();
    }
};

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/details/component_tuple.hpp"
