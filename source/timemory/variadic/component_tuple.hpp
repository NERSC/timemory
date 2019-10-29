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
#include "timemory/bits/settings.hpp"
#include "timemory/components.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/storage.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//
// variadic list of components
//
template <typename... Types>
class component_tuple
{
    // empty init for friends
    explicit component_tuple() {}

    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

    template <typename... _Types>
    friend class auto_tuple;

public:
    template <typename... _Types>
    struct filtered
    {
    };

    template <typename... _Types>
    struct filtered<std::tuple<_Types...>>
    {
        using this_type  = component_tuple<_Types...>;
        using data_type  = std::tuple<_Types...>;
        using type_tuple = std::tuple<_Types...>;

        template <typename _Archive>
        using serialize_t = std::tuple<operation::serialization<Types, _Archive>...>;
        template <typename _Scope>
        using insert_node_t = std::tuple<operation::insert_node<_Types, _Scope>...>;
        using pop_node_t    = std::tuple<operation::pop_node<_Types>...>;
        using measure_t     = std::tuple<operation::measure<_Types>...>;
        using record_t      = std::tuple<operation::record<_Types>...>;
        using reset_t       = std::tuple<operation::reset<_Types>...>;
        using plus_t        = std::tuple<operation::plus<_Types>...>;
        using minus_t       = std::tuple<operation::minus<_Types>...>;
        using multiply_t    = std::tuple<operation::multiply<_Types>...>;
        using divide_t      = std::tuple<operation::divide<_Types>...>;
        using print_t       = std::tuple<operation::print<_Types>...>;
        using prior_start_t = std::tuple<operation::priority_start<_Types>...>;
        using prior_stop_t  = std::tuple<operation::priority_stop<_Types>...>;
        using stand_start_t = std::tuple<operation::standard_start<_Types>...>;
        using stand_stop_t  = std::tuple<operation::standard_stop<_Types>...>;
        using mark_begin_t  = std::tuple<operation::mark_begin<_Types>...>;
        using mark_end_t    = std::tuple<operation::mark_end<_Types>...>;
        using customize_t   = std::tuple<operation::customize<_Types>...>;
        using set_prefix_t  = std::tuple<operation::set_prefix<_Types>...>;
        using get_data_t    = std::tuple<operation::get_data<_Types>...>;
        using auto_type     = auto_tuple<_Types...>;
    };

public:
    using string_t    = std::string;
    using size_type   = int64_t;
    using string_hash = std::hash<string_t>;
    using this_type   = component_tuple<Types...>;
    using data_type   = typename filtered<available_tuple<concat<Types...>>>::data_type;
    using type_tuple  = typename filtered<available_tuple<concat<Types...>>>::type_tuple;
    using data_value_type = get_data_value_t<data_type>;
    using data_label_type = get_data_label_t<data_type>;

    // used by gotcha
    using component_type = this_type;
    using auto_type      = auto_tuple<Types...>;

    // used by component hybrid
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = true;
    static constexpr bool is_component_hybrid = false;

    // used by gotcha component to prevent recursion
    static constexpr bool contains_gotcha =
        (std::tuple_size<filter_gotchas<Types...>>::value != 0);

public:
    // modifier types
    // clang-format off
    template <typename _Archive>
    using serialize_t   = typename filtered<available_tuple<concat<Types...>>>::template serialize_t<_Archive>;
    template <typename _Scope>
    using insert_node_t = typename filtered<available_tuple<concat<Types...>>>::template insert_node_t<_Scope>;
    using pop_node_t    = typename filtered<available_tuple<concat<Types...>>>::pop_node_t;
    using measure_t     = typename filtered<available_tuple<concat<Types...>>>::measure_t;
    using record_t      = typename filtered<available_tuple<concat<Types...>>>::record_t;
    using reset_t       = typename filtered<available_tuple<concat<Types...>>>::reset_t;
    using plus_t        = typename filtered<available_tuple<concat<Types...>>>::plus_t;
    using minus_t       = typename filtered<available_tuple<concat<Types...>>>::minus_t;
    using multiply_t    = typename filtered<available_tuple<concat<Types...>>>::multiply_t;
    using divide_t      = typename filtered<available_tuple<concat<Types...>>>::divide_t;
    using print_t       = typename filtered<available_tuple<concat<Types...>>>::print_t;
    using prior_start_t = typename filtered<available_tuple<concat<Types...>>>::prior_start_t;
    using prior_stop_t  = typename filtered<available_tuple<concat<Types...>>>::prior_stop_t;
    using stand_start_t = typename filtered<available_tuple<concat<Types...>>>::stand_start_t;
    using stand_stop_t  = typename filtered<available_tuple<concat<Types...>>>::stand_stop_t;
    using mark_begin_t  = typename filtered<available_tuple<concat<Types...>>>::mark_begin_t;
    using mark_end_t    = typename filtered<available_tuple<concat<Types...>>>::mark_end_t;
    using customize_t   = typename filtered<available_tuple<concat<Types...>>>::customize_t;
    using set_prefix_t  = typename filtered<available_tuple<concat<Types...>>>::set_prefix_t;
    using get_data_t    = typename filtered<available_tuple<concat<Types...>>>::get_data_t;
    // clang-format on

public:
    template <typename _Scope = scope::process,
              bool _Flat      = std::is_same<_Scope, scope::flat>::value>
    explicit component_tuple(const string_t& key, const bool& store = false,
                             const bool& flat = (_Flat || settings::flat_profile()))
    : m_store(store && settings::enabled())
    , m_flat(flat)
    , m_laps(0)
    , m_hash((settings::enabled()) ? add_hash_id(key) : 0)
    , m_key(key)
    {
        // if(settings::enabled())
        {
            compute_width(key);
            // init_storage();
        }
    }

    ~component_tuple() { pop(); }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_tuple(const component_tuple&) = default;
    component_tuple(component_tuple&&)      = default;

    component_tuple& operator=(const component_tuple& rhs) = default;
    component_tuple& operator=(component_tuple&&) = default;

    template <typename _Scope = scope::process,
              bool _Flat      = std::is_same<_Scope, scope::flat>::value>
    component_tuple clone(bool store, bool flat = _Flat)
    {
        component_tuple tmp(*this);
        tmp.m_store = store;
        tmp.m_flat  = flat;
        return tmp;
    }

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return std::tuple_size<type_tuple>::value; }
    static constexpr std::size_t available_size()
    {
        return std::tuple_size<data_type>::value;
    }

    //----------------------------------------------------------------------------------//
    // insert into graph
    inline void push()
    {
        if(m_store && !m_is_pushed)
        {
            // reset the data
            apply<void>::access<reset_t>(m_data);
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = true;
            // insert node or find existing node
            if(m_flat)
                apply<void>::access<insert_node_t<scope::flat>>(m_data, m_hash);
            else
                apply<void>::access<insert_node_t<scope::process>>(m_data, m_hash);
        }
    }

    //----------------------------------------------------------------------------------//
    // pop out of graph
    inline void pop()
    {
        if(m_store && m_is_pushed)
        {
            // set the current node to the parent node
            apply<void>::access<pop_node_t>(m_data);
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = false;
        }
    }

    //----------------------------------------------------------------------------------//
    // measure functions
    void measure() { apply<void>::access<measure_t>(m_data); }

    //----------------------------------------------------------------------------------//
    // start/stop functions
    void start()
    {
        push();
        // increment laps
        ++m_laps;
        // start components
        apply<void>::access<prior_start_t>(m_data);
        apply<void>::access<stand_start_t>(m_data);
    }

    void stop()
    {
        // stop components
        apply<void>::access<prior_stop_t>(m_data);
        apply<void>::access<stand_stop_t>(m_data);
        // pop them off the running stack
        pop();
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... _Args>
    void mark_begin(_Args&&... _args)
    {
        apply<void>::access<mark_begin_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... _Args>
    void mark_end(_Args&&... _args)
    {
        apply<void>::access<mark_end_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform a customized operation (typically for GOTCHA)
    //
    template <typename... _Args>
    void customize(_Args&&... _args)
    {
        apply<void>::access<customize_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // recording
    //
    this_type& record()
    {
        ++m_laps;
        apply<void>::access<record_t>(m_data);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // reset data
    //
    void reset()
    {
        apply<void>::access<reset_t>(m_data);
        m_laps = 0;
    }

    //----------------------------------------------------------------------------------//
    // get data
    //
    data_value_type get() const
    {
        const_cast<this_type&>(*this).stop();
        data_value_type _ret_data;
        apply<void>::access2<get_data_t>(m_data, _ret_data);
        return _ret_data;
    }

    //----------------------------------------------------------------------------------//
    // reset data
    //
    data_label_type get_labeled() const
    {
        const_cast<this_type&>(*this).stop();
        data_label_type _ret_data;
        apply<void>::access2<get_data_t>(m_data, _ret_data);
        return _ret_data;
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        apply<void>::access2<minus_t>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        apply<void>::access2<minus_t>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        apply<void>::access2<plus_t>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        apply<void>::access2<plus_t>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename _Op>
    this_type& operator-=(_Op&& rhs)
    {
        apply<void>::access<minus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        apply<void>::access<plus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        apply<void>::access<multiply_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        apply<void>::access<divide_t>(m_data, std::forward<_Op>(rhs));
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
        if(available_size() == 0)
            return os;
        // stop, if not already stopped
        apply<void>::access<prior_stop_t>(obj.m_data);
        apply<void>::access<stand_stop_t>(obj.m_data);
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        apply<void>::access_with_indices<print_t>(obj.m_data, std::ref(ss_data), false);
        if(obj.m_print_prefix)
        {
            obj.update_width();
            std::stringstream ss_id;
            ss_id << obj.get_prefix() << " " << std::left << obj.m_key;
            ss_prefix << std::setw(output_width()) << std::left << ss_id.str() << " : ";
            os << ss_prefix.str();
        }
        os << ss_data.str();
        if(obj.m_laps > 0 && obj.m_print_laps)
            os << " [laps: " << obj.m_laps << "]";
        return os;
    }

    //----------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar(serializer::make_nvp("key", m_key), serializer::make_nvp("laps", m_laps));
        ar.setNextName("data");
        ar.startNode();
        apply<void>::access<serialize_t<Archive>>(m_data, std::ref(ar), version);
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

    const int64_t&  hash() const { return m_hash; }
    const string_t& key() const { return m_key; }
    void            rekey(const string_t& _key) { compute_width(m_key = _key); }

    bool&       store() { return m_store; }
    const bool& store() const { return m_store; }

public:
    // get member functions taking either a type
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

    //----------------------------------------------------------------------------------//
    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, data_type>::value == true), int> = 0>
    void type_apply(_Func&& _func, _Args&&... _args)
    {
        auto&& _obj = get<_Tp>();
        ((_obj).*(_func))(std::forward<_Args>(_args)...);
    }

    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, data_type>::value == false), int> = 0>
    void type_apply(_Func&&, _Args&&...)
    {
    }

protected:
    // protected member functions
    data_type&       get_data() { return m_data; }
    const data_type& get_data() const { return m_data; }

protected:
    // objects
    bool              m_store        = false;
    bool              m_flat         = false;
    bool              m_is_pushed    = false;
    bool              m_print_prefix = true;
    bool              m_print_laps   = true;
    int64_t           m_laps         = 0;
    int64_t           m_hash         = 0;
    string_t          m_key          = "";
    mutable data_type m_data;

protected:
    string_t get_prefix() const
    {
        auto _get_prefix = []() {
            if(!mpi::is_initialized())
                return string_t(">>> ");

            // prefix spacing
            static uint16_t width = 1;
            if(mpi::size() > 9)
                width = std::max(width, (uint16_t)(log10(mpi::size()) + 1));
            std::stringstream ss;
            ss.fill('0');
            ss << "|" << std::setw(width) << mpi::rank() << ">>> ";
            return ss.str();
        };
        static string_t _prefix = _get_prefix();
        return _prefix;
    }

    void compute_width(const string_t& key)
    {
        static string_t _prefix = get_prefix();
        output_width(key.length() + _prefix.length() + 1);
        set_object_prefix(key);
    }

    void update_width() const { const_cast<this_type&>(*this).compute_width(m_key); }

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

    void set_object_prefix(const string_t& key)
    {
        apply<void>::access<set_prefix_t>(m_data, key);
    }

public:
    static void init_storage()
    {
        apply<void>::type_access<operation::init_storage, data_type>();
    }
};

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/variadic/bits/component_tuple.hpp"
