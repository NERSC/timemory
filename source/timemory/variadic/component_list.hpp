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

/** \file component_list.hpp
 * \headerfile component_list.hpp "timemory/variadic/component_list.hpp"
 * This is similar to component_tuple but not as optimized.
 * This exists so that Python and C, which do not support templates,
 * can implement a subset of the tools
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

#include "timemory/backends/mpi.hpp"
#include "timemory/components.hpp"
#include "timemory/details/settings.hpp"
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
class auto_list;

template <typename _CompTuple, typename _CompList>
class component_hybrid;

template <typename... _Types>
class component_list;

//======================================================================================//
// variadic list of components
//
namespace filt
{
template <typename... Types>
class component_list
{
    static const std::size_t num_elements = sizeof...(Types);

    // empty init for friends
    explicit component_list() {}

    // manager is friend so can use above
    friend class ::tim::manager;

    template <typename _TupleC, typename _ListC>
    friend class ::tim::component_hybrid;

    template <typename... _Types>
    friend class ::tim::component_list;

public:
    using string_t        = std::string;
    using size_type       = int64_t;
    using this_type       = component_list<Types...>;
    using data_type       = std::tuple<Types*...>;
    using reference_type  = std::tuple<Types...>;
    using type_tuple      = std::tuple<Types...>;
    using string_hash     = std::hash<string_t>;
    using init_func_t     = std::function<void(this_type&)>;
    using data_value_type = get_data_value_t<reference_type>;
    using data_label_type = get_data_label_t<reference_type>;

    // used by component hybrid
    static constexpr bool is_component_list  = true;
    static constexpr bool is_component_tuple = false;

    // used by gotcha component to prevent recursion
    static constexpr bool contains_gotcha =
        (std::tuple_size<filter_gotchas<Types...>>::value != 0);

public:
    // modifier types
    // clang-format off
    template <typename _Archive>
    using serialize_t        = std::tuple<operation::pointer_operator<Types, operation::serialization<Types, _Archive>>...>;
    template <typename _Scope>
    using insert_node_t      = std::tuple<operation::pointer_operator<Types, operation::insert_node<Types, _Scope>>...>;
    using pop_node_t         = std::tuple<operation::pointer_operator<Types, operation::pop_node<Types>>...>;
    using measure_t          = std::tuple<operation::pointer_operator<Types, operation::measure<Types>>...>;
    using record_t           = std::tuple<operation::pointer_operator<Types, operation::record<Types>>...>;
    using reset_t            = std::tuple<operation::pointer_operator<Types, operation::reset<Types>>...>;
    using plus_t             = std::tuple<operation::pointer_operator<Types, operation::plus<Types>>...>;
    using minus_t            = std::tuple<operation::pointer_operator<Types, operation::minus<Types>>...>;
    using multiply_t         = std::tuple<operation::pointer_operator<Types, operation::multiply<Types>>...>;
    using divide_t           = std::tuple<operation::pointer_operator<Types, operation::divide<Types>>...>;
    using prior_start_t      = std::tuple<operation::pointer_operator<Types, operation::priority_start<Types>>...>;
    using prior_stop_t       = std::tuple<operation::pointer_operator<Types, operation::priority_stop<Types>>...>;
    using stand_start_t      = std::tuple<operation::pointer_operator<Types, operation::standard_start<Types>>...>;
    using stand_stop_t       = std::tuple<operation::pointer_operator<Types, operation::standard_stop<Types>>...>;
    using mark_begin_t       = std::tuple<operation::pointer_operator<Types, operation::mark_begin<Types>>...>;
    using mark_end_t         = std::tuple<operation::pointer_operator<Types, operation::mark_end<Types>>...>;
    using customize_t        = std::tuple<operation::pointer_operator<Types, operation::customize<Types>>...>;
    using set_prefix_extra_t = std::tuple<operation::pointer_operator<Types, operation::set_prefix<Types>>...>;
    using get_data_t         = std::tuple<operation::pointer_operator<Types, operation::get_data<Types>>...>;
    using print_t            = std::tuple<operation::print<Types>...>;
    using pointer_count_t    = std::tuple<operation::pointer_counter<Types>...>;
    using deleter_t          = std::tuple<operation::pointer_deleter<Types>...>;
    using copy_t             = std::tuple<operation::copy<Types>...>;
    // clang-format on

public:
    using auto_type = auto_list<Types...>;

public:
    template <typename _Func>
    explicit component_list(const string_t& key, const bool& store, const bool& flat,
                            _Func&& _func)
    : m_store(store && settings::enabled())
    , m_flat(flat)
    , m_laps(0)
    , m_key(key)
    {
        apply<void>::set_value(m_data, nullptr);
        // if(settings::enabled())
        {
            compute_width(key);
            init_manager();
            init_storage();
            _func(*this);
        }
    }

    ~component_list()
    {
        pop();
        apply<void>::access<deleter_t>(m_data);
    }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_list(component_list&&) = default;
    component_list& operator=(component_list&&) = default;

    component_list(const component_list& rhs)
    : m_store(rhs.m_store)
    , m_flat(rhs.m_flat)
    , m_is_pushed(rhs.m_is_pushed)
    , m_laps(rhs.m_laps)
    , m_key(rhs.m_key)
    {
        apply<void>::set_value(m_data, nullptr);
        apply<void>::access2<copy_t>(m_data, rhs.m_data);
    }

    component_list& operator=(const component_list& rhs)
    {
        if(this != &rhs)
        {
            m_store     = rhs.m_store;
            m_flat      = rhs.m_flat;
            m_is_pushed = rhs.m_is_pushed;
            m_laps      = rhs.m_laps;
            m_key       = rhs.m_key;
            apply<void>::access<deleter_t>(m_data);
            apply<void>::access2<copy_t>(m_data, rhs.m_data);
        }
        return *this;
    }

    component_list clone(bool store, bool flat)
    {
        component_list tmp(*this);
        tmp.m_store = store;
        tmp.m_flat  = flat;
        return tmp;
    }

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return num_elements; }
    static constexpr std::size_t available_size()
    {
        return std::tuple_size<type_tuple>::value;
    }

    //----------------------------------------------------------------------------------//
    // insert into graph
    inline void push()
    {
        uint64_t count = 0;
        apply<void>::access<pointer_count_t>(m_data, std::ref(count));
        if(m_store && !m_is_pushed && count > 0)
        {
            apply<void>::access<reset_t>(m_data);
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = true;
            // compute the hash
            int64_t _hash = add_hash_id(m_key);
            // insert node or find existing node
            if(m_flat)
                apply<void>::access<insert_node_t<scope::flat>>(m_data, _hash);
            else
                apply<void>::access<insert_node_t<scope::process>>(m_data, _hash);
        }
    }

    //----------------------------------------------------------------------------------//
    // pop out of grapsh
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
    // reset to zero
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
        uint64_t count = 0;
        apply<void>::access<pointer_count_t>(obj.m_data, std::ref(count));
        if(count < 1)
            return os;
        // stop, if not already stopped
        apply<void>::access<prior_stop_t>(obj.m_data);
        apply<void>::access<stand_stop_t>(obj.m_data);
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        apply<void>::access_with_indices<print_t>(obj.m_data, std::ref(ss_data), false);
        if(ss_data.str().length() > 0)
        {
            if(obj.m_print_prefix)
            {
                obj.update_width();
                std::stringstream ss_id;
                ss_id << obj.get_prefix() << " " << std::left << obj.m_key;
                ss_prefix << std::setw(output_width()) << std::left << ss_id.str()
                          << " : ";
                os << ss_prefix.str();
            }
            os << ss_data.str();
            if(obj.laps() > 0 && obj.m_print_laps)
                os << " [laps: " << obj.m_laps << "]";
        }
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
        apply<void>::type_access<operation::print_storage, reference_type>();
    }

public:
    inline data_type&       data() { return m_data; }
    inline const data_type& data() const { return m_data; }
    inline int64_t          laps() const { return m_laps; }

    string_t& key() { return m_key; }

    const string_t& key() const { return m_key; }
    void            rekey(const string_t& _key) { compute_width(m_key = _key); }

    bool&       store() { return m_store; }
    const bool& store() const { return m_store; }

public:
    // get member functions taking a type
    template <typename _Tp, enable_if_t<std::is_pointer<_Tp>::value, char> = 0>
    _Tp& get()
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    template <typename _Tp, enable_if_t<(!std::is_pointer<_Tp>::value), char> = 0>
    _Tp*& get()
    {
        return std::get<index_of<_Tp*, data_type>::value>(m_data);
    }

    template <typename _Tp, enable_if_t<std::is_pointer<_Tp>::value, char> = 0>
    const _Tp& get() const
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    template <typename _Tp, enable_if_t<(!std::is_pointer<_Tp>::value), char> = 0>
    const _Tp* get() const
    {
        return std::get<index_of<_Tp*, data_type>::value>(m_data);
    }

    //----------------------------------------------------------------------------------//
    ///  initialize a type that is in variadic list AND is available
    ///
    template <typename _Tp, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == true), int> = 0,
              enable_if_t<(trait::is_available<_Tp>::value == true), int>       = 0>
    void init(_Args&&... _args)
    {
        auto&& _obj = get<_Tp>();
        if(!_obj)
        {
            if(settings::debug())
            {
                std::string _id = demangle(typeid(_Tp).name());
                printf("[component_list::init]> initializing type '%s'...\n",
                       _id.c_str());
            }
            _obj = new _Tp(std::forward<_Args>(_args)...);
            set_object_prefix(_obj);
        }
        else
        {
            static std::atomic<int> _count;
            if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
            {
                std::string _id = demangle(typeid(_Tp).name());
                printf(
                    "[component_list::init]> skipping re-initialization of type"
                    " \"%s\"...\n",
                    _id.c_str());
            }
        }
    }

    //----------------------------------------------------------------------------------//
    ///  "initialize" a type that is in variadic list BUT is NOT available
    ///
    template <typename _Tp, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == true), int> = 0,
              enable_if_t<(trait::is_available<_Tp>::value == false), int>      = 0>
    void init(_Args&&...)
    {
        static std::atomic<int> _count;
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            std::string _id = demangle(typeid(_Tp).name());
            printf("[component_list::init]> skipping unavailable type '%s'...\n",
                   _id.c_str());
        }
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tp, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == false), int> = 0>
    void init(_Args&&...)
    {
    }

    //----------------------------------------------------------------------------------//
    //  variadic initialization
    //
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void initialize()
    {
        this->init<_Tp>();
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void initialize()
    {
        this->init<_Tp>();
        this->initialize<_Tail...>();
    }

    //----------------------------------------------------------------------------------//
    /// apply a member function to a type that is in variadic list AND is available
    ///
    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == true), int> = 0,
              enable_if_t<(trait::is_available<_Tp>::value == true), int>       = 0>
    void type_apply(_Func&& _func, _Args&&... _args)
    {
        auto&& _obj = get<_Tp>();
        if(_obj != nullptr)
            ((*_obj).*(_func))(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// "apply" a member function to a type that is in variadic list BUT is NOT available
    ///
    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == true), int> = 0,
              enable_if_t<(trait::is_available<_Tp>::value == false), int>      = 0>
    void type_apply(_Func&&, _Args&&...)
    {
    }

    //----------------------------------------------------------------------------------//
    /// invoked when a request to apply a member function to a type not in variadic list
    ///
    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == false), int> = 0>
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
    string_t          m_key          = "";
    mutable data_type m_data;

protected:
    string_t get_prefix() const
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
        apply<void>::access<set_prefix_extra_t>(m_data, key);
    }

    template <typename _Tp, enable_if_t<(trait::requires_prefix<_Tp>::value), int> = 0>
    void set_object_prefix(_Tp* obj)
    {
        if(obj)
            obj->prefix = m_key;
    }

    template <typename _Tp,
              enable_if_t<(trait::requires_prefix<_Tp>::value == false), int> = 0>
    void set_object_prefix(_Tp*)
    {
    }

public:
    static void init_manager();
    static void init_storage()
    {
        apply<void>::type_access<operation::init_storage, reference_type>();
    }
};

//--------------------------------------------------------------------------------------//
//  unused base class
//
template <typename... _Types>
struct comp_list_t
{
    using type = component_list<_Types...>;
};

//--------------------------------------------------------------------------------------//
//  tuple overloaded base class
//
template <typename... _Types>
struct comp_list_t<std::tuple<_Types...>>
{
    using type = component_list<_Types...>;
};

}  // namespace filt

//======================================================================================//
//
template <typename... _Types>
class component_list : public filt::comp_list_t<implemented<_Types...>>::type
{
    // empty init for friends
    explicit component_list() = default;

    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

public:
    using string_t       = std::string;
    using this_type      = component_list<_Types...>;
    using init_func_t    = std::function<void(this_type&)>;
    using base_type      = typename filt::comp_list_t<implemented<_Types...>>::type;
    using size_type      = typename base_type::size_type;
    using data_type      = typename base_type::data_type;
    using reference_type = typename base_type::reference_type;
    using type_tuple     = typename base_type::type_tuple;
    using auto_type      = typename base_type::auto_type;

    static constexpr bool is_component_list  = base_type::is_component_list;
    static constexpr bool is_component_tuple = base_type::is_component_tuple;
    static constexpr bool contains_gotcha    = base_type::contains_gotcha;

    template <typename _Scope = scope::process>
    explicit component_list(const string_t& key, const bool& store = false,
                            const bool& flat = (settings::flat_profile() ||
                                                std::is_same<_Scope, scope::flat>::value))
    : base_type(key, store, flat, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    ~component_list() {}

    component_list(const component_list&) = default;
    component_list(component_list&&)      = default;

    component_list& operator=(const component_list&) = default;
    component_list& operator=(component_list&&) = default;

    component_list(const base_type& rhs)
    : base_type(rhs)
    {
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        base_type::operator-=(static_cast<const base_type&>(rhs));
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        base_type::operator-=(static_cast<base_type&>(rhs));
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        base_type::operator+=(static_cast<const base_type&>(rhs));
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        base_type::operator+=(static_cast<base_type&>(rhs));
        return *this;
    }

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_COMPONENT_LIST_INIT", "");
        };
        return _instance;
    }
};

//======================================================================================//
//
template <typename... _Types>
class component_list<std::tuple<_Types...>>
: public filt::comp_list_t<implemented<_Types...>>::type
{
    // empty init for friends
    explicit component_list() = default;

    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

public:
    using string_t       = std::string;
    using this_type      = component_list<std::tuple<_Types...>>;
    using init_func_t    = std::function<void(this_type&)>;
    using base_type      = typename filt::comp_list_t<implemented<_Types...>>::type;
    using size_type      = typename base_type::size_type;
    using data_type      = typename base_type::data_type;
    using reference_type = typename base_type::reference_type;
    using type_tuple     = typename base_type::type_tuple;
    using auto_type      = typename base_type::auto_type;

    static constexpr bool is_component_list  = base_type::is_component_list;
    static constexpr bool is_component_tuple = base_type::is_component_tuple;
    static constexpr bool contains_gotcha    = base_type::contains_gotcha;

    template <typename _Scope = scope::process>
    explicit component_list(const string_t& key, const bool& store = false,
                            const bool& flat = (settings::flat_profile() ||
                                                std::is_same<_Scope, scope::flat>::value))
    : base_type(key, store, flat, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    ~component_list() {}

    component_list(const component_list&) = default;
    component_list(component_list&&)      = default;

    component_list& operator=(const component_list&) = default;
    component_list& operator=(component_list&&) = default;

    component_list(const base_type& rhs)
    : base_type(rhs)
    {
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        base_type::operator-=(static_cast<const base_type&>(rhs));
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        base_type::operator-=(static_cast<base_type&>(rhs));
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        base_type::operator+=(static_cast<const base_type&>(rhs));
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        base_type::operator+=(static_cast<base_type&>(rhs));
        return *this;
    }

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_COMPONENT_LIST_INIT", "");
        };
        return _instance;
    }
};

//======================================================================================//
//
template <typename... _CompTypes, typename... _Types>
class component_list<component_list<_CompTypes...>, _Types...>
: public filt::comp_list_t<_CompTypes..., _Types...>::type
{
    // empty init for friends
    explicit component_list() = default;

    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

public:
    using string_t       = std::string;
    using this_type      = component_list<component_list<_CompTypes...>, _Types...>;
    using init_func_t    = std::function<void(this_type&)>;
    using base_type      = typename filt::comp_list_t<_CompTypes..., _Types...>::type;
    using size_type      = typename base_type::size_type;
    using data_type      = typename base_type::data_type;
    using reference_type = typename base_type::reference_type;
    using type_tuple     = typename base_type::type_tuple;
    using auto_type      = typename base_type::auto_type;

    static constexpr bool is_component_list  = base_type::is_component_list;
    static constexpr bool is_component_tuple = base_type::is_component_tuple;
    static constexpr bool contains_gotcha    = base_type::contains_gotcha;

    template <typename _Scope = scope::process>
    explicit component_list(const string_t& key, const bool& store = false,
                            const bool& flat = (settings::flat_profile() ||
                                                std::is_same<_Scope, scope::flat>::value))
    : base_type(key, store, flat, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    ~component_list() {}

    component_list(const component_list&) = default;
    component_list(component_list&&)      = default;

    component_list& operator=(const component_list&) = default;
    component_list& operator=(component_list&&) = default;

    component_list(const base_type& rhs)
    : base_type(rhs)
    {
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        base_type::operator-=(static_cast<const base_type&>(rhs));
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        base_type::operator-=(static_cast<base_type&>(rhs));
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        base_type::operator+=(static_cast<const base_type&>(rhs));
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        base_type::operator+=(static_cast<base_type&>(rhs));
        return *this;
    }

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_COMPONENT_LIST_INIT", "");
        };
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/details/component_list.hpp"
