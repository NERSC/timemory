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

//======================================================================================//
// variadic list of components
//
template <typename... Types>
class component_list
{
    static const std::size_t num_elements = sizeof...(Types);

    // empty init for friends
    explicit component_list() {}
    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

public:
    using size_type      = int64_t;
    using this_type      = component_list<Types...>;
    using data_type      = std::tuple<Types*...>;
    using reference_type = std::tuple<Types...>;
    using string_hash    = std::hash<string_t>;
    using counter_type   = tim::counted_object<this_type>;
    using counter_void   = tim::counted_object<void>;
    using hashed_type    = tim::hashed_object<this_type>;
    using language_t     = tim::language;
    using init_func_t    = std::function<void(this_type&)>;

    // used by component hybrid
    static constexpr bool is_component_list  = true;
    static constexpr bool is_component_tuple = false;

public:
    using auto_type = auto_list<Types...>;
    // using op_count_t = tim::modifiers<operation::pointer_counter, Types...>;
    using op_count_t = std::tuple<operation::pointer_counter<Types>...>;

public:
    explicit component_list(const string_t& key, const bool& store,
                            const int64_t& ncount = 0, const int64_t& nhash = 0,
                            const language_t& lang = language_t::cxx())
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash((nhash == 0) ? string_hash()(key) : nhash)
    , m_key(key)
    , m_lang(lang)
    , m_identifier("")
    {
        apply<void>::set_value(m_data, nullptr);
        compute_identifier(key, lang);
        init_manager();
        init_storage();
        get_initializer()(*this);
    }

    explicit component_list(const string_t& key, const bool& store,
                            const language_t& lang = language_t::cxx(),
                            const int64_t& ncount = 0, const int64_t& nhash = 0)
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash((nhash == 0) ? string_hash()(key) : nhash)
    , m_key(key)
    , m_lang(lang)
    , m_identifier("")
    {
        apply<void>::set_value(m_data, nullptr);
        compute_identifier(key, lang);
        init_manager();
        init_storage();
        get_initializer()(*this);
    }

    explicit component_list(const string_t&   key,
                            const language_t& lang = language_t::cxx(),
                            const int64_t& ncount = 0, const int64_t& nhash = 0,
                            bool store = true)
    : m_store(store)
    , m_laps(0)
    , m_count(ncount)
    , m_hash((nhash == 0) ? string_hash()(key) : nhash)
    , m_key(key)
    , m_lang(lang)
    , m_identifier("")
    {
        apply<void>::set_value(m_data, nullptr);
        compute_identifier(key, lang);
        init_manager();
        init_storage();
        get_initializer()(*this);
    }

    ~component_list()
    {
        pop();
        using deleter_types = std::tuple<operation::pointer_deleter<Types>...>;
        apply<void>::access<deleter_types>(m_data);
    }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_list(component_list&&) = default;
    component_list& operator=(component_list&&) = default;

    component_list(const component_list& rhs)
    : m_store(rhs.m_store)
    , m_is_pushed(rhs.m_is_pushed)
    , m_laps(rhs.m_laps)
    , m_count(rhs.m_count)
    , m_hash(rhs.m_hash)
    , m_key(rhs.m_key)
    , m_lang(rhs.m_lang)
    , m_identifier(rhs.m_identifier)
    {
        apply<void>::set_value(m_data, nullptr);
        using apply_types = std::tuple<operation::copy<Types>...>;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
    }

    component_list& operator=(const component_list& rhs)
    {
        using deleter_types = std::tuple<operation::pointer_deleter<Types>...>;
        using copy_types    = std::tuple<operation::copy<Types>...>;
        if(this != &rhs)
        {
            m_store      = rhs.m_store;
            m_is_pushed  = rhs.m_is_pushed;
            m_laps       = rhs.m_laps;
            m_count      = rhs.m_count;
            m_hash       = rhs.m_hash;
            m_key        = rhs.m_key;
            m_lang       = std::move(language_t(rhs.m_lang));
            m_identifier = rhs.m_identifier;
            apply<void>::access<deleter_types>(m_data);
            apply<void>::access2<copy_types>(m_data, rhs.m_data);
        }
        return *this;
    }

    component_list clone(const int64_t& nhash, bool store)
    {
        component_list tmp(*this);
        tmp.m_hash  = nhash;
        tmp.m_store = store;
        return tmp;
    }

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return num_elements; }
    static constexpr std::size_t available_size()
    {
        using implemented_tuple_t = implemented<Types...>;
        return std::tuple_size<implemented_tuple_t>::value;
    }

    //----------------------------------------------------------------------------------//
    // function for default initialization
    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            tim::env::initialize(al, "TIMEMORY_COMPONENT_LIST_INIT", "");
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    // insert into graph
    inline void push()
    {
        uint64_t count = 0;
        apply<void>::access<op_count_t>(m_data, std::ref(count));
        if(m_store && !m_is_pushed && count > 0)
        {
            using apply_types = std::tuple<
                operation::pointer_operator<Types, operation::reset<Types>>...>;
            apply<void>::access<apply_types>(m_data);

            using insert_types = std::tuple<
                operation::pointer_operator<Types, operation::insert_node<Types>>...>;
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = true;
            // insert node or find existing node
            apply<void>::access<insert_types>(m_data, m_identifier, m_hash);
        }
    }

    //----------------------------------------------------------------------------------//
    // pop out of grapsh
    inline void pop()
    {
        if(m_store && m_is_pushed)
        {
            using apply_types = std::tuple<
                operation::pointer_operator<Types, operation::pop_node<Types>>...>;
            // set the current node to the parent node
            apply<void>::access<apply_types>(m_data);
            // avoid pushing/popping when already pushed/popped
            m_is_pushed = false;
        }
    }

    //----------------------------------------------------------------------------------//
    // measure functions
    void measure()
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::measure<Types>>...>;
        apply<void>::access<apply_types>(m_data);
    }

    //----------------------------------------------------------------------------------//
    // start/stop functions
    void start()
    {
        using prior_apply_types = std::tuple<
            operation::pointer_operator<Types, operation::priority_start<Types>>...>;
        using stand_apply_types = std::tuple<
            operation::pointer_operator<Types, operation::standard_start<Types>>...>;

        push();
        ++m_laps;
        // start components
        apply<void>::access<prior_apply_types>(m_data);
        apply<void>::access<stand_apply_types>(m_data);
    }

    void stop()
    {
        using prior_apply_types = std::tuple<
            operation::pointer_operator<Types, operation::priority_stop<Types>>...>;
        using stand_apply_types = std::tuple<
            operation::pointer_operator<Types, operation::standard_stop<Types>>...>;
        // stop components
        apply<void>::access<prior_apply_types>(m_data);
        apply<void>::access<stand_apply_types>(m_data);
        // pop them off the running stack
        pop();
    }

    void conditional_start()
    {
        push();
        // start, if not already started
        using prior_apply_types = std::tuple<operation::pointer_operator<
            Types, operation::conditional_priority_start<Types>>...>;
        using stand_apply_types = std::tuple<operation::pointer_operator<
            Types, operation::conditional_standard_start<Types>>...>;
        apply<void>::access<prior_apply_types>(m_data);
        apply<void>::access<stand_apply_types>(m_data);
    }

    void conditional_stop()
    {
        // stop, if not already stopped
        using prior_apply_types = std::tuple<operation::pointer_operator<
            Types, operation::conditional_priority_stop<Types>>...>;
        using stand_apply_types = std::tuple<operation::pointer_operator<
            Types, operation::conditional_standard_stop<Types>>...>;
        apply<void>::access<prior_apply_types>(m_data);
        apply<void>::access<stand_apply_types>(m_data);
        // pop them off the running stack
        pop();
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    void mark_begin()
    {
        using apply_types = std::tuple<
            operation::pointer_operator<Types, operation::mark_begin<Types>>...>;
        apply<void>::access<apply_types>(m_data);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    void mark_end()
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::mark_end<Types>>...>;
        apply<void>::access<apply_types>(m_data);
    }

    //----------------------------------------------------------------------------------//
    // recording
    //
    this_type& record()
    {
        ++m_laps;
        {
            using apply_types = std::tuple<
                operation::pointer_operator<Types, operation::record<Types>>...>;
            apply<void>::access<apply_types>(m_data);
        }
        return *this;
    }

    //----------------------------------------------------------------------------------//
    void reset()
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::reset<Types>>...>;
        apply<void>::access<apply_types>(m_data);
        m_laps = 0;
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::minus<Types>>...>;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::minus<Types>>...>;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps -= rhs.m_laps;
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::plus<Types>>...>;
        apply<void>::access2<apply_types>(m_data, rhs.m_data);
        m_laps += rhs.m_laps;
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::plus<Types>>...>;
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
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::minus<Types>>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::plus<Types>>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::multiply<Types>>...>;
        apply<void>::access<apply_types>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        using apply_types =
            std::tuple<operation::pointer_operator<Types, operation::divide<Types>>...>;
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
        uint64_t count = 0;
        apply<void>::access<op_count_t>(obj.m_data, std::ref(count));
        if(count < 1)
            return os;
        {
            // stop, if not already stopped
            using apply_types = std::tuple<operation::pointer_operator<
                Types, operation::conditional_stop<Types>>...>;
            apply<void>::access<apply_types>(obj.m_data);
        }
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        {
            using apply_types = std::tuple<operation::print<Types>...>;
            apply<void>::access_with_indices<apply_types>(obj.m_data, std::ref(ss_data),
                                                          false);
        }
        if(ss_data.str().length() > 0)
        {
            if(obj.m_print_prefix)
            {
                obj.update_identifier();
                ss_prefix << std::setw(output_width()) << std::left << obj.m_identifier
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
        using apply_types = std::tuple<operation::pointer_operator<
            Types, operation::serialization<Types, Archive>>...>;
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
        apply<void>::type_access<operation::print_storage, reference_type>();
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
    void              rekey(const string_t& _key) { compute_identifier(_key, m_lang); }

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

    template <typename _Tp, tim::enable_if_t<std::is_pointer<_Tp>::value, char> = 0>
    _Tp& get()
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    template <typename _Tp, tim::enable_if_t<(!std::is_pointer<_Tp>::value), char> = 0>
    _Tp*& get()
    {
        return std::get<index_of<_Tp*, data_type>::value>(m_data);
    }

    template <typename _Tp, tim::enable_if_t<std::is_pointer<_Tp>::value, char> = 0>
    const _Tp& get() const
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    template <typename _Tp, tim::enable_if_t<(!std::is_pointer<_Tp>::value), char> = 0>
    const _Tp* get() const
    {
        return std::get<index_of<_Tp*, data_type>::value>(m_data);
    }

    //----------------------------------------------------------------------------------//
    template <typename _Tp, typename... _Args,
              tim::enable_if_t<(is_one_of<_Tp, reference_type>::value == true), int> = 0>
    void init(_Args&&... _args)
    {
        if(!trait::is_available<_Tp>::value)
        {
            static std::atomic<int> _count;
            if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
            {
                std::string _id = tim::demangle(typeid(_Tp).name());
                printf("[component_list::init]> skipping unavailable type '%s'...\n",
                       _id.c_str());
            }
            return;
        }

        auto&& _obj = get<_Tp>();
        if(!_obj)
        {
            if(settings::debug())
            {
                std::string _id = tim::demangle(typeid(_Tp).name());
                printf("[component_list::init]> initializing type '%s'...\n",
                       _id.c_str());
            }
            _obj = new _Tp(std::forward<_Args>(_args)...);
            compute_identifier_extra(_obj);
        }
        else
        {
            static std::atomic<int> _count;
            if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
            {
                std::string _id = tim::demangle(typeid(_Tp).name());
                printf(
                    "[component_list::init]> skipping re-initialization of type"
                    " \"%s\"...\n",
                    _id.c_str());
            }
        }
    }

    template <typename _Tp, typename... _Args,
              tim::enable_if_t<(is_one_of<_Tp, reference_type>::value == false), int> = 0>
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
    //
    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, reference_type>::value == true), int> = 0>
    void type_apply(_Func&& _func, _Args&&... _args)
    {
        auto&& _obj = get<_Tp>();
        ((*_obj).*(_func))(std::forward<_Args>(_args)...);
    }

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
    bool              m_is_pushed    = false;
    bool              m_print_prefix = true;
    bool              m_print_laps   = true;
    int64_t           m_laps         = 0;
    int64_t           m_count        = 0;
    int64_t           m_hash         = 0;
    string_t          m_key          = "";
    language_t        m_lang         = language_t::cxx();
    string_t          m_identifier   = "";
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
        compute_identifier_extra(key, lang);
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

    void compute_identifier_extra(const string_t& key, const language_t&)
    {
        using apply_types = std::tuple<
            operation::pointer_operator<Types, operation::set_prefix<Types>>...>;
        apply<void>::access<apply_types>(m_data, key);
    }

    template <typename _Tp, enable_if_t<(trait::requires_prefix<_Tp>::value), int> = 0>
    void compute_identifier_extra(_Tp* obj)
    {
        if(obj)
            obj->prefix = m_key;
    }

    template <typename _Tp,
              enable_if_t<(trait::requires_prefix<_Tp>::value == false), int> = 0>
    void compute_identifier_extra(_Tp*)
    {
    }

public:
    static void init_manager();
    static void init_storage()
    {
        // apply<void>::type_access<operation::init_storage, implemented<Types...>>();
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#include "timemory/manager.hpp"

//======================================================================================//

template <typename... Types>
void
tim::component_list<Types...>::init_manager()
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
get(tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::component_list<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::component_list<Types...>>(obj).data()))
{
    using obj_type = tim::component_list<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std

//--------------------------------------------------------------------------------------//
