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

/** \file timemory/variadic/component_hybrid.hpp
 * \headerfile variadic/component_hybrid.hpp "timemory/variadic/component_hybrid.hpp"
 * This is the C++ class that bundles together components and enables
 * operation on the components as a single entity
 *
 */

#pragma once

#include "timemory/general/source_location.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

//======================================================================================//

namespace tim
{
//======================================================================================//
// variadic list of components
//
template <typename CompTuple, typename CompList>
class component_hybrid
{
    static_assert((concepts::is_stack_wrapper<CompTuple>::value &&
                   concepts::is_heap_wrapper<CompList>::value),
                  "Error! CompTuple must be tim::component_tuple<...> and CompList "
                  "must be tim::component_list<...>");

    static const std::size_t num_elements = CompTuple::size() + CompList::size();

    // manager is friend so can use above
    friend class manager;

    template <typename T, typename L>
    friend class auto_hybrid;

public:
    using this_type       = component_hybrid<CompTuple, CompList>;
    using tuple_t         = typename CompTuple::component_type;
    using list_t          = typename CompList::component_type;
    using tuple_data_type = typename tuple_t::data_type;
    using list_data_type  = typename list_t::data_type;
    using data_type       = decltype(
        std::tuple_cat(std::declval<tuple_t>().data(), std::declval<list_t>().data()));
    using tuple_type =
        tim::tuple_concat_t<typename tuple_t::tuple_type, typename list_t::tuple_type>;

    using tuple_type_list = typename tuple_t::reference_type;
    using list_type_list  = typename list_t::reference_type;

    // used by gotcha
    using component_type = component_hybrid<tuple_t, list_t>;
    using auto_type      = auto_hybrid<tuple_t, list_t>;
    using type = component_hybrid<typename tuple_t::type, typename list_t::type>;

    static constexpr bool is_component = false;
    static constexpr bool has_gotcha_v = (tuple_t::has_gotcha_v || list_t::has_gotcha_v);
    static constexpr bool has_user_bundle_v =
        (tuple_t::has_user_bundle_v || list_t::has_user_bundle_v);

    using size_type           = int64_t;
    using captured_location_t = source_location::captured;
    using initializer_type    = std::function<void(this_type&)>;

public:
    //----------------------------------------------------------------------------------//
    //
    static void init_storage()
    {
        tuple_t::init_storage();
        list_t::init_storage();
    }

    //----------------------------------------------------------------------------------//
    //
    static initializer_type& get_initializer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }

public:
    explicit component_hybrid()
    : m_store(false)
    , m_tuple()
    , m_list()
    {}

    template <typename Func = initializer_type>
    explicit component_hybrid(const string_t& _key, const bool& _store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&   _func  = this_type::get_initializer())
    : m_store(_store)
    , m_tuple(_key, false, _scope)
    , m_list(_key, false, _scope)
    {
        _func(*this);
    }

    template <typename Func = initializer_type>
    explicit component_hybrid(const captured_location_t& _loc, const bool& _store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&   _func  = this_type::get_initializer())
    : m_store(_store)
    , m_tuple(_loc, false, _scope)
    , m_list(_loc, false, _scope)
    {
        _func(*this);
    }

    template <typename Func = initializer_type>
    explicit component_hybrid(size_t _hash, const bool& _store = true,
                              scope::config _scope = scope::get_default(),
                              const Func&   _func  = this_type::get_initializer())
    : m_store(_store)
    , m_tuple(_hash, false, _scope)
    , m_list(_hash, false, _scope)
    {
        _func(*this);
    }

    ~component_hybrid() {}

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_hybrid(const tuple_t& _tuple, const list_t& _list)
    : m_tuple(_tuple)
    , m_list(_list)
    {}

    component_hybrid(const component_hybrid&)     = default;
    component_hybrid(component_hybrid&&) noexcept = default;

    component_hybrid& operator=(const component_hybrid& rhs) = default;
    component_hybrid& operator=(component_hybrid&&) noexcept = default;

    component_hybrid clone(bool _store, scope::config _scope)
    {
        return component_hybrid(m_tuple.clone(_store, _scope),
                                m_list.clone(_store, _scope));
    }

public:
    tuple_t&       get_tuple() { return m_tuple; }
    const tuple_t& get_tuple() const { return m_tuple; }
    list_t&        get_list() { return m_list; }
    const list_t&  get_list() const { return m_list; }

    tuple_t&       get_first() { return m_tuple; }
    const tuple_t& get_first() const { return m_tuple; }
    list_t&        get_second() { return m_list; }
    const list_t&  get_second() const { return m_list; }

    tuple_t&       get_lhs() { return m_tuple; }
    const tuple_t& get_lhs() const { return m_tuple; }
    list_t&        get_rhs() { return m_list; }
    const list_t&  get_rhs() const { return m_list; }

public:
    int64_t  laps() const { return m_tuple.laps(); }
    string_t key() const { return m_tuple.key(); }
    uint64_t hash() const { return m_tuple.hash(); }
    bool     store() const { return m_tuple.store(); }
    void     rekey(const string_t& _key)
    {
        m_tuple.rekey(_key);
        m_list.rekey(_key);
    }

public:
    //----------------------------------------------------------------------------------//
    // get the size
    //
    static constexpr std::size_t size() { return num_elements; }

    //----------------------------------------------------------------------------------//
    // insert into graph
    void push()
    {
        m_tuple.push();
        m_list.push();
    }

    //----------------------------------------------------------------------------------//
    // pop out of graph
    void pop()
    {
        m_tuple.pop();
        m_list.pop();
    }

    //----------------------------------------------------------------------------------//
    // measure functions
    template <typename... Args>
    void measure(Args&&... args)
    {
        m_tuple.measure(std::forward<Args>(args)...);
        m_list.measure(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // sample functions
    template <typename... Args>
    void sample(Args&&... args)
    {
        m_tuple.sample(std::forward<Args>(args)...);
        m_list.sample(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // start/stop functions
    template <typename... Args>
    void start(Args&&... args)
    {
        push();
        assemble(*this);

        m_tuple.start(std::forward<Args>(args)...);
        m_list.start(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void stop(Args&&... args)
    {
        m_tuple.stop(std::forward<Args>(args)...);
        m_list.stop(std::forward<Args>(args)...);

        derive(*this);
        pop();
    }

    //----------------------------------------------------------------------------------//
    // construct the objects that have constructors with matching arguments
    //
    template <typename... Args>
    void construct(Args&&... args)
    {
        m_tuple.construct(std::forward<Args>(args)...);
        m_list.construct(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    /// provide preliminary info to the objects with matching arguments
    //
    template <typename... Args>
    void assemble(Args&&... args)
    {
        m_tuple.assemble(std::forward<Args>(args)...);
        m_list.assemble(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    /// provide conclusive info to the objects with matching arguments
    //
    template <typename... Args>
    void derive(Args&&... args)
    {
        m_tuple.derive(std::forward<Args>(args)...);
        m_list.derive(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... Args>
    void mark_begin(Args&&... args)
    {
        m_tuple.mark_begin(std::forward<Args>(args)...);
        m_list.mark_begin(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... Args>
    void mark_end(Args&&... args)
    {
        m_tuple.mark_end(std::forward<Args>(args)...);
        m_list.mark_end(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // store a value
    //
    template <typename... Args>
    void store(Args&&... args)
    {
        m_tuple.store(std::forward<Args>(args)...);
        m_list.store(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform a auditd operation (typically for GOTCHA)
    //
    template <typename... Args>
    void audit(Args&&... args)
    {
        m_tuple.audit(std::forward<Args>(args)...);
        m_list.audit(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform an add_secondary operation
    //
    template <typename... Args>
    void add_secondary(Args&&... args)
    {
        m_tuple.add_secondary(std::forward<Args>(args)...);
        m_list.add_secondary(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class OpT, typename... Args>
    void invoke(Args&&... args)
    {
        m_tuple.template invoke<OpT>(std::forward<Args>(args)...);
        m_list.template invoke<OpT>(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // recording
    //
    template <typename... Args>
    this_type& record(Args&&... args)
    {
        m_tuple.record(std::forward<Args>(args)...);
        m_list.record(std::forward<Args>(args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // reset data
    //
    template <typename... Args>
    void reset(Args&&... args)
    {
        m_tuple.reset(std::forward<Args>(args)...);
        m_list.reset(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // get data
    //
    template <typename... Args>
    auto get(Args&&... args) const
    {
        return std::tuple_cat(get_lhs().get(std::forward<Args>(args)...),
                              get_rhs().get(std::forward<Args>(args)...));
    }

    //----------------------------------------------------------------------------------//
    // reset data
    //
    template <typename... Args>
    auto get_labeled(Args&&... args) const
    {
        return std::tuple_cat(get_lhs().get_labeled(std::forward<Args>(args)...),
                              get_rhs().get_labeled(std::forward<Args>(args)...));
    }

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs)
    {
        m_tuple -= rhs.m_tuple;
        m_list -= rhs.m_list;
        return *this;
    }

    this_type& operator-=(this_type& rhs)
    {
        m_tuple -= rhs.m_tuple;
        m_list -= rhs.m_list;
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        m_tuple += rhs.m_tuple;
        m_list += rhs.m_list;
        return *this;
    }

    this_type& operator+=(this_type& rhs)
    {
        m_tuple += rhs.m_tuple;
        m_list += rhs.m_list;
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename OpT>
    this_type& operator-=(OpT&& rhs)
    {
        m_tuple -= std::forward<OpT>(rhs);
        m_list -= std::forward<OpT>(rhs);
        return *this;
    }

    template <typename OpT>
    this_type& operator+=(OpT&& rhs)
    {
        m_tuple += std::forward<OpT>(rhs);
        m_list += std::forward<OpT>(rhs);
        return *this;
    }

    template <typename OpT>
    this_type& operator*=(OpT&& rhs)
    {
        m_tuple *= std::forward<OpT>(rhs);
        m_list *= std::forward<OpT>(rhs);
        return *this;
    }

    template <typename OpT>
    this_type& operator/=(OpT&& rhs)
    {
        m_tuple /= std::forward<OpT>(rhs);
        m_list /= std::forward<OpT>(rhs);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple += rhs.m_tuple;
        tmp.m_list += rhs.m_list;
        return tmp;
    }

    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple -= rhs.m_tuple;
        tmp.m_list -= rhs.m_list;
        return tmp;
    }

    template <typename OpT>
    friend this_type operator*(const this_type& lhs, OpT&& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple *= std::forward<OpT>(rhs);
        tmp.m_list *= std::forward<OpT>(rhs);
        return tmp;
    }

    template <typename OpT>
    friend this_type operator/(const this_type& lhs, OpT&& rhs)
    {
        this_type tmp(lhs);
        tmp.m_tuple /= std::forward<OpT>(rhs);
        tmp.m_list /= std::forward<OpT>(rhs);
        return tmp;
    }

    //----------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        if((obj.m_tuple.hash() + obj.m_list.hash()) == 0)
            return os;
        std::stringstream tss, lss;

        obj.m_tuple.template print<true, false>(tss);
        obj.m_list.template print<false, false>(lss);

        if(tss.str().length() > 0)
            os << tss.str();
        if(tss.str().length() > 0 && lss.str().length() > 0)
            os << ", ";
        if(lss.str().length() > 0)
            os << lss.str();

        if(obj.m_tuple.laps() > 0)
            os << " [laps: " << obj.m_tuple.laps() << "]";

        return os;
    }

    //----------------------------------------------------------------------------------//
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        ar(cereal::make_nvp("tuple", m_tuple));
        ar(cereal::make_nvp("list", m_list));
    }

    //----------------------------------------------------------------------------------//
    static void print_storage()
    {
        tuple_t::print_storage();
        list_t::print_storage();
    }

public:
    data_type data() const { return std::tuple_cat(m_tuple.data(), m_list.data()); }

public:
    //----------------------------------------------------------------------------------//
    //  get access to a type
    //
    template <typename Tp,
              enable_if_t<(is_one_of<Tp, tuple_type_list>::value == true), int> = 0>
    auto get() -> decltype(std::declval<tuple_t>().template get<Tp>())
    {
        return m_tuple.template get<Tp>();
    }

    template <typename Tp,
              enable_if_t<(is_one_of<Tp, list_type_list>::value == true), int> = 0>
    auto get() -> decltype(std::declval<list_t>().template get<Tp>())
    {
        return m_list.template get<Tp>();
    }

    void get(void*& ptr, size_t _hash)
    {
        m_tuple.get(ptr, _hash);
        if(!ptr)
            m_list.get(ptr, _hash);
    }

    //----------------------------------------------------------------------------------//
    /// this is a simple alternative to get<T>() when used from SFINAE in operation
    /// namespace which has a struct get also templated. Usage there can cause error
    /// with older compilers
    template <typename T, enable_if_t<(is_one_of<T, tuple_type_list>::value), int> = 0>
    auto get_component()
    {
        return m_tuple.template get_component<T>();
    }

    template <typename T, enable_if_t<(is_one_of<T, list_type_list>::value), long> = 0>
    auto get_component()
    {
        return m_list.template get_component<T>();
    }

    template <typename Tp, typename... Args>
    void init(Args&&... args)
    {
        m_list.template init<Tp>(std::forward<Args>(args)...);
    }

    template <typename... Tp, typename... Args>
    void initialize(Args&&... args)
    {
        m_list.template initialize<Tp...>(std::forward<Args>(args)...);
    }

public:
    //----------------------------------------------------------------------------------//
    //  apply a member function to a type
    //
    template <typename Tp, typename Func, typename... Args,
              enable_if_t<(is_one_of<Tp, tuple_type_list>::value), int> = 0,
              enable_if_t<!(is_one_of<Tp, list_type_list>::value), int> = 0>
    void type_apply(Func&& _func, Args&&... args)
    {
        m_tuple.template type_apply<Tp>(_func, std::forward<Args>(args)...);
    }

    template <typename Tp, typename Func, typename... Args,
              enable_if_t<!(is_one_of<Tp, tuple_type_list>::value), int> = 0,
              enable_if_t<(is_one_of<Tp, list_type_list>::value), int>   = 0>
    void type_apply(Func&& _func, Args&&... args)
    {
        m_list.template type_apply<Tp>(_func, std::forward<Args>(args)...);
    }

    template <typename Tp, typename Func, typename... Args,
              enable_if_t<!(is_one_of<Tp, tuple_type_list>::value), int> = 0,
              enable_if_t<!(is_one_of<Tp, list_type_list>::value), int>  = 0>
    void type_apply(Func&&, Args&&...)
    {}

protected:
    // objects
    bool    m_store = false;
    tuple_t m_tuple = tuple_t{};
    list_t  m_list  = list_t{};
};

//--------------------------------------------------------------------------------------//

template <typename... T>
using component_hybrid_t = typename component_hybrid<T...>::type;

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//
