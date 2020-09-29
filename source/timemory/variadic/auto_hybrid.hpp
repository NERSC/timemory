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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file timemory/variadic/auto_hybrid.hpp
 * \headerfile timemory/variadic/auto_hybrid.hpp "timemory/variadic/auto_hybrid.hpp"
 * Automatic starting and stopping of components. Accept a component_tuple as first
 * type and component_list as second type
 *
 */

#pragma once

#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <string>

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename CompTuple, typename CompList>
class auto_hybrid
{
    static_assert((concepts::is_stack_wrapper<CompTuple>::value &&
                   concepts::is_heap_wrapper<CompList>::value),
                  "Error! CompTuple must be tim::component_tuple<...> and CompList "
                  "must be tim::component_list<...>");

public:
    using this_type           = auto_hybrid<CompTuple, CompList>;
    using base_type           = component_hybrid<CompTuple, CompList>;
    using auto_type           = this_type;
    using tuple_t             = typename base_type::tuple_t;
    using list_t              = typename base_type::list_t;
    using component_type      = typename base_type::component_type;
    using data_type           = typename component_type::data_type;
    using tuple_type          = typename component_type::tuple_type;
    using tuple_type_list     = typename component_type::tuple_type_list;
    using list_type_list      = typename component_type::list_type_list;
    using string_t            = std::string;
    using captured_location_t = typename component_type::captured_location_t;
    using type =
        convert_t<typename component_type::type, auto_hybrid<type_list<>, type_list<>>>;
    using initializer_type = std::function<void(this_type&)>;

    static constexpr bool is_component      = false;
    static constexpr bool has_gotcha_v      = component_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v = component_type::has_user_bundle_v;

public:
    //----------------------------------------------------------------------------------//
    //
    static void init_storage() { component_type::init_storage(); }

    //----------------------------------------------------------------------------------//
    //
    static initializer_type& get_initializer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }

public:
    template <typename FuncT = initializer_type>
    explicit auto_hybrid(const string_t&, scope::config = scope::get_default(),
                         bool         report_at_exit = settings::destructor_report(),
                         const FuncT& _func          = this_type::get_initializer());

    template <typename FuncT = initializer_type>
    explicit auto_hybrid(const captured_location_t&, scope::config = scope::get_default(),
                         bool         report_at_exit = settings::destructor_report(),
                         const FuncT& _func          = this_type::get_initializer());

    template <typename FuncT = initializer_type>
    explicit auto_hybrid(size_t, scope::config = scope::get_default(),
                         bool         report_at_exit = settings::destructor_report(),
                         const FuncT& _func          = this_type::get_initializer());

    explicit auto_hybrid(component_type& tmp, scope::config = scope::get_default(),
                         bool            report_at_exit = settings::destructor_report());

    ~auto_hybrid();

    // copy and move
    auto_hybrid(const this_type&)     = default;
    auto_hybrid(this_type&&) noexcept = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) noexcept = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    component_type&       get_component() { return m_temporary; }
    const component_type& get_component() const { return m_temporary; }

    operator component_type&() { return m_temporary; }
    operator const component_type&() const { return m_temporary; }

    // partial interface to underlying component_hybrid
    void push()
    {
        if(m_enabled)
            m_temporary.push();
    }
    void pop()
    {
        if(m_enabled)
            m_temporary.pop();
    }
    template <typename... Args>
    void measure(Args&&... args)
    {
        if(m_enabled)
            m_temporary.measure(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void sample(Args&&... args)
    {
        if(m_enabled)
            m_temporary.sample(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void start(Args&&... args)
    {
        if(m_enabled)
            m_temporary.start(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void stop(Args&&... args)
    {
        if(m_enabled)
            m_temporary.stop(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void assemble(Args&&... args)
    {
        if(m_enabled)
            m_temporary.assemble(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void derive(Args&&... args)
    {
        if(m_enabled)
            m_temporary.derive(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void mark_begin(Args&&... args)
    {
        if(m_enabled)
            m_temporary.mark_begin(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void mark_end(Args&&... args)
    {
        if(m_enabled)
            m_temporary.mark_end(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void store(Args&&... args)
    {
        if(m_enabled)
            m_temporary.store(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void audit(Args&&... args)
    {
        if(m_enabled)
            m_temporary.audit(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void add_secondary(Args&&... args)
    {
        if(m_enabled)
            m_temporary.add_secondary(std::forward<Args>(args)...);
    }
    template <template <typename> class OpT, typename... Args>
    void invoke(Args&&... _args)
    {
        if(m_enabled)
            m_temporary.template invoke<OpT>(std::forward<Args>(_args)...);
    }
    template <typename... Args>
    auto get(Args&&... args) const
    {
        return m_temporary.get(std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto get_labeled(Args&&... args) const
    {
        return m_temporary.get_labeled(std::forward<Args>(args)...);
    }

    bool enabled() const { return m_enabled; }
    void report_at_exit(bool val) { m_report_at_exit = val; }
    bool report_at_exit() const { return m_report_at_exit; }

    bool      store() const { return m_temporary.store(); }
    data_type data() const { return m_temporary.data(); }
    int64_t   laps() const { return m_temporary.laps(); }
    string_t  key() const { return m_temporary.key(); }
    uint64_t  hash() const { return m_temporary.hash(); }
    void      rekey(const string_t& _key) { m_temporary.rekey(_key); }

public:
    tuple_t&       get_tuple() { return m_temporary.get_tuple(); }
    const tuple_t& get_tuple() const { return m_temporary.get_tuple(); }
    list_t&        get_list() { return m_temporary.get_list(); }
    const list_t&  get_list() const { return m_temporary.get_list(); }

    tuple_t&       get_first() { return m_temporary.get_tuple(); }
    const tuple_t& get_first() const { return m_temporary.get_tuple(); }
    list_t&        get_second() { return m_temporary.get_list(); }
    const list_t&  get_second() const { return m_temporary.get_list(); }

    tuple_t&       get_lhs() { return m_temporary.get_tuple(); }
    const tuple_t& get_lhs() const { return m_temporary.get_tuple(); }
    list_t&        get_rhs() { return m_temporary.get_list(); }
    const list_t&  get_rhs() const { return m_temporary.get_list(); }

    template <typename Tp>
    decltype(auto) get()
    {
        return m_temporary.template get<Tp>();
    }

    template <typename Tp>
    decltype(auto) get() const
    {
        return m_temporary.template get<Tp>();
    }

    template <typename T>
    auto get_component()
        -> decltype(std::declval<component_type>().template get_component<T>())
    {
        return m_temporary.template get_component<T>();
    }

    void get(void*& ptr, size_t _hash) { m_temporary.get(ptr, _hash); }

    template <typename Tp, typename... Args>
    void init(Args&&... _args)
    {
        m_temporary.template init<Tp>(std::forward<Args>(_args)...);
    }

    template <typename... Tp, typename... Args>
    void initialize(Args&&... _args)
    {
        m_temporary.template initialize<Tp...>(std::forward<Args>(_args)...);
    }

public:
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << obj.m_temporary;
        return os;
    }

private:
    bool            m_enabled        = true;
    bool            m_report_at_exit = false;
    component_type  m_temporary;
    component_type* m_reference_object = nullptr;
};

//======================================================================================//

template <typename CompTuple, typename CompList>
template <typename FuncT>
auto_hybrid<CompTuple, CompList>::auto_hybrid(const string_t& object_tag,
                                              scope::config _scope, bool report_at_exit,
                                              const FuncT& _func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary(m_enabled ? component_type(object_tag, m_enabled, _scope)
                        : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        _func(*this);
        m_temporary.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename CompTuple, typename CompList>
template <typename FuncT>
auto_hybrid<CompTuple, CompList>::auto_hybrid(const captured_location_t& object_loc,
                                              scope::config _scope, bool report_at_exit,
                                              const FuncT& _func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary(m_enabled ? component_type(object_loc, m_enabled, _scope)
                        : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        _func(*this);
        m_temporary.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename CompTuple, typename CompList>
template <typename FuncT>
auto_hybrid<CompTuple, CompList>::auto_hybrid(size_t _hash, scope::config _scope,
                                              bool report_at_exit, const FuncT& _func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary(m_enabled ? component_type(_hash, m_enabled, _scope) : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        _func(*this);
        m_temporary.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename CompTuple, typename CompList>
auto_hybrid<CompTuple, CompList>::auto_hybrid(component_type& tmp, scope::config _scope,
                                              bool report_at_exit)
: m_enabled(true)
, m_report_at_exit(report_at_exit)
, m_temporary(tmp.clone(true, _scope))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        m_temporary.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename CompTuple, typename CompList>
auto_hybrid<CompTuple, CompList>::~auto_hybrid()
{
    if(m_enabled)
    {
        // stop the timer
        m_temporary.stop();

        // report timer at exit
        if(m_report_at_exit)
        {
            std::stringstream ss;
            ss << m_temporary;
            if(ss.str().length() > 0)
                std::cout << ss.str() << std::endl;
        }

        if(m_reference_object)
        {
            *m_reference_object += m_temporary;
        }
    }
}

//======================================================================================//

template <typename TupleT, typename ListT>
auto
get(const auto_hybrid<TupleT, ListT>& _obj)
{
    return get(_obj.get_component());
}

//--------------------------------------------------------------------------------------//

template <typename TupleT, typename ListT>
auto
get_labeled(const auto_hybrid<TupleT, ListT>& _obj)
{
    return get_labeled(_obj.get_component());
}

//--------------------------------------------------------------------------------------//

template <typename... T>
using auto_hybrid_t = typename auto_hybrid<T...>::type;

//======================================================================================//

template <typename... Lhs, typename... Rhs>
class auto_hybrid<type_list<Lhs...>, type_list<Rhs...>>
: public auto_hybrid<component_tuple<Lhs...>, component_list<Rhs...>>
{
public:
    using real_type = auto_hybrid<component_tuple<Lhs...>, component_list<Rhs...>>;

    using this_type       = typename real_type::base_type;
    using base_type       = typename real_type::base_type;
    using auto_type       = typename real_type::auto_type;
    using tuple_t         = typename real_type::tuple_t;
    using list_t          = typename real_type::list_t;
    using component_type  = typename real_type::component_type;
    using data_type       = typename real_type::data_type;
    using tuple_type      = typename real_type::tuple_type;
    using tuple_type_list = typename real_type::tuple_type_list;
    using list_type_list  = typename real_type::list_type_list;

    template <typename... Args>
    auto_hybrid(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}
};

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//
