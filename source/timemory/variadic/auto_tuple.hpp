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

/** \file timemory/variadic/auto_tuple.hpp
 * \headerfile timemory/variadic/auto_tuple.hpp "timemory/variadic/auto_tuple.hpp"
 * Automatic starting and stopping of components. Accept unlimited number of
 * parameters. The constructor starts the components, the destructor stops the
 * components
 *
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_TUPLE()
 *    \param TIMEMORY_BASIC_AUTO_TUPLE()
 *    \param auto t = TIMEMORY_AUTO_TUPLE_OBJ()
 *    \param auto t = TIMEMORY_BASIC_AUTO_TUPLE_OBJ()
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/general/source_location.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
class auto_tuple
{
public:
    using this_type           = auto_tuple<Types...>;
    using base_type           = component_tuple<Types...>;
    using auto_type           = this_type;
    using component_type      = typename base_type::component_type;
    using type_tuple          = typename component_type::type_tuple;
    using data_type           = typename component_type::data_type;
    using sample_type         = typename component_type::sample_type;
    using type                = convert_t<typename component_type::type, auto_tuple<>>;
    using string_t            = std::string;
    using initializer_type    = std::function<void(this_type&)>;
    using captured_location_t = typename component_type::captured_location_t;

    // used by component hybrid and gotcha
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = false;
    static constexpr bool is_component_hybrid = false;
    static constexpr bool is_component_type   = false;
    static constexpr bool is_auto_list        = false;
    static constexpr bool is_auto_tuple       = true;
    static constexpr bool is_auto_hybrid      = false;
    static constexpr bool is_auto_type        = true;
    static constexpr bool is_component        = false;
    static constexpr bool has_gotcha_v        = component_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v   = component_type::has_user_bundle_v;

public:
    using concat_type = concat<Types...>;

    template <typename T>
    static constexpr bool get_config()
    {
        using var_config_t = contains_one_of_t<variadic::is_config, concat_type>;
        return (is_one_of<T, var_config_t>::value);
    }

    template <typename T, typename Config>
    static constexpr bool get_config(Config&&)
    {
        using var_config_t = contains_one_of_t<variadic::is_config, concat_type>;
        return (is_one_of<T, var_config_t>::value || is_one_of<T, Config>::value);
    }

public:
    //
    //----------------------------------------------------------------------------------//
    //
    static void init_storage() { component_type::init_storage(); }
    //
    //----------------------------------------------------------------------------------//
    //
    static initializer_type& get_initializer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    static initializer_type& get_finalizer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }

public:
    template <typename... T, typename Init = initializer_type>
    explicit auto_tuple(const string_t&, variadic::config<T...>,
                        const Init& = this_type::get_initializer());

    template <typename... T, typename Init = initializer_type>
    explicit auto_tuple(const captured_location_t&, variadic::config<T...>,
                        const Init& = this_type::get_initializer());

    template <typename Init = initializer_type>
    explicit auto_tuple(const string_t&, scope::config = scope::get_default(),
                        bool report_at_exit = settings::destructor_report(),
                        const Init&         = this_type::get_initializer());

    template <typename Init = initializer_type>
    explicit auto_tuple(const captured_location_t&, scope::config = scope::get_default(),
                        bool report_at_exit = settings::destructor_report(),
                        const Init&         = this_type::get_initializer());

    template <typename Init = initializer_type>
    explicit auto_tuple(size_t, scope::config = scope::get_default(),
                        bool report_at_exit = settings::destructor_report(),
                        const Init&         = this_type::get_initializer());

    explicit auto_tuple(component_type& tmp, scope::config = scope::get_default(),
                        bool            report_at_exit = settings::destructor_report());

    template <typename Init, typename Arg, typename... Args>
    auto_tuple(const string_t&, bool store, scope::config _scope, const Init&, Arg&&,
               Args&&...);

    template <typename Init, typename Arg, typename... Args>
    auto_tuple(const captured_location_t&, bool store, scope::config _scope, const Init&,
               Arg&&, Args&&...);

    template <typename Init, typename Arg, typename... Args>
    auto_tuple(size_t, bool store, scope::config _scope, const Init&, Arg&&, Args&&...);

    ~auto_tuple();

    // copy and move
    auto_tuple(const this_type&) = default;
    auto_tuple(this_type&&)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    component_type&       get_component() { return m_temporary_object; }
    const component_type& get_component() const { return m_temporary_object; }

    operator component_type&() { return m_temporary_object; }
    operator const component_type&() const { return m_temporary_object; }

    // partial interface to underlying component_tuple
    void push()
    {
        if(m_enabled)
            m_temporary_object.push();
    }
    void pop()
    {
        if(m_enabled)
            m_temporary_object.pop();
    }
    template <typename... Args>
    void measure(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.measure(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void sample(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.sample(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void start(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.start(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void stop(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.stop(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void assemble(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.assemble(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void derive(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.derive(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void mark_begin(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.mark_begin(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void mark_end(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.mark_end(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void store(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.store(std::forward<Args>(args)...);
    }
    template <typename... Args>
    void audit(Args&&... args)
    {
        if(m_enabled)
            m_temporary_object.audit(std::forward<Args>(args)...);
    }
    template <template <typename> class OpT, typename... Args>
    void invoke(Args&&... _args)
    {
        if(m_enabled)
            m_temporary_object.template invoke<OpT>(std::forward<Args>(_args)...);
    }
    template <typename... Args>
    auto get(Args&&... args) const
    {
        return m_temporary_object.get(std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto get_labeled(Args&&... args) const
    {
        return m_temporary_object.get_labeled(std::forward<Args>(args)...);
    }

    bool enabled() const { return m_enabled; }
    void report_at_exit(bool val) { m_report_at_exit = val; }
    bool report_at_exit() const { return m_report_at_exit; }

    bool             store() const { return m_temporary_object.store(); }
    data_type&       data() { return m_temporary_object.data(); }
    const data_type& data() const { return m_temporary_object.data(); }
    int64_t          laps() const { return m_temporary_object.laps(); }
    string_t         key() const { return m_temporary_object.key(); }
    uint64_t         hash() const { return m_temporary_object.hash(); }
    void             rekey(const string_t& _key) { m_temporary_object.rekey(_key); }

public:
    template <typename Tp>
    decltype(auto) get()
    {
        return m_temporary_object.template get<Tp>();
    }

    template <typename Tp>
    decltype(auto) get() const
    {
        return m_temporary_object.template get<Tp>();
    }

    void get(void*& ptr, size_t hash) { m_temporary_object.get(ptr, hash); }

    template <typename T>
    auto get_component()
        -> decltype(std::declval<component_type>().template get_component<T>())
    {
        return m_temporary_object.template get_component<T>();
    }

protected:
    template <typename Func>
    void init(Func&& _init)
    {
        if(m_enabled)
            _init(*this);
    }

    template <typename Func, typename Arg, typename... Args>
    void init(Func&& _init, Arg&& _arg, Args&&... _args)
    {
        if(m_enabled)
        {
            _init(*this);
            m_temporary_object.construct(std::forward<Arg>(_arg),
                                         std::forward<Args>(_args)...);
        }
    }

public:
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << obj.m_temporary_object;
        return os;
    }

protected:
    bool            m_enabled        = true;
    bool            m_report_at_exit = false;
    component_type  m_temporary_object;
    component_type* m_reference_object = nullptr;
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename... T, typename Init>
auto_tuple<Types...>::auto_tuple(const string_t& key, variadic::config<T...> config,
                                 const Init& init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(get_config<variadic::exit_report>(config))
, m_temporary_object(
      m_enabled ? component_type(key, m_enabled, get_config<variadic::flat_scope>(config))
                : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>(config))
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename... T, typename Init>
auto_tuple<Types...>::auto_tuple(const captured_location_t& loc,
                                 variadic::config<T...> config, const Init& init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(get_config<variadic::exit_report>(config))
, m_temporary_object(
      m_enabled ? component_type(loc, m_enabled, get_config<variadic::flat_scope>(config))
                : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename Init>
auto_tuple<Types...>::auto_tuple(const string_t& key, scope::config _scope,
                                 bool report_at_exit, const Init& init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit || get_config<variadic::exit_report>())
, m_temporary_object(m_enabled ? component_type(key, m_enabled, _scope)
                               : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename Init>
auto_tuple<Types...>::auto_tuple(const captured_location_t& loc, scope::config _scope,
                                 bool report_at_exit, const Init& init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit || get_config<variadic::exit_report>())
, m_temporary_object(m_enabled ? component_type(loc, m_enabled, _scope)
                               : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename Init>
auto_tuple<Types...>::auto_tuple(size_t hash, scope::config _scope, bool report_at_exit,
                                 const Init& init_func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit || get_config<variadic::exit_report>())
, m_temporary_object(m_enabled ? component_type(hash, m_enabled, _scope)
                               : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(component_type& tmp, scope::config _scope,
                                 bool report_at_exit)
: m_enabled(true)
, m_report_at_exit(report_at_exit || get_config<variadic::exit_report>())
, m_temporary_object(component_type(tmp.clone(true, _scope)))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename Init, typename Arg, typename... Args>
auto_tuple<Types...>::auto_tuple(const string_t& key, bool store, scope::config _scope,
                                 const Init& init_func, Arg&& arg, Args&&... args)
: m_enabled(store && settings::enabled())
, m_report_at_exit(settings::destructor_report() || get_config<variadic::exit_report>())
, m_temporary_object(m_enabled ? component_type(key, m_enabled, _scope)
                               : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func, std::forward<Arg>(arg), std::forward<Args>(args)...);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename Init, typename Arg, typename... Args>
auto_tuple<Types...>::auto_tuple(const captured_location_t& loc, bool store,
                                 scope::config _scope, const Init& init_func, Arg&& arg,
                                 Args&&... args)
: m_enabled(store && settings::enabled())
, m_report_at_exit(settings::destructor_report() || get_config<variadic::exit_report>())
, m_temporary_object(m_enabled ? component_type(loc, m_enabled, _scope)
                               : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func, std::forward<Arg>(arg), std::forward<Args>(args)...);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename Init, typename Arg, typename... Args>
auto_tuple<Types...>::auto_tuple(size_t hash, bool store, scope::config _scope,
                                 const Init& init_func, Arg&& arg, Args&&... args)
: m_enabled(store && settings::enabled())
, m_report_at_exit(settings::destructor_report() || get_config<variadic::exit_report>())
, m_temporary_object(m_enabled ? component_type(hash, m_enabled, _scope)
                               : component_type{})
, m_reference_object(nullptr)

{
    if(m_enabled)
    {
        init(init_func, std::forward<Arg>(arg), std::forward<Args>(args)...);
        IF_CONSTEXPR(!get_config<variadic::explicit_start>())
        {
            m_temporary_object.start();
        }
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_tuple<Types...>::~auto_tuple()
{
    IF_CONSTEXPR(!get_config<variadic::explicit_stop>())
    {
        if(m_enabled)
        {
            // stop the timer
            m_temporary_object.stop();

            // report timer at exit
            if(m_report_at_exit)
            {
                std::stringstream ss;
                ss << m_temporary_object;
                if(ss.str().length() > 0)
                    std::cout << ss.str() << std::endl;
            }

            if(m_reference_object)
            {
                *m_reference_object += m_temporary_object;
            }
        }
    }
}

//======================================================================================//

template <typename... Types>
auto
get(const auto_tuple<Types...>& _obj)
{
    return get(_obj.get_component());
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const auto_tuple<Types...>& _obj)
{
    return get_labeled(_obj.get_component());
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
//
// variadic versions
//
#define TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE(tag, ...)                                     \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;                       \
    TIMEMORY_BLANK_AUTO_TUPLE(_TIM_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE(tag, ...)                                     \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;                       \
    TIMEMORY_BASIC_AUTO_TUPLE(_TIM_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_TUPLE(tag, ...)                                           \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;                       \
    TIMEMORY_AUTO_TUPLE(_TIM_TYPEDEF(__LINE__), tag);

//======================================================================================//
//
//      std::get operator
//
//======================================================================================//
//
namespace std
{
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(tim::auto_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::auto_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename... Types>
auto
get(tim::auto_tuple<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::auto_tuple<Types...>>(obj).data()))
{
    using obj_type = tim::auto_tuple<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace std
//
//======================================================================================//
