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

/** \file auto_list.hpp
 * \headerfile auto_list.hpp "timemory/variadic/auto_list.hpp"
 * Automatic starting and stopping of components. Accept unlimited number of
 * parameters. The constructor starts the components, the destructor stops the
 * components
 *
 * Usage with macros (recommended):
 *    \param TIMEMORY_AUTO_LIST()
 *    \param TIMEMORY_BASIC_AUTO_LIST()
 *    \param auto t = TIMEMORY_AUTO_LIST_OBJ()
 *    \param auto t = TIMEMORY_BASIC_AUTO_LIST_OBJ()
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/component_list.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
class auto_list
{
public:
    using this_type           = auto_list<Types...>;
    using base_type           = component_list<Types...>;
    using component_type      = typename base_type::component_type;
    using data_type           = typename component_type::data_type;
    using type_tuple          = typename component_type::type_tuple;
    using data_value_type     = typename component_type::data_value_type;
    using data_label_type     = typename component_type::data_label_type;
    using init_func_t         = std::function<void(this_type&)>;
    using string_t            = std::string;
    using captured_location_t = typename component_type::captured_location_t;

    // used by component hybrid and gotcha
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = false;
    static constexpr bool is_component_hybrid = false;
    static constexpr bool contains_gotcha     = component_type::contains_gotcha;

public:
    //----------------------------------------------------------------------------------//
    //
    static void init_storage() { component_type::init_storage(); }

    //----------------------------------------------------------------------------------//
    //
    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_AUTO_LIST_INIT", "");
        };
        return _instance;
    }

public:
    template <typename _Func = init_func_t>
    inline explicit auto_list(const string_t&, bool flat = settings::flat_profile(),
                              bool         report_at_exit = false,
                              const _Func& _func          = this_type::get_initializer());

    template <typename _Func = init_func_t>
    inline explicit auto_list(const captured_location_t&,
                              bool         flat           = settings::flat_profile(),
                              bool         report_at_exit = false,
                              const _Func& _func          = this_type::get_initializer());

    template <typename _Func = init_func_t>
    inline explicit auto_list(component_type& tmp, bool flat = settings::flat_profile(),
                              bool         report_at_exit = false,
                              const _Func& _func          = this_type::get_initializer());
    inline ~auto_list();

    // copy and move
    inline auto_list(const this_type&) = default;
    inline auto_list(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    inline component_type&       get_component() { return m_temporary_object; }
    inline const component_type& get_component() const { return m_temporary_object; }

    // partial interface to underlying component_list
    inline void record()
    {
        if(m_enabled)
            m_temporary_object.record();
    }
    inline void start()
    {
        if(m_enabled)
            m_temporary_object.start();
    }
    inline void stop()
    {
        if(m_enabled)
            m_temporary_object.stop();
    }
    inline void push()
    {
        if(m_enabled)
            m_temporary_object.push();
    }
    inline void pop()
    {
        if(m_enabled)
            m_temporary_object.pop();
    }
    template <typename... _Args>
    inline void mark_begin(_Args&&... _args)
    {
        if(m_enabled)
            m_temporary_object.mark_begin(std::forward<_Args>(_args)...);
    }
    template <typename... _Args>
    inline void mark_end(_Args&&... _args)
    {
        if(m_enabled)
            m_temporary_object.mark_end(std::forward<_Args>(_args)...);
    }
    template <typename... _Args>
    inline void customize(_Args&&... _args)
    {
        if(m_enabled)
            m_temporary_object.customize(std::forward<_Args>(_args)...);
    }

    inline data_value_type get() const { return m_temporary_object.get(); }

    inline data_label_type get_labeled() const
    {
        return m_temporary_object.get_labeled();
    }

    inline bool enabled() const { return m_enabled; }
    inline void report_at_exit(bool val) { m_report_at_exit = val; }
    inline bool report_at_exit() const { return m_report_at_exit; }

    inline bool             store() const { return m_temporary_object.store(); }
    inline const data_type& data() const { return m_temporary_object.data(); }
    inline int64_t          laps() const { return m_temporary_object.laps(); }
    inline const string_t&  key() const { return m_temporary_object.key(); }
    inline void rekey(const string_t& _key) { m_temporary_object.rekey(_key); }

public:
    template <typename _Tp>
    auto get() -> decltype(std::declval<component_type>().template get<_Tp>())
    {
        return m_temporary_object.template get<_Tp>();
    }

    template <typename _Tp>
    auto get() const -> decltype(std::declval<const component_type>().template get<_Tp>())
    {
        return m_temporary_object.template get<_Tp>();
    }

    template <typename _Tp, typename... _Args,
              enable_if_t<(is_one_of<_Tp, type_tuple>::value == true), int> = 0>
    void init(_Args&&... _args)
    {
        m_temporary_object.template init<_Tp>(std::forward<_Args>(_args)...);
    }

    template <typename _Tp, typename... _Args,
              enable_if_t<(is_one_of<_Tp, type_tuple>::value == false), int> = 0>
    void init(_Args&&...)
    {}

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

public:
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << obj.m_temporary_object;
        return os;
    }

private:
    bool            m_enabled        = true;
    bool            m_report_at_exit = false;
    component_type  m_temporary_object;
    component_type* m_reference_object = nullptr;
};

//======================================================================================//

template <typename... Types>
template <typename _Func>
auto_list<Types...>::auto_list(const string_t& object_tag, bool flat, bool report_at_exit,
                               const _Func& _func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(m_enabled ? component_type(object_tag, m_enabled, flat)
                               : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        _func(*this);
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename... Types>
template <typename _Func>
auto_list<Types...>::auto_list(const captured_location_t& object_loc, bool flat,
                               bool report_at_exit, const _Func& _func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(m_enabled ? component_type(object_loc, m_enabled, flat)
                               : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        _func(*this);
        m_temporary_object.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
template <typename _Func>
auto_list<Types...>::auto_list(component_type& tmp, bool flat, bool report_at_exit,
                               const _Func& _func)
: m_enabled(true)
, m_report_at_exit(report_at_exit)
, m_temporary_object(tmp.clone(true, flat))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        _func(*this);
        m_temporary_object.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_list<Types...>::~auto_list()
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

//======================================================================================//

template <typename... _Types,
          typename _Ret = typename auto_list<_Types...>::data_value_type>
_Ret
get(const auto_list<_Types...>& _obj)
{
    return (_obj.enabled()) ? get(_obj.get_component()) : _Ret{};
}

//--------------------------------------------------------------------------------------//

template <typename... _Types,
          typename _Ret = typename auto_list<_Types...>::data_label_type>
_Ret
get_labeled(const auto_list<_Types...>& _obj)
{
    return (_obj.enabled()) ? get_labeled(_obj.get_component()) : _Ret{};
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

//--------------------------------------------------------------------------------------//
// variadic versions

#define TIMEMORY_VARIADIC_BASIC_AUTO_LIST(tag, ...)                                      \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_list<__VA_ARGS__>;                        \
    TIMEMORY_BASIC_AUTO_LIST(_TIM_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_LIST(tag, ...)                                            \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_list<__VA_ARGS__>;                        \
    TIMEMORY_AUTO_LIST(_TIM_TYPEDEF(__LINE__), tag);

//======================================================================================//
