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

/** \file auto_tuple.hpp
 * \headerfile auto_tuple.hpp "timemory/variadic/auto_tuple.hpp"
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
    using this_type       = auto_tuple<Types...>;
    using base_type       = component_tuple<Types...>;
    using component_type  = typename base_type::component_type;
    using type_tuple      = typename component_type::type_tuple;
    using data_value_type = typename component_type::data_value_type;
    using data_label_type = typename component_type::data_label_type;
    using data_type       = typename component_type::data_type;
    using string_t        = std::string;

    // used by component hybrid and gotcha
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = false;
    static constexpr bool is_component_hybrid = false;
    static constexpr bool contains_gotcha     = component_type::contains_gotcha;

public:
    inline explicit auto_tuple(const string_t&, bool flat = settings::flat_profile(),
                               bool report_at_exit = false);
    inline explicit auto_tuple(const source_location::captured&,
                               bool flat           = settings::flat_profile(),
                               bool report_at_exit = false);
    inline explicit auto_tuple(component_type& tmp, bool flat = settings::flat_profile(),
                               bool report_at_exit = false);
    inline ~auto_tuple();

    // copy and move
    inline auto_tuple(const this_type&) = default;
    inline auto_tuple(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    inline component_type&       get_component() { return m_temporary_object; }
    inline const component_type& get_component() const { return m_temporary_object; }

    // partial interface to underlying component_tuple
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

public:
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        os << obj.m_temporary_object;
        return os;
    }

    //----------------------------------------------------------------------------------//
    static void init_storage() { component_type::init_storage(); }

protected:
    bool            m_enabled        = true;
    bool            m_report_at_exit = false;
    component_type  m_temporary_object;
    component_type* m_reference_object = nullptr;
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(const string_t& object_tag, bool flat,
                                 bool report_at_exit)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(m_enabled ? component_type(object_tag, m_enabled, flat)
                               : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(const source_location::captured& captured, bool flat,
                                 bool report_at_exit)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(m_enabled ? component_type(captured, m_enabled, flat)
                               : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_tuple<Types...>::auto_tuple(component_type& tmp, bool flat, bool report_at_exit)
: m_enabled(true)
, m_report_at_exit(report_at_exit)
, m_temporary_object(component_type(tmp.clone(true, flat)))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto_tuple<Types...>::~auto_tuple()
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
          typename _Ret = typename auto_tuple<_Types...>::data_value_type>
_Ret
get(const auto_tuple<_Types...>& _obj)
{
    return (_obj.enabled()) ? get(_obj.get_component()) : _Ret{};
}

//--------------------------------------------------------------------------------------//

template <typename... _Types,
          typename _Ret = typename auto_tuple<_Types...>::data_label_type>
_Ret
get_labeled(const auto_tuple<_Types...>& _obj)
{
    return (_obj.enabled()) ? get_labeled(_obj.get_component()) : _Ret{};
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

//--------------------------------------------------------------------------------------//
// variadic versions

#define TIMEMORY_VARIADIC_BASIC_AUTO_TUPLE(tag, ...)                                     \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;                       \
    TIMEMORY_BASIC_AUTO_TUPLE(_TIM_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_TUPLE(tag, ...)                                           \
    using _TIM_TYPEDEF(__LINE__) = ::tim::auto_tuple<__VA_ARGS__>;                       \
    TIMEMORY_AUTO_TUPLE(_TIM_TYPEDEF(__LINE__), tag);

//======================================================================================//
