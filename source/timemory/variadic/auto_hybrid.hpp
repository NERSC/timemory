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

/** \file auto_hybrid.hpp
 * \headerfile auto_hybrid.hpp "timemory/variadic/auto_hybrid.hpp"
 * Automatic starting and stopping of components. Accept a component_tuple as first
 * type and component_list as second type
 *
 */

#pragma once

#include <cstdint>
#include <string>

#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/component_hybrid.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename _CompTuple, typename _CompList>
class auto_hybrid
{
    static_assert(_CompTuple::is_component_tuple && _CompList::is_component_list,
                  "Error! _CompTuple must be tim::component_tuple<...> and _CompList "
                  "must be tim::component_list<...>");

public:
    using this_type           = auto_hybrid<_CompTuple, _CompList>;
    using base_type           = component_hybrid<_CompTuple, _CompList>;
    using tuple_type          = typename base_type::tuple_type;
    using list_type           = typename base_type::list_type;
    using component_type      = typename base_type::component_type;
    using data_type           = typename component_type::data_type;
    using type_tuple          = typename component_type::type_tuple;
    using tuple_type_list     = typename component_type::tuple_type_list;
    using list_type_list      = typename component_type::list_type_list;
    using data_value_type     = typename component_type::data_value_type;
    using data_label_type     = typename component_type::data_label_type;
    using string_t            = std::string;
    using captured_location_t = typename component_type::captured_location_t;

    // used by gotcha
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = false;
    static constexpr bool is_component_hybrid = false;
    static constexpr bool contains_gotcha     = component_type::contains_gotcha;

public:
    inline explicit auto_hybrid(const string_t&, bool flat = settings::flat_profile(),
                                bool report_at_exit = settings::destructor_report());

    inline explicit auto_hybrid(const captured_location_t&,
                                bool flat           = settings::flat_profile(),
                                bool report_at_exit = settings::destructor_report());

    template <typename _Scope>
    inline auto_hybrid(const string_t&, _Scope = _Scope{},
                       bool report_at_exit = settings::destructor_report());

    inline explicit auto_hybrid(component_type& tmp, bool flat = settings::flat_profile(),
                                bool report_at_exit = settings::destructor_report());
    inline ~auto_hybrid();

    // copy and move
    inline auto_hybrid(const this_type&) = default;
    inline auto_hybrid(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    inline component_type&       get_component() { return m_temporary_object; }
    inline const component_type& get_component() const { return m_temporary_object; }

    // partial interface to underlying component_hybrid
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

    inline bool            store() const { return m_temporary_object.store(); }
    inline data_type       data() const { return m_temporary_object.data(); }
    inline int64_t         laps() const { return m_temporary_object.laps(); }
    inline const string_t& key() const { return m_temporary_object.key(); }
    inline void            rekey(const string_t& _key) { m_temporary_object.rekey(_key); }

public:
    tuple_type&       get_tuple() { return m_temporary_object.get_tuple(); }
    const tuple_type& get_tuple() const { return m_temporary_object.get_tuple(); }
    list_type&        get_list() { return m_temporary_object.get_list(); }
    const list_type&  get_list() const { return m_temporary_object.get_list(); }

    tuple_type&       get_first() { return m_temporary_object.get_tuple(); }
    const tuple_type& get_first() const { return m_temporary_object.get_tuple(); }
    list_type&        get_second() { return m_temporary_object.get_list(); }
    const list_type&  get_second() const { return m_temporary_object.get_list(); }

    tuple_type&       get_lhs() { return m_temporary_object.get_tuple(); }
    const tuple_type& get_lhs() const { return m_temporary_object.get_tuple(); }
    list_type&        get_rhs() { return m_temporary_object.get_list(); }
    const list_type&  get_rhs() const { return m_temporary_object.get_list(); }

    template <typename _Tp, enable_if_t<(is_one_of<_Tp, tuple_type_list>::value ||
                                         is_one_of<_Tp, list_type_list>::value),
                                        int> = 0>
    auto get() -> decltype(std::declval<component_type>().template get<_Tp>())
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

private:
    bool            m_enabled        = true;
    bool            m_report_at_exit = false;
    component_type  m_temporary_object;
    component_type* m_reference_object = nullptr;
};

//======================================================================================//

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::auto_hybrid(const string_t& object_tag, bool flat,
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

//======================================================================================//

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::auto_hybrid(const captured_location_t& object_loc,
                                                bool flat, bool report_at_exit)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(m_enabled ? component_type(object_loc, m_enabled, flat)
                               : component_type{})
, m_reference_object(nullptr)
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename _CompTuple, typename _CompList>
template <typename _Scope>
auto_hybrid<_CompTuple, _CompList>::auto_hybrid(const string_t& object_tag, _Scope,
                                                bool            report_at_exit)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(m_enabled ? component_type(object_tag, m_enabled,
                                                std::is_same<_Scope, scope::flat>::value)
                               : component_type{})
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::auto_hybrid(component_type& tmp, bool flat,
                                                bool report_at_exit)
: m_enabled(true)
, m_report_at_exit(report_at_exit)
, m_temporary_object(tmp.clone(true, flat))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename _CompTuple, typename _CompList>
auto_hybrid<_CompTuple, _CompList>::~auto_hybrid()
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

template <typename _Tuple, typename _List,
          typename _Ret = typename auto_hybrid<_Tuple, _List>::data_value_type>
_Ret
get(const auto_hybrid<_Tuple, _List>& _obj)
{
    return (_obj.enabled()) ? get(_obj.get_component()) : _Ret{};
}

//--------------------------------------------------------------------------------------//

template <typename _Tuple, typename _List,
          typename _Ret = typename auto_hybrid<_Tuple, _List>::data_label_type>
_Ret
get_labeled(const auto_hybrid<_Tuple, _List>& _obj)
{
    return (_obj.enabled()) ? get_labeled(_obj.get_component()) : _Ret{};
}

//======================================================================================//

}  // namespace tim
