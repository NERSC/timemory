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

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
class auto_list
: public counted_object<auto_list<Types...>>
, public hashed_object<auto_list<Types...>>
{
public:
    using component_type   = component_list<Types...>;
    using this_type        = auto_list<Types...>;
    using data_type        = typename component_type::data_type;
    using counter_type     = counted_object<this_type>;
    using counter_void     = counted_object<void>;
    using hashed_type      = hashed_object<this_type>;
    using string_t         = std::string;
    using string_hash      = std::hash<string_t>;
    using base_type        = component_type;
    using language_t       = language;
    using tuple_type       = implemented<Types...>;
    using init_func_t      = std::function<void(this_type&)>;
    using type_tuple       = typename component_type::type_tuple;
    using data_value_tuple = typename component_type::data_value_tuple;
    using data_label_tuple = typename component_type::data_label_tuple;

    static constexpr bool contains_gotcha = component_type::contains_gotcha;

public:
    inline explicit auto_list(const string_t&, const int64_t& lineno = 0,
                              const language_t& lang = language_t::cxx(),
                              bool report_at_exit    = settings::destructor_report());
    inline explicit auto_list(component_type& tmp, const int64_t& lineno = 0,
                              bool report_at_exit = settings::destructor_report());
    inline ~auto_list();

    // copy and move
    inline auto_list(const this_type&) = default;
    inline auto_list(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

    static constexpr std::size_t available_size()
    {
        return component_type::available_size();
    }

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
            m_temporary_object.conditional_start();
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
    inline void conditional_start()
    {
        if(m_enabled)
            m_temporary_object.conditional_start();
    }
    inline void conditional_stop()
    {
        if(m_enabled)
            m_temporary_object.conditional_stop();
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

    inline bool enabled() const { return m_enabled; }
    inline void report_at_exit(bool val) { m_report_at_exit = val; }
    inline bool report_at_exit() const { return m_report_at_exit; }

    inline const bool&      store() const { return m_temporary_object.store(); }
    inline const data_type& data() const { return m_temporary_object.data(); }
    inline int64_t          laps() const { return m_temporary_object.laps(); }
    inline const int64_t&   hash() const { return m_temporary_object.hash(); }
    inline const string_t&  key() const { return m_temporary_object.key(); }
    inline const language&  lang() const { return m_temporary_object.lang(); }
    inline const string_t&  identifier() const { return m_temporary_object.identifier(); }
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
              enable_if_t<(is_one_of<_Tp, tuple_type>::value == true), int> = 0>
    void init(_Args&&... _args)
    {
        m_temporary_object.template init<_Tp>(std::forward<_Args>(_args)...);
    }

    template <typename _Tp, typename... _Args,
              enable_if_t<(is_one_of<_Tp, tuple_type>::value == false), int> = 0>
    void init(_Args&&...)
    {
    }

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

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_AUTO_LIST_INIT", "");
        };
        return _instance;
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
auto_list<Types...>::auto_list(const string_t& object_tag, const int64_t& lineno,
                               const language_t& lang, bool report_at_exit)
: counter_type()
, hashed_type((counter_type::enable())
                  ? (string_hash()(object_tag) + static_cast<int64_t>(lang) +
                     (counter_type::live() + hashed_type::live() + lineno))
                  : 0)
, m_enabled(counter_type::enable() && settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(object_tag, m_enabled, lang, counter_type::m_count,
                     hashed_type::m_hash)
{
    if(m_enabled)
    {
        get_initializer()(*this);
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename... Types>
auto_list<Types...>::auto_list(component_type& tmp, const int64_t& lineno,
                               bool report_at_exit)
: counter_type()
, hashed_type((counter_type::enable())
                  ? (string_hash()(tmp.key()) + static_cast<int64_t>(tmp.lang()) +
                     (counter_type::live() + hashed_type::live() + lineno))
                  : 0)
, m_enabled(true)
, m_report_at_exit(report_at_exit)
, m_temporary_object(tmp.clone(hashed_type::m_hash, true))
, m_reference_object(&tmp)
{
    if(m_enabled)
    {
        get_initializer()(*this);
        m_temporary_object.start();
    }
}

//======================================================================================//

template <typename... Types>
auto_list<Types...>::~auto_list()
{
    if(m_enabled)
    {
        // stop the timer
        m_temporary_object.conditional_stop();

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
          typename _Ret = typename auto_list<_Types...>::data_value_tuple>
_Ret
get(const auto_list<_Types...>& _obj)
{
    return (_obj.enabled()) ? get(_obj.get_component()) : _Ret{};
}

//--------------------------------------------------------------------------------------//

template <typename... _Types,
          typename _Ret = typename auto_list<_Types...>::data_label_tuple>
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
    using _AUTO_TYPEDEF(__LINE__) = tim::auto_list<__VA_ARGS__>;                         \
    TIMEMORY_BASIC_AUTO_LIST(_AUTO_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_LIST(tag, ...)                                            \
    using _AUTO_TYPEDEF(__LINE__) = tim::auto_list<__VA_ARGS__>;                         \
    TIMEMORY_AUTO_LIST(_AUTO_TYPEDEF(__LINE__), tag);

//======================================================================================//
