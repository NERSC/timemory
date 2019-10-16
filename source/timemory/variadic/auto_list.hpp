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

namespace filt
{
template <typename... Types>
class auto_list
{
public:
    using component_type  = ::tim::component_list<Types...>;
    using this_type       = auto_list<Types...>;
    using data_type       = typename component_type::data_type;
    using string_t        = std::string;
    using string_hash     = std::hash<string_t>;
    using base_type       = component_type;
    using tuple_type      = implemented<Types...>;
    using init_func_t     = std::function<void(this_type&)>;
    using type_tuple      = typename component_type::type_tuple;
    using data_value_type = typename component_type::data_value_type;
    using data_label_type = typename component_type::data_label_type;

    static constexpr bool contains_gotcha = component_type::contains_gotcha;

public:
    template <typename _Func>
    inline explicit auto_list(const string_t&, bool flat, bool report_at_exit,
                              _Func&& _func);
    template <typename _Func>
    inline explicit auto_list(component_type& tmp, bool flat, bool report_at_exit,
                              _Func&& _func);
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

    data_value_type inline get() const { return m_temporary_object.get(); }

    data_label_type inline get_labeled() const
    {
        return m_temporary_object.get_labeled();
    }

    inline bool enabled() const { return m_enabled; }
    inline void report_at_exit(bool val) { m_report_at_exit = val; }
    inline bool report_at_exit() const { return m_report_at_exit; }

    inline bool             store() const { return m_temporary_object.store(); }
    inline const data_type& data() const { return m_temporary_object.data(); }
    inline int64_t          laps() const { return m_temporary_object.laps(); }
    inline const int64_t&   hash() const { return m_temporary_object.hash(); }
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
                               _Func&& _func)
: m_enabled(settings::enabled())
, m_report_at_exit(report_at_exit)
, m_temporary_object(object_tag, m_enabled, flat)
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
                               _Func&& _func)
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

//--------------------------------------------------------------------------------------//
//  unused base class
//
template <typename... _Types>
struct auto_list_t
{
    using type = auto_list<_Types...>;
};

//--------------------------------------------------------------------------------------//
//  tuple overloaded base class
//
template <typename... _Types>
struct auto_list_t<std::tuple<_Types...>>
{
    using type = auto_list<_Types...>;
};

//--------------------------------------------------------------------------------------//

}  // namespace filt

//======================================================================================//
//
//                                      AUTO LIST
//
//======================================================================================//

template <typename... _Types>
class auto_list : public filt::auto_list_t<implemented<_Types...>>::type
{
public:
    using base_type      = typename filt::auto_list_t<implemented<_Types...>>::type;
    using component_type = typename base_type::component_type;
    using this_type      = auto_list<_Types...>;
    using data_type      = typename base_type::data_type;
    using string_t       = std::string;
    using type_tuple     = typename base_type::type_tuple;
    using init_func_t    = std::function<void(this_type&)>;

    static constexpr bool contains_gotcha = base_type::contains_gotcha;

public:
    template <typename _Scope = scope::process>
    inline explicit auto_list(const string_t& label,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(label, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Scope = scope::process>
    inline explicit auto_list(component_type& tmp,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(tmp, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Func>
    inline explicit auto_list(const string_t& label, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(label, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    template <typename _Func>
    inline explicit auto_list(component_type& tmp, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(tmp, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    inline ~auto_list() {}

    // copy and move
    inline auto_list(const this_type&) = default;
    inline auto_list(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_AUTO_LIST_INIT", "");
        };
        return _instance;
    }
};

//======================================================================================//

template <typename... _Types>
class auto_list<std::tuple<_Types...>>
: public filt::auto_list_t<implemented<_Types...>>::type
{
public:
    using base_type      = typename filt::auto_list_t<implemented<_Types...>>::type;
    using this_type      = auto_list<std::tuple<_Types...>>;
    using component_type = typename base_type::component_type;
    using data_type      = typename base_type::data_type;
    using string_t       = std::string;
    using type_tuple     = typename base_type::type_tuple;
    using init_func_t    = std::function<void(this_type&)>;

    static constexpr bool contains_gotcha = base_type::contains_gotcha;

public:
    template <typename _Scope = scope::process>
    inline explicit auto_list(const string_t& label,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(label, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Scope = scope::process>
    inline explicit auto_list(component_type& tmp,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(tmp, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Func>
    inline explicit auto_list(const string_t& label, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(label, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    template <typename _Func>
    inline explicit auto_list(component_type& tmp, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(tmp, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    inline ~auto_list() {}

    // copy and move
    inline auto_list(const this_type&) = default;
    inline auto_list(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_AUTO_LIST_INIT", "");
        };
        return _instance;
    }
};

//======================================================================================//

template <typename... _CompTypes, typename... _Types>
class auto_list<component_list<_CompTypes...>, _Types...>
: public filt::auto_list_t<implemented<_CompTypes..., _Types...>>::type
{
public:
    using base_type =
        typename filt::auto_list_t<implemented<_CompTypes..., _Types...>>::type;
    using this_type      = auto_list<component_list<_CompTypes...>, _Types...>;
    using component_type = typename base_type::component_type;
    using data_type      = typename base_type::data_type;
    using string_t       = std::string;
    using type_tuple     = typename base_type::type_tuple;
    using init_func_t    = std::function<void(this_type&)>;

    static constexpr bool contains_gotcha = base_type::contains_gotcha;

public:
    template <typename _Scope = scope::process>
    inline explicit auto_list(const string_t& label,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(label, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Scope = scope::process>
    inline explicit auto_list(component_type& tmp,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(tmp, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Func>
    inline explicit auto_list(const string_t& label, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(label, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    template <typename _Func>
    inline explicit auto_list(component_type& tmp, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(tmp, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    inline ~auto_list() {}

    // copy and move
    inline auto_list(const this_type&) = default;
    inline auto_list(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_AUTO_LIST_INIT", "");
        };
        return _instance;
    }
};

//======================================================================================//

template <typename... _CompTypes, typename... _Types>
class auto_list<auto_list<_CompTypes...>, _Types...>
: public filt::auto_list_t<implemented<_CompTypes..., _Types...>>::type
{
public:
    using base_type =
        typename filt::auto_list_t<implemented<_CompTypes..., _Types...>>::type;
    using this_type      = auto_list<auto_list<_CompTypes...>, _Types...>;
    using component_type = typename base_type::component_type;
    using data_type      = typename base_type::data_type;
    using string_t       = std::string;
    using type_tuple     = typename base_type::type_tuple;
    using init_func_t    = std::function<void(this_type&)>;

    static constexpr bool contains_gotcha = base_type::contains_gotcha;

public:
    template <typename _Scope = scope::process>
    inline explicit auto_list(const string_t& label,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(label, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Scope = scope::process>
    inline explicit auto_list(component_type& tmp,
                              bool            flat = (settings::flat_profile() ||
                                           std::is_same<_Scope, scope::flat>::value),
                              bool report_at_exit  = settings::destructor_report())
    : base_type(tmp, flat, report_at_exit, [](base_type& _core) {
        this_type::get_initializer()(static_cast<this_type&>(_core));
    })
    {
    }

    template <typename _Func>
    inline explicit auto_list(const string_t& label, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(label, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    template <typename _Func>
    inline explicit auto_list(component_type& tmp, bool flat, bool report_at_exit,
                              _Func&& _func)
    : base_type(tmp, flat, report_at_exit,
                [&](base_type& _core) { _func(static_cast<this_type&>(_core)); })
    {
    }

    inline ~auto_list() {}

    // copy and move
    inline auto_list(const this_type&) = default;
    inline auto_list(this_type&&)      = default;
    inline this_type& operator=(const this_type&) = default;
    inline this_type& operator=(this_type&&) = default;

    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type& al) {
            env::initialize(al, "TIMEMORY_AUTO_LIST_INIT", "");
        };
        return _instance;
    }
};

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

//--------------------------------------------------------------------------------------//

template <typename... _Types,
          typename _Ret = typename filt::auto_list<_Types...>::data_value_type>
_Ret
get(const filt::auto_list<_Types...>& _obj)
{
    return (_obj.enabled()) ? get(_obj.get_component()) : _Ret{};
}

//--------------------------------------------------------------------------------------//

template <typename... _Types,
          typename _Ret = typename filt::auto_list<_Types...>::data_label_type>
_Ret
get_labeled(const filt::auto_list<_Types...>& _obj)
{
    return (_obj.enabled()) ? get_labeled(_obj.get_component()) : _Ret{};
}

//======================================================================================//

}  // namespace tim

//======================================================================================//

//--------------------------------------------------------------------------------------//
// variadic versions

#define TIMEMORY_VARIADIC_BASIC_AUTO_LIST(tag, ...)                                      \
    using _AUTO_TYPEDEF(__LINE__) = ::tim::auto_list<__VA_ARGS__>;                       \
    TIMEMORY_BASIC_AUTO_LIST(_AUTO_TYPEDEF(__LINE__), tag);

#define TIMEMORY_VARIADIC_AUTO_LIST(tag, ...)                                            \
    using _AUTO_TYPEDEF(__LINE__) = ::tim::auto_list<__VA_ARGS__>;                       \
    TIMEMORY_AUTO_LIST(_AUTO_TYPEDEF(__LINE__), tag);

//======================================================================================//
