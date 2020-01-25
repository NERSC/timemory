//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file operations.hpp
 * \headerfile operations.hpp "timemory/mpl/operations.hpp"
 * These are structs and functions that provide the operations on the
 * components
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/serializer.hpp"

// this file needs to be able to see the full definition of components
#include "timemory/components.hpp"

#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

//======================================================================================//

namespace tim
{
namespace operation
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct init_storage
{
    using Type         = _Tp;
    using value_type   = typename Type::value_type;
    using base_type    = typename Type::base_type;
    using string_t     = std::string;
    using storage_type = storage<Type>;
    using this_type    = init_storage<_Tp>;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    init_storage()
    {
        static thread_local auto _instance = storage_type::instance();
        _instance->initialize();
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    init_storage()
    {}

    using master_pointer_t = decltype(storage_type::master_instance());
    using pointer_t        = decltype(storage_type::instance());

    using get_type = std::tuple<master_pointer_t, pointer_t, bool, bool, bool>;

    template <typename U = base_type, enable_if_t<(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static auto _lambda = []() {
            static thread_local auto _main_inst = storage_type::master_instance();
            static thread_local auto _this_inst = storage_type::instance();
            if(_main_inst != _this_inst)
            {
                static bool              _main_glob = _main_inst->global_init();
                static bool              _this_glob = _this_inst->global_init();
                static thread_local bool _main_work = _main_inst->thread_init();
                static thread_local bool _this_work = _this_inst->thread_init();
                static thread_local bool _main_data = _main_inst->data_init();
                static thread_local bool _this_data = _this_inst->data_init();
                return get_type{ _main_inst, _this_inst, (_main_glob && _this_glob),
                                 (_main_work && _this_work), (_main_data && _this_data) };
            }
            else
            {
                static bool              _this_glob = _this_inst->global_init();
                static thread_local bool _this_work = _this_inst->thread_init();
                static thread_local bool _this_data = _this_inst->data_init();
                return get_type{ _main_inst, _this_inst, (_this_glob), (_this_work),
                                 (_this_data) };
            }
        };
        static thread_local auto _instance = _lambda();
        return _instance;
    }

    template <typename U = base_type, enable_if_t<!(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static auto _lambda = []() {
            static thread_local auto _main_inst = storage_type::master_instance();
            static thread_local auto _this_inst = storage_type::instance();
            return get_type{ _main_inst, _this_inst, false, false, false };
        };
        static thread_local auto _instance = _lambda();
        return _instance;
    }

    static void init()
    {
        if(!trait::is_available<_Tp>::get())
            return;

        static thread_local auto _init = this_type::get();
        consume_parameters(_init);
    }
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::construct
///
/// \brief The purpose of this operation class is construct an object with specific args
///
template <typename _Tp>
struct construct
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args>
    construct(Type& obj, _Args&&... _args)
    {
        construct_sfinae(obj, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename _Up, typename... _Args>
    auto construct_sfinae_impl(_Up& obj, int, _Args&&... _args)
        -> decltype(_Up(std::forward<_Args>(_args)...), void())
    {
        obj = _Up(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename _Up, typename... _Args>
    auto construct_sfinae_impl(_Up&, long, _Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename _Up, typename... _Args>
    auto construct_sfinae(_Up& obj, _Args&&... _args)
        -> decltype(construct_sfinae_impl(obj, 0, std::forward<_Args>(_args)...), void())
    {
        construct_sfinae_impl(obj, 0, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_prefix
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using string_t   = std::string;

    template <typename _Up                                           = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value), int> = 0>
    set_prefix(Type& obj, const string_t& _prefix)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.set_prefix(_prefix);
    }

    template <typename _Up                                                    = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value == false), int> = 0>
    set_prefix(Type&, const string_t&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct insert_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    //  has run-time optional flat storage implementation
    //
    template <typename _Up = base_type, typename T = Type,
              enable_if_t<!(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(_Up::implements_storage_v), int>       = 0>
    explicit insert_node(base_type& obj, const uint64_t& _hash, bool flat)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        if(flat)
            obj.insert_node(scope::flat{}, _hash);
        else
            obj.insert_node(scope::tree{}, _hash);
    }

    //----------------------------------------------------------------------------------//
    //  has compile-time fixed flat storage implementation
    //
    template <typename _Up = base_type, typename T = Type,
              enable_if_t<(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(_Up::implements_storage_v), int>      = 0>
    explicit insert_node(base_type& obj, const uint64_t& _hash, bool)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.insert_node(scope::flat{}, _hash);
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename _Up                                   = base_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type&, const uint64_t&, bool)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pop_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    //  has storage implementation
    //
    template <typename _Up = base_type, enable_if_t<(_Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.pop_node();
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename _Up                                   = base_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct record
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    explicit record(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.value = Type::record();
    }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char>                = 0>
    record(base_type& obj, const base_type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj += rhs;
    }

    template <typename... _Args, typename _Up = _Tp,
              enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    record(_Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct reset
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit reset(base_type& obj) { obj.reset(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct measure
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit measure(Type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.measure();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct sample
{
    static constexpr bool enable = trait::sampler<_Tp>::value;
    using EmptyT                 = std::tuple<>;
    using Type                   = _Tp;
    using value_type             = typename Type::value_type;
    using base_type              = typename Type::base_type;
    using this_type              = sample<_Tp>;
    using data_type = conditional_t<enable, decltype(std::declval<_Tp>().get()), EmptyT>;

    sample()              = default;
    ~sample()             = default;
    sample(const sample&) = default;
    sample(sample&&)      = default;
    sample& operator=(const sample&) = default;
    sample& operator=(sample&&) = default;

    template <typename _Up, enable_if_t<(std::is_same<_Up, this_type>::value), int> = 0>
    explicit sample(Type& obj, _Up data)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.sample();
        data.value = obj.get();
        obj.add_sample(std::move(data));
    }

    template <typename _Up, enable_if_t<!(std::is_same<_Up, this_type>::value), int> = 0>
    explicit sample(Type&, _Up)
    {}

    data_type value;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit start(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct priority_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value >= 0), int> = 0>
    explicit priority_start(base_type&)
    {}

    template <typename _Up                                              = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value < 0), int> = 0>
    explicit priority_start(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct standard_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value != 0), int> = 0>
    explicit standard_start(base_type&)
    {}

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == 0), int> = 0>
    explicit standard_start(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct delayed_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value <= 0), int> = 0>
    explicit delayed_start(base_type&)
    {}

    template <typename _Up                                              = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value > 0), int> = 0>
    explicit delayed_start(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        init_storage<_Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit stop(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct priority_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                              = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value >= 0), int> = 0>
    explicit priority_stop(base_type&)
    {}

    template <typename _Up                                             = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value < 0), int> = 0>
    explicit priority_stop(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct standard_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                              = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value != 0), int> = 0>
    explicit standard_stop(base_type&)
    {}

    template <typename _Up                                              = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == 0), int> = 0>
    explicit standard_stop(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct delayed_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                              = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value <= 0), int> = 0>
    explicit delayed_stop(base_type&)
    {}

    template <typename _Up                                             = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value > 0), int> = 0>
    explicit delayed_stop(base_type& obj)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct mark_begin
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args>
    explicit mark_begin(Type& obj, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        mark_begin_sfinae(obj, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename _Up, typename... _Args>
    auto mark_begin_sfinae_impl(_Up& obj, int, _Args&&... _args)
        -> decltype(obj.mark_begin(std::forward<_Args>(_args)...), void())
    {
        obj.mark_begin(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename _Up, typename... _Args>
    auto mark_begin_sfinae_impl(_Up&, long, _Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename _Up, typename... _Args>
    auto mark_begin_sfinae(_Up& obj, _Args&&... _args)
        -> decltype(mark_begin_sfinae_impl(obj, 0, std::forward<_Args>(_args)...), void())
    {
        mark_begin_sfinae_impl(obj, 0, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct mark_end
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args>
    explicit mark_end(Type& obj, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        mark_end_sfinae(obj, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename _Up, typename... _Args>
    auto mark_end_sfinae_impl(_Up& obj, int, _Args&&... _args)
        -> decltype(obj.mark_end(std::forward<_Args>(_args)...), void())
    {
        obj.mark_end(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename _Up, typename... _Args>
    auto mark_end_sfinae_impl(_Up&, long, _Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename _Up, typename... _Args>
    auto mark_end_sfinae(_Up& obj, _Args&&... _args)
        -> decltype(mark_end_sfinae_impl(obj, 0, std::forward<_Args>(_args)...), void())
    {
        mark_end_sfinae_impl(obj, 0, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::audit
///
/// \brief The purpose of this operation class is for a component to provide some extra
/// customization within a GOTCHA function. It allows a GOTCHA component to inspect
/// the arguments and the return type of a wrapped function. To add support to a
/// component, define `void audit(std::string, context, <Args...>)`. The first argument is
/// the function name (possibly mangled), the second is either type \class audit::incoming
/// or \class audit::outgoing, and the remaining arguments are the corresponding types
///
/// One such purpose may be to create a custom component that intercepts a malloc and
/// uses the arguments to get the exact allocation size.
///
template <typename _Tp>
struct audit
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args>
    audit(Type& obj, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        audit_sfinae(obj, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename _Up, typename... _Args>
    auto audit_sfinae_impl(_Up& obj, int, _Args&&... _args)
        -> decltype(obj.audit(std::forward<_Args>(_args)...), void())
    {
        obj.audit(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename _Up, typename... _Args>
    auto audit_sfinae_impl(_Up&, long, _Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename _Up, typename... _Args>
    auto audit_sfinae(_Up& obj, _Args&&... _args)
        -> decltype(audit_sfinae_impl(obj, 0, std::forward<_Args>(_args)...), void())
    {
        audit_sfinae_impl(obj, 0, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::compose
///
/// \brief The purpose of this operation class is operating on two components to compose
/// a result, e.g. use system-clock and user-clock to get a cpu-clock
///
template <typename RetType, typename LhsType, typename RhsType>
struct compose
{
    using ret_value_type = typename RetType::value_type;
    using lhs_value_type = typename LhsType::value_type;
    using rhs_value_type = typename RhsType::value_type;

    using ret_base_type = typename RetType::base_type;
    using lhs_base_type = typename LhsType::base_type;
    using rhs_base_type = typename RhsType::base_type;

    static_assert(std::is_same<ret_value_type, lhs_value_type>::value,
                  "Value types of RetType and LhsType are different!");

    static_assert(std::is_same<lhs_value_type, rhs_value_type>::value,
                  "Value types of LhsType and RhsType are different!");

    static RetType generate(const lhs_base_type& _lhs, const rhs_base_type& _rhs)
    {
        RetType _ret;
        _ret.is_running   = false;
        _ret.is_on_stack  = false;
        _ret.is_transient = (_lhs.is_transient && _rhs.is_transient);
        _ret.laps         = std::min(_lhs.laps, _rhs.laps);
        _ret.value        = (_lhs.value + _rhs.value);
        _ret.accum        = (_lhs.accum + _rhs.accum);
        return _ret;
    }

    template <typename _Func, typename... _Args>
    static RetType generate(const lhs_base_type& _lhs, const rhs_base_type& _rhs,
                            const _Func& _func, _Args&&... _args)
    {
        RetType _ret(std::forward<_Args>(_args)...);
        _ret.is_running   = false;
        _ret.is_on_stack  = false;
        _ret.is_transient = (_lhs.is_transient && _rhs.is_transient);
        _ret.laps         = std::min(_lhs.laps, _rhs.laps);
        _ret.value        = _func(_lhs.value, _rhs.value);
        _ret.accum        = _func(_lhs.accum, _rhs.accum);
        return _ret;
    }
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::plus
///
/// \brief Define addition operations
///
template <typename _Tp>
struct plus
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0,
              enable_if_t<(has_data<_Up>::value), char> = 0>
    plus(Type& obj, const Type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj.base_type::plus(rhs);
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0,
              enable_if_t<(has_data<_Up>::value), char>                  = 0>
    plus(Type& obj, const Type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj.base_type::plus(rhs);
        obj += rhs;
    }

    template <typename _Vt, typename _Up = _Tp,
              enable_if_t<!(has_data<_Up>::value), char> = 0>
    plus(Type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::minus
///
/// \brief Define subtraction operations
///
template <typename _Tp>
struct minus
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    minus(Type& obj, const Type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        // ensures update to laps
        obj.base_type::minus(rhs);
        obj -= rhs;
    }

    template <typename _Vt, typename _Up = _Tp,
              enable_if_t<!(has_data<_Up>::value), char> = 0>
    minus(Type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct multiply
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    multiply(Type& obj, const int64_t& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj *= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    multiply(Type& obj, const Type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj *= rhs;
    }

    template <typename _Vt, typename _Up = _Tp,
              enable_if_t<!(has_data<_Up>::value), char> = 0>
    multiply(Type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct divide
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    divide(Type& obj, const int64_t& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj /= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    divide(Type& obj, const Type& rhs)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj /= rhs;
    }

    template <typename _Vt, typename _Up = _Tp,
              enable_if_t<!(has_data<_Up>::value), char> = 0>
    divide(Type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::get_data
///
/// \brief The purpose of this operation class is to combine the output types from the
/// "get()" member function for multiple components -- this is specifically used in the
/// Python interface to provide direct access to the results
///
template <typename _Tp>
struct get_data
{
    using Type            = _Tp;
    using DataType        = decltype(std::declval<Type>().get());
    using LabeledDataType = std::tuple<std::string, decltype(std::declval<Type>().get())>;

    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    get_data(const Type& _obj, DataType& _dst)
    {
        _dst = _obj.get();
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    get_data(const Type&, DataType&)
    {}

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    get_data(const Type& _obj, LabeledDataType& _dst)
    {
        _dst = LabeledDataType(Type::get_label(), _obj.get());
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    get_data(const Type&, LabeledDataType&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Archive>
struct serialization
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    serialization(const base_type& obj, _Archive& ar, const unsigned int)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        auto _data = static_cast<const Type&>(obj).get();
        ar(cereal::make_nvp("is_transient", obj.is_transient),
           cereal::make_nvp("laps", obj.laps), cereal::make_nvp("repr_data", _data),
           cereal::make_nvp("value", obj.value), cereal::make_nvp("accum", obj.accum));
    }

    template <typename _Up = _Tp, enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    serialization(const base_type&, _Archive&, const unsigned int)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct copy
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    copy(_Up& obj, const _Up& rhs)
    {
        obj = _Up(rhs);
    }

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    copy(_Up*& obj, const _Up* rhs)
    {
        if(rhs)
        {
            if(!obj)
                obj = new Type(*rhs);
            else
                *obj = Type(*rhs);
        }
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    copy(_Up&, const _Up&)
    {}

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    copy(_Up*&, const _Up*)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::pointer_operator
///
/// \brief This operation class enables pointer-safety for the components created
/// on the heap (e.g. within a component_list) by ensuring other operation
/// classes are not invoked on a null pointer
///
template <typename _Tp, typename _Op>
struct pointer_operator
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    pointer_operator()                        = delete;
    pointer_operator(const pointer_operator&) = delete;
    pointer_operator(pointer_operator&&)      = delete;

    pointer_operator& operator=(const pointer_operator&) = delete;
    pointer_operator& operator=(pointer_operator&&) = delete;

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, base_type* rhs, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        if(obj && rhs)
            _Op(*obj, *rhs, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, Type* rhs, _Args&&... _args)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        if(obj && rhs)
            _Op(*obj, *rhs, std::forward<_Args>(_args)...);
    }

    // if the type is not available, never do anything
    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value == false), int> = 0>
    pointer_operator(_Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pointer_deleter
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit pointer_deleter(Type*& obj) { delete obj; }
    explicit pointer_deleter(base_type*& obj) { delete static_cast<Type*&>(obj); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pointer_counter
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit pointer_counter(Type* obj, uint64_t& count)
    {
        if(!trait::is_available<_Tp>::get())
            return;

        if(obj)
            ++count;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace operation

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/data/statistics.hpp"
#include "timemory/mpl/math.hpp"

//--------------------------------------------------------------------------------------//

inline tim::component::cpu_clock
operator+(const tim::component::user_clock&   _user,
          const tim::component::system_clock& _sys)
{
    return tim::operation::compose<tim::component::cpu_clock, tim::component::user_clock,
                                   tim::component::system_clock>::generate(_user, _sys);
}

//--------------------------------------------------------------------------------------//

#include "timemory/mpl/bits/operations.hpp"

#if defined(TIMEMORY_EXTERN_INIT)

namespace tim
{
namespace operation
{
//
//
extern template struct init_storage<component::trip_count>;
extern template struct construct<component::trip_count>;
extern template struct set_prefix<component::trip_count>;
extern template struct insert_node<component::trip_count>;
extern template struct pop_node<component::trip_count>;
extern template struct record<component::trip_count>;
extern template struct reset<component::trip_count>;
extern template struct measure<component::trip_count>;
extern template struct sample<component::trip_count>;
extern template struct start<component::trip_count>;
extern template struct priority_start<component::trip_count>;
extern template struct standard_start<component::trip_count>;
extern template struct delayed_start<component::trip_count>;
extern template struct stop<component::trip_count>;
extern template struct priority_stop<component::trip_count>;
extern template struct standard_stop<component::trip_count>;
extern template struct delayed_stop<component::trip_count>;
extern template struct mark_begin<component::trip_count>;
extern template struct mark_end<component::trip_count>;
extern template struct audit<component::trip_count>;
extern template struct plus<component::trip_count>;
extern template struct minus<component::trip_count>;
extern template struct multiply<component::trip_count>;
extern template struct divide<component::trip_count>;
extern template struct get_data<component::trip_count>;
extern template struct copy<component::trip_count>;

extern template struct init_storage<component::wall_clock>;
extern template struct construct<component::wall_clock>;
extern template struct set_prefix<component::wall_clock>;
extern template struct insert_node<component::wall_clock>;
extern template struct pop_node<component::wall_clock>;
extern template struct record<component::wall_clock>;
extern template struct reset<component::wall_clock>;
extern template struct measure<component::wall_clock>;
extern template struct sample<component::wall_clock>;
extern template struct start<component::wall_clock>;
extern template struct priority_start<component::wall_clock>;
extern template struct standard_start<component::wall_clock>;
extern template struct delayed_start<component::wall_clock>;
extern template struct stop<component::wall_clock>;
extern template struct priority_stop<component::wall_clock>;
extern template struct standard_stop<component::wall_clock>;
extern template struct delayed_stop<component::wall_clock>;
extern template struct mark_begin<component::wall_clock>;
extern template struct mark_end<component::wall_clock>;
extern template struct audit<component::wall_clock>;
extern template struct plus<component::wall_clock>;
extern template struct minus<component::wall_clock>;
extern template struct multiply<component::wall_clock>;
extern template struct divide<component::wall_clock>;
extern template struct get_data<component::wall_clock>;
extern template struct copy<component::wall_clock>;
extern template struct print_statistics<component::wall_clock>;
extern template struct print_header<component::wall_clock>;
extern template struct print<component::wall_clock>;
extern template struct print_storage<component::wall_clock>;
extern template struct echo_measurement<component::wall_clock, true>;
extern template struct finalize::storage::get<component::wall_clock, true>;
extern template struct finalize::storage::mpi_get<component::wall_clock, true>;
extern template struct finalize::storage::upc_get<component::wall_clock, true>;
extern template struct finalize::storage::dmp_get<component::wall_clock, true>;

extern template struct init_storage<component::cpu_clock>;
extern template struct construct<component::cpu_clock>;
extern template struct set_prefix<component::cpu_clock>;
extern template struct insert_node<component::cpu_clock>;
extern template struct pop_node<component::cpu_clock>;
extern template struct record<component::cpu_clock>;
extern template struct reset<component::cpu_clock>;
extern template struct measure<component::cpu_clock>;
extern template struct sample<component::cpu_clock>;
extern template struct start<component::cpu_clock>;
extern template struct priority_start<component::cpu_clock>;
extern template struct standard_start<component::cpu_clock>;
extern template struct delayed_start<component::cpu_clock>;
extern template struct stop<component::cpu_clock>;
extern template struct priority_stop<component::cpu_clock>;
extern template struct standard_stop<component::cpu_clock>;
extern template struct delayed_stop<component::cpu_clock>;
extern template struct mark_begin<component::cpu_clock>;
extern template struct mark_end<component::cpu_clock>;
extern template struct audit<component::cpu_clock>;
extern template struct plus<component::cpu_clock>;
extern template struct minus<component::cpu_clock>;
extern template struct multiply<component::cpu_clock>;
extern template struct divide<component::cpu_clock>;
extern template struct get_data<component::cpu_clock>;
extern template struct copy<component::cpu_clock>;
extern template struct print_statistics<component::cpu_clock>;
extern template struct print_header<component::cpu_clock>;
extern template struct print<component::cpu_clock>;
extern template struct print_storage<component::cpu_clock>;
extern template struct echo_measurement<component::cpu_clock, true>;
extern template struct finalize::storage::get<component::cpu_clock, true>;
extern template struct finalize::storage::mpi_get<component::cpu_clock, true>;
extern template struct finalize::storage::upc_get<component::cpu_clock, true>;
extern template struct finalize::storage::dmp_get<component::cpu_clock, true>;

extern template struct init_storage<component::read_bytes>;
extern template struct construct<component::read_bytes>;
extern template struct set_prefix<component::read_bytes>;
extern template struct insert_node<component::read_bytes>;
extern template struct pop_node<component::read_bytes>;
extern template struct record<component::read_bytes>;
extern template struct reset<component::read_bytes>;
extern template struct measure<component::read_bytes>;
extern template struct sample<component::read_bytes>;
extern template struct start<component::read_bytes>;
extern template struct priority_start<component::read_bytes>;
extern template struct standard_start<component::read_bytes>;
extern template struct delayed_start<component::read_bytes>;
extern template struct stop<component::read_bytes>;
extern template struct priority_stop<component::read_bytes>;
extern template struct standard_stop<component::read_bytes>;
extern template struct delayed_stop<component::read_bytes>;
extern template struct mark_begin<component::read_bytes>;
extern template struct mark_end<component::read_bytes>;
extern template struct audit<component::read_bytes>;
extern template struct plus<component::read_bytes>;
extern template struct minus<component::read_bytes>;
extern template struct multiply<component::read_bytes>;
extern template struct divide<component::read_bytes>;
extern template struct get_data<component::read_bytes>;
extern template struct copy<component::read_bytes>;
extern template struct print_statistics<component::read_bytes>;
extern template struct print_header<component::read_bytes>;
extern template struct print<component::read_bytes>;
extern template struct print_storage<component::read_bytes>;
extern template struct echo_measurement<component::read_bytes, true>;
extern template struct finalize::storage::get<component::read_bytes, true>;
extern template struct finalize::storage::mpi_get<component::read_bytes, true>;
extern template struct finalize::storage::upc_get<component::read_bytes, true>;
extern template struct finalize::storage::dmp_get<component::read_bytes, true>;

extern template struct init_storage<component::written_bytes>;
extern template struct construct<component::written_bytes>;
extern template struct set_prefix<component::written_bytes>;
extern template struct insert_node<component::written_bytes>;
extern template struct pop_node<component::written_bytes>;
extern template struct record<component::written_bytes>;
extern template struct reset<component::written_bytes>;
extern template struct measure<component::written_bytes>;
extern template struct sample<component::written_bytes>;
extern template struct start<component::written_bytes>;
extern template struct priority_start<component::written_bytes>;
extern template struct standard_start<component::written_bytes>;
extern template struct delayed_start<component::written_bytes>;
extern template struct stop<component::written_bytes>;
extern template struct priority_stop<component::written_bytes>;
extern template struct standard_stop<component::written_bytes>;
extern template struct delayed_stop<component::written_bytes>;
extern template struct mark_begin<component::written_bytes>;
extern template struct mark_end<component::written_bytes>;
extern template struct audit<component::written_bytes>;
extern template struct plus<component::written_bytes>;
extern template struct minus<component::written_bytes>;
extern template struct multiply<component::written_bytes>;
extern template struct divide<component::written_bytes>;
extern template struct get_data<component::written_bytes>;
extern template struct copy<component::written_bytes>;
extern template struct print_statistics<component::written_bytes>;
extern template struct print_header<component::written_bytes>;
extern template struct print<component::written_bytes>;
extern template struct print_storage<component::written_bytes>;
extern template struct echo_measurement<component::written_bytes, true>;
extern template struct finalize::storage::get<component::written_bytes, true>;
extern template struct finalize::storage::mpi_get<component::written_bytes, true>;
extern template struct finalize::storage::upc_get<component::written_bytes, true>;
extern template struct finalize::storage::dmp_get<component::written_bytes, true>;
//
//
}  // namespace operation
}  // namespace tim

#endif

//--------------------------------------------------------------------------------------//
