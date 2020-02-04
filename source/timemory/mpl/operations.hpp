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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        obj.set_prefix(_prefix);
    }

    template <typename _Up                                                    = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value == false), int> = 0>
    set_prefix(Type& obj, const string_t& _prefix)
    {
        set_prefix_sfinae(obj, 0, _prefix);
    }

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a set_prefix(const string_t&) member function
    //
    template <typename U = Type>
    auto set_prefix_sfinae(U& obj, int, const string_t& _prefix)
        -> decltype(obj.set_prefix(_prefix), void())
    {
        if(!trait::runtime_enabled<U>::get())
            return;

        obj.set_prefix(_prefix);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a set_prefix(const string_t&) member function
    //
    template <typename U = Type>
    auto set_prefix_sfinae(U&, long, const string_t&) -> decltype(void(), void())
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_flat_profile
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using string_t   = std::string;

    set_flat_profile(Type& obj, bool flat)
    {
        if(!trait::runtime_enabled<Type>::get())
            return;

        set_flat_profile_sfinae(obj, 0, flat);
    }

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a set_flat_profile(bool) member function
    //
    template <typename T = Type>
    auto set_flat_profile_sfinae(T& obj, int, bool flat)
        -> decltype(obj.set_flat_profile(flat), void())
    {
        obj.set_flat_profile(flat);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a set_flat_profile(bool) member function
    //
    template <typename T = Type>
    auto set_flat_profile_sfinae(T&, long, bool) -> decltype(void(), void())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        obj.value = Type::record();
    }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char>                = 0>
    record(base_type& obj, const base_type& rhs)
    {
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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

template <typename _Tp>
struct store
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args>
    explicit store(Type& obj, _Args&&... _args)
    {
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        store_sfinae(obj, 0, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename _Up, typename... _Args>
    auto store_sfinae(_Up& obj, int, _Args&&... _args)
        -> decltype(obj.store(std::forward<_Args>(_args)...), void())
    {
        obj.store(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename _Up, typename... _Args>
    auto store_sfinae(_Up&, long, _Args&&...) -> decltype(void(), void())
    {}
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj *= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    multiply(Type& obj, const Type& rhs)
    {
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        using namespace tim::stl_overload;
        obj /= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    divide(Type& obj, const Type& rhs)
    {
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, _Args&&... _args)
    {
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, base_type* rhs, _Args&&... _args)
    {
        if(!trait::runtime_enabled<_Tp>::get())
            return;

        if(obj && rhs)
            _Op(*obj, *rhs, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, Type* rhs, _Args&&... _args)
    {
        if(!trait::runtime_enabled<_Tp>::get())
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
        if(!trait::runtime_enabled<_Tp>::get())
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

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DECLARE_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)                     \
    namespace tim                                                                        \
    {                                                                                    \
    namespace operation                                                                  \
    {                                                                                    \
    extern template struct init_storage<COMPONENT_NAME>;                                 \
    extern template struct construct<COMPONENT_NAME>;                                    \
    extern template struct set_prefix<COMPONENT_NAME>;                                   \
    extern template struct insert_node<COMPONENT_NAME>;                                  \
    extern template struct pop_node<COMPONENT_NAME>;                                     \
    extern template struct record<COMPONENT_NAME>;                                       \
    extern template struct reset<COMPONENT_NAME>;                                        \
    extern template struct measure<COMPONENT_NAME>;                                      \
    extern template struct sample<COMPONENT_NAME>;                                       \
    extern template struct start<COMPONENT_NAME>;                                        \
    extern template struct priority_start<COMPONENT_NAME>;                               \
    extern template struct standard_start<COMPONENT_NAME>;                               \
    extern template struct delayed_start<COMPONENT_NAME>;                                \
    extern template struct stop<COMPONENT_NAME>;                                         \
    extern template struct priority_stop<COMPONENT_NAME>;                                \
    extern template struct standard_stop<COMPONENT_NAME>;                                \
    extern template struct delayed_stop<COMPONENT_NAME>;                                 \
    extern template struct mark_begin<COMPONENT_NAME>;                                   \
    extern template struct mark_end<COMPONENT_NAME>;                                     \
    extern template struct audit<COMPONENT_NAME>;                                        \
    extern template struct plus<COMPONENT_NAME>;                                         \
    extern template struct minus<COMPONENT_NAME>;                                        \
    extern template struct multiply<COMPONENT_NAME>;                                     \
    extern template struct divide<COMPONENT_NAME>;                                       \
    extern template struct get_data<COMPONENT_NAME>;                                     \
    extern template struct copy<COMPONENT_NAME>;                                         \
    extern template struct echo_measurement<COMPONENT_NAME,                              \
                                            trait::echo_enabled<COMPONENT_NAME>::value>; \
    extern template struct finalize::storage::get<COMPONENT_NAME, HAS_DATA>;             \
    extern template struct finalize::storage::mpi_get<COMPONENT_NAME, HAS_DATA>;         \
    extern template struct finalize::storage::upc_get<COMPONENT_NAME, HAS_DATA>;         \
    extern template struct finalize::storage::dmp_get<COMPONENT_NAME, HAS_DATA>;         \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)                 \
    namespace tim                                                                        \
    {                                                                                    \
    namespace operation                                                                  \
    {                                                                                    \
    template struct init_storage<COMPONENT_NAME>;                                        \
    template struct construct<COMPONENT_NAME>;                                           \
    template struct set_prefix<COMPONENT_NAME>;                                          \
    template struct insert_node<COMPONENT_NAME>;                                         \
    template struct pop_node<COMPONENT_NAME>;                                            \
    template struct record<COMPONENT_NAME>;                                              \
    template struct reset<COMPONENT_NAME>;                                               \
    template struct measure<COMPONENT_NAME>;                                             \
    template struct sample<COMPONENT_NAME>;                                              \
    template struct start<COMPONENT_NAME>;                                               \
    template struct priority_start<COMPONENT_NAME>;                                      \
    template struct standard_start<COMPONENT_NAME>;                                      \
    template struct delayed_start<COMPONENT_NAME>;                                       \
    template struct stop<COMPONENT_NAME>;                                                \
    template struct priority_stop<COMPONENT_NAME>;                                       \
    template struct standard_stop<COMPONENT_NAME>;                                       \
    template struct delayed_stop<COMPONENT_NAME>;                                        \
    template struct mark_begin<COMPONENT_NAME>;                                          \
    template struct mark_end<COMPONENT_NAME>;                                            \
    template struct audit<COMPONENT_NAME>;                                               \
    template struct plus<COMPONENT_NAME>;                                                \
    template struct minus<COMPONENT_NAME>;                                               \
    template struct multiply<COMPONENT_NAME>;                                            \
    template struct divide<COMPONENT_NAME>;                                              \
    template struct get_data<COMPONENT_NAME>;                                            \
    template struct copy<COMPONENT_NAME>;                                                \
    template struct echo_measurement<COMPONENT_NAME,                                     \
                                     trait::echo_enabled<COMPONENT_NAME>::value>;        \
    template struct finalize::storage::get<COMPONENT_NAME, HAS_DATA>;                    \
    template struct finalize::storage::mpi_get<COMPONENT_NAME, HAS_DATA>;                \
    template struct finalize::storage::upc_get<COMPONENT_NAME, HAS_DATA>;                \
    template struct finalize::storage::dmp_get<COMPONENT_NAME, HAS_DATA>;                \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_EXTERN_INIT)

//======================================================================================//
//  general
//
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::trip_count, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::gperf_cpu_profiler, false)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::gperf_heap_profiler, false)

//======================================================================================//
//  rusage
//
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::peak_rss, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::page_rss, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::stack_rss, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::data_rss, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_io_in, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_io_out, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_major_page_faults, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_minor_page_faults, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_msg_recv, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_msg_sent, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_signals, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::num_swap, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::voluntary_context_switch, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::priority_context_switch, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::read_bytes, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::written_bytes, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::virtual_memory, true)

//======================================================================================//
//  timing
//
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::wall_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::system_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::user_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cpu_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::monotonic_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::monotonic_raw_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::thread_cpu_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::process_cpu_clock, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cpu_util, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::process_cpu_util, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::thread_cpu_util, true)

TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::user_tuple_bundle, false)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::user_list_bundle, false)

//======================================================================================//
//  caliper
//
#    if defined(TIMEMORY_USE_CALIPER)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::caliper, false)
#    endif

//======================================================================================//
//  papi
//
#    if defined(TIMEMORY_USE_PAPI)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::papi_array<8>, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::papi_array<16>, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::papi_array<32>, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cpu_roofline_flops, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cpu_roofline_sp_flops, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cpu_roofline_dp_flops, true)
#    endif

//======================================================================================//
//  cuda
//
#    if defined(TIMEMORY_USE_CUDA)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cuda_event, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cuda_profiler, false)
#    endif

//======================================================================================//
//  NVTX
//
#    if defined(TIMEMORY_USE_NVTX)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::nvtx_marker, false)
#    endif

//======================================================================================//
//  cupti
//
#    if defined(TIMEMORY_USE_CUPTI)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cupti_activity, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::cupti_counters, true)
// TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::gpu_roofline_flops, true)
// TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::gpu_roofline_hp_flops, true)
// TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::gpu_roofline_sp_flops, true)
// TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::gpu_roofline_dp_flops, true)
#    endif

//======================================================================================//
//  likwid
//
#    if defined(TIMEMORY_USE_LIKWID)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::likwid_perfmon, false)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::likwid_nvmon, false)
#    endif

//======================================================================================//
//  tau
//
#    if defined(TIMEMORY_USE_TAU)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::tau_marker, false)
#    endif

//======================================================================================//
//  vtune
//
#    if defined(TIMEMORY_USE_VTUNE)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::vtune_profiler, false)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::vtune_event, false)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::vtune_frame, false)
#    endif

#endif

//--------------------------------------------------------------------------------------//
