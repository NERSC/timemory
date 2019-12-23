//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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
//--------------------------------------------------------------------------------------//

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
        obj.set_prefix(_prefix);
    }

    template <typename _Up                                                    = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value == false), int> = 0>
    set_prefix(Type&, const string_t&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Scope>
struct insert_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    //  has storage implementation
    //
    template <typename _Up = base_type, enable_if_t<(_Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type& obj, const uint64_t& _hash)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);

        obj.insert_node(_Scope{}, _hash);
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename _Up                                   = base_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type&, const uint64_t&)
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
        obj.value = Type::record();
    }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char>                = 0>
    record(base_type& obj, const base_type& rhs)
    {
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

    explicit measure(base_type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.measure();
    }
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
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
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

    template <typename _Up                                          = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    explicit priority_start(base_type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.start();
    }

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit priority_start(base_type&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct standard_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                          = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    explicit standard_start(base_type&)
    {}

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit standard_start(base_type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
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
        obj.stop();
        // obj.activate_noop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct priority_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    explicit priority_stop(base_type& obj)
    {
        obj.stop();
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit priority_stop(base_type&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct standard_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    explicit standard_stop(base_type&)
    {}

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit standard_stop(base_type& obj)
    {
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

    template <typename _Up                                                       = _Tp,
              enable_if_t<(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_begin(Type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.mark_begin();
    }

    template <typename _Up                                                        = _Tp,
              enable_if_t<!(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_begin(Type&)
    {}

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                     = 0,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_begin(Type& obj, _Args&&... _args)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.mark_begin(std::forward<_Args>(_args)...);
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                      = 0,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_begin(Type&, _Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct mark_end
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                                       = _Tp,
              enable_if_t<(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_end(Type& obj)
    {
        obj.mark_end();
    }

    template <typename _Up                                                        = _Tp,
              enable_if_t<!(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_end(Type&)
    {}

    // mark_end(Type& obj) { obj.mark_end(); }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                     = 0,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_end(Type& obj, _Args&&... _args)
    {
        obj.mark_end(std::forward<_Args>(_args)...);
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                      = 0,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_end(Type&, _Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::audit
///
/// \brief The purpose of this operation class is for a component to provide some extra
/// customization within a GOTCHA function.
///
/// It will require overloading `tim::trait::supports_args`:
///   `template <> trait::supports_args<MyType, std::tuple<string, _Args...>> : true_type`
/// where `_Args...` are the GOTCHA function arguments. The string will be the function
/// name (possibly mangled). One such purpose may be to create a custom component
/// that intercepts a malloc and uses the arguments to get the exact allocation
/// size.
///
template <typename _Tp>
struct audit
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    /*
    //----------------------------------------------------------------------------------//
    //  Explicit support provided
    //
    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    audit(Type& obj, _Args&&... _args)
    {
        obj.audit(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Implicit support provided
    //
    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    audit(Type& obj, _Args&&... _args)
    {
        audit_sfinae(obj, std::forward<_Args>(_args)...);
    }
    */

    template <typename... _Args>
    audit(Type& obj, _Args&&... _args)
    {
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

    plus(Type& obj, const int64_t& rhs) { obj += rhs; }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0,
              enable_if_t<(has_data<_Up>::value), char> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj.base_type::plus(rhs);
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0,
              enable_if_t<(has_data<_Up>::value), char>                  = 0>
    plus(Type& obj, const Type& rhs)
    {
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
    minus(Type& obj, const int64_t& rhs)
    {
        obj -= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    minus(Type& obj, const Type& rhs)
    {
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
        obj *= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    multiply(Type& obj, const Type& rhs)
    {
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
        obj /= rhs;
    }

    template <typename _Up = _Tp, enable_if_t<(has_data<_Up>::value), char> = 0>
    divide(Type& obj, const Type& rhs)
    {
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
        _dst = LabeledDataType(Type::label(), _obj.get());
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
        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, _Args&&... _args)
    {
        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, base_type* rhs, _Args&&... _args)
    {
        if(obj && rhs)
            _Op(*obj, *rhs, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, Type* rhs, _Args&&... _args)
    {
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
        if(obj)
            ++count;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace operation

//--------------------------------------------------------------------------------------//

}  // namespace tim

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
