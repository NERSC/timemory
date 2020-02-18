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
#include "timemory/mpl/function_traits.hpp"
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
#if !defined(TIMEMORY_OPERATION_DEFAULT)
#    define TIMEMORY_OPERATION_DEFAULT(NAME)                                             \
        NAME()            = delete;                                                      \
        NAME(const NAME&) = delete;                                                      \
        NAME(NAME&&)      = delete;                                                      \
        NAME& operator=(const NAME&) = delete;                                           \
        NAME& operator=(NAME&&) = delete;
#endif

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct init_storage
{
    using type         = Tp;
    using value_type   = typename type::value_type;
    using base_type    = typename type::base_type;
    using string_t     = std::string;
    using storage_type = storage<type>;
    using this_type    = init_storage<Tp>;

    template <typename Up = Tp, enable_if_t<(trait::is_available<Up>::value), char> = 0>
    init_storage()
    {
        static thread_local auto _instance = storage_type::instance();
        _instance->initialize();
    }

    template <typename Up = Tp, enable_if_t<!(trait::is_available<Up>::value), char> = 0>
    init_storage()
    {}

    using master_pointer_t = decltype(storage_type::master_instance());
    using pointer_t        = decltype(storage_type::instance());

    using get_type = std::tuple<master_pointer_t, pointer_t, bool, bool, bool>;

    template <typename U = base_type, enable_if_t<(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static thread_local auto _instance = []() {
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
        }();
        return _instance;
    }

    template <typename U = base_type, enable_if_t<!(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static thread_local auto _instance = []() {
            static thread_local auto _main_inst = storage_type::master_instance();
            static thread_local auto _this_inst = storage_type::instance();
            return get_type{ _main_inst, _this_inst, false, false, false };
        }();
        return _instance;
    }

    static void init()
    {
        if(!trait::runtime_enabled<type>::get())
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
template <typename Tp>
struct construct
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(construct)

    template <typename... Args, enable_if_t<(sizeof...(Args) > 0), int> = 0>
    construct(type& obj, Args&&... _args)
    {
        construct_sfinae(obj, std::forward<Args>(_args)...);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) == 0), int> = 0>
    construct(type&, Args&&...)
    {}

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto construct_sfinae_impl(Up& obj, int, Args&&... _args)
        -> decltype(Up(std::forward<Args>(_args)...), void())
    {
        obj = Up(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename... Args>
    auto construct_sfinae_impl(Up&, long, Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename Up, typename... Args>
    auto construct_sfinae(Up& obj, Args&&... _args)
        -> decltype(construct_sfinae_impl(obj, 0, std::forward<Args>(_args)...), void())
    {
        construct_sfinae_impl(obj, 0, std::forward<Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct set_prefix
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using string_t   = std::string;

    TIMEMORY_OPERATION_DEFAULT(set_prefix)

    template <typename Up = Tp, enable_if_t<(trait::requires_prefix<Up>::value), int> = 0>
    set_prefix(type& obj, const string_t& _prefix)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.set_prefix(_prefix);
    }

    template <typename Up                                            = Tp,
              enable_if_t<!(trait::requires_prefix<Up>::value), int> = 0>
    set_prefix(type& obj, const string_t& _prefix)
    {
        set_prefix_sfinae(obj, 0, _prefix);
    }

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a set_prefix(const string_t&) member function
    //
    template <typename U = type>
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
    template <typename U = type>
    auto set_prefix_sfinae(U&, long, const string_t&) -> decltype(void(), void())
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct set_flat_profile
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using string_t   = std::string;

    TIMEMORY_OPERATION_DEFAULT(set_flat_profile)

    set_flat_profile(type& obj, bool flat)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        set_flat_profile_sfinae(obj, 0, flat);
    }

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a set_flat_profile(bool) member function
    //
    template <typename T = type>
    auto set_flat_profile_sfinae(T& obj, int, bool flat)
        -> decltype(obj.set_flat_profile(flat), void())
    {
        obj.set_flat_profile(flat);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a set_flat_profile(bool) member function
    //
    template <typename T = type>
    auto set_flat_profile_sfinae(T&, long, bool) -> decltype(void(), void())
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct insert_node
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(insert_node)

    //----------------------------------------------------------------------------------//
    //  has run-time optional flat storage implementation
    //
    template <typename Up = base_type, typename T = type,
              enable_if_t<!(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(Up::implements_storage_v), int>        = 0>
    explicit insert_node(base_type& obj, const uint64_t& _hash, bool flat)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        if(flat)
            obj.insert_node(scope::flat{}, _hash);
        else
            obj.insert_node(scope::tree{}, _hash);
    }

    //----------------------------------------------------------------------------------//
    //  has compile-time fixed flat storage implementation
    //
    template <typename Up = base_type, typename T = type,
              enable_if_t<(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(Up::implements_storage_v), int>       = 0>
    explicit insert_node(base_type& obj, const uint64_t& _hash, bool)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.insert_node(scope::flat{}, _hash);
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename Up = base_type, enable_if_t<!(Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type&, const uint64_t&, bool)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct pop_node
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(pop_node)

    //----------------------------------------------------------------------------------//
    //  has storage implementation
    //
    template <typename Up = base_type, enable_if_t<(Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.pop_node();
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename Up = base_type, enable_if_t<!(Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct record
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(record)

    template <typename T = type, typename V = value_type,
              typename R = typename function_traits<decltype(&T::record)>::result_type,
              enable_if_t<(std::is_same<V, R>::value && !std::is_same<V, void>::value),
                          int> = 0>
    explicit record(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        obj.value = type::record();
    }

    template <typename T = type, typename V = value_type,
              typename R = typename function_traits<decltype(&T::record)>::result_type,
              enable_if_t<!(std::is_same<V, R>::value) || std::is_same<V, void>::value,
                          int> = 0>
    explicit record(base_type&)
    {}

    template <typename T = type, enable_if_t<(trait::record_max<T>::value), int> = 0,
              enable_if_t<(is_enabled<T>::value), char> = 0>
    record(T& obj, const T& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj = std::max(obj, rhs);
    }

    template <typename T = type, enable_if_t<!(trait::record_max<T>::value), int> = 0,
              enable_if_t<(is_enabled<T>::value), char> = 0>
    record(T& obj, const T& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj += rhs;
    }

    template <typename... Args, typename T = type,
              enable_if_t<!(is_enabled<T>::value), char> = 0>
    record(Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct reset
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(reset)

    explicit reset(base_type& obj) { obj.reset(); }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct measure
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(measure)

    explicit measure(type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.measure();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct sample
{
    static constexpr bool enable = trait::sampler<Tp>::value;
    using EmptyT                 = std::tuple<>;
    using type                   = Tp;
    using value_type             = typename type::value_type;
    using base_type              = typename type::base_type;
    using this_type              = sample<Tp>;
    using data_type = conditional_t<enable, decltype(std::declval<Tp>().get()), EmptyT>;

    sample()              = default;
    ~sample()             = default;
    sample(const sample&) = default;
    sample(sample&&)      = default;
    sample& operator=(const sample&) = default;
    sample& operator=(sample&&) = default;

    template <typename Up, enable_if_t<(std::is_same<Up, this_type>::value), int> = 0>
    explicit sample(type& obj, Up data)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.sample();
        data.value = obj.get();
        obj.add_sample(std::move(data));
    }

    template <typename Up, enable_if_t<!(std::is_same<Up, this_type>::value), int> = 0>
    explicit sample(type&, Up)
    {}

    data_type value;
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(start)

    explicit start(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct priority_start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(priority_start)

    template <typename Up                                               = Tp,
              enable_if_t<(trait::start_priority<Up>::value >= 0), int> = 0>
    explicit priority_start(base_type&)
    {}

    template <typename Up                                              = Tp,
              enable_if_t<(trait::start_priority<Up>::value < 0), int> = 0>
    explicit priority_start(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct standard_start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(standard_start)

    template <typename Up                                               = Tp,
              enable_if_t<(trait::start_priority<Up>::value != 0), int> = 0>
    explicit standard_start(base_type&)
    {}

    template <typename Up                                               = Tp,
              enable_if_t<(trait::start_priority<Up>::value == 0), int> = 0>
    explicit standard_start(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct delayed_start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(delayed_start)

    template <typename Up                                               = Tp,
              enable_if_t<(trait::start_priority<Up>::value <= 0), int> = 0>
    explicit delayed_start(base_type&)
    {}

    template <typename Up                                              = Tp,
              enable_if_t<(trait::start_priority<Up>::value > 0), int> = 0>
    explicit delayed_start(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(stop)

    explicit stop(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct priority_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(priority_stop)

    template <typename Up                                              = Tp,
              enable_if_t<(trait::stop_priority<Up>::value >= 0), int> = 0>
    explicit priority_stop(base_type&)
    {}

    template <typename Up                                             = Tp,
              enable_if_t<(trait::stop_priority<Up>::value < 0), int> = 0>
    explicit priority_stop(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct standard_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(standard_stop)

    template <typename Up                                              = Tp,
              enable_if_t<(trait::stop_priority<Up>::value != 0), int> = 0>
    explicit standard_stop(base_type&)
    {}

    template <typename Up                                              = Tp,
              enable_if_t<(trait::stop_priority<Up>::value == 0), int> = 0>
    explicit standard_stop(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct delayed_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(delayed_stop)

    template <typename Up                                              = Tp,
              enable_if_t<(trait::stop_priority<Up>::value <= 0), int> = 0>
    explicit delayed_stop(base_type&)
    {}

    template <typename Up                                             = Tp,
              enable_if_t<(trait::stop_priority<Up>::value > 0), int> = 0>
    explicit delayed_stop(base_type& obj)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct mark_begin
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(mark_begin)

    template <typename... Args>
    explicit mark_begin(type& obj, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        mark_begin_sfinae(obj, std::forward<Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto mark_begin_sfinae_impl(Up& obj, int, Args&&... _args)
        -> decltype(obj.mark_begin(std::forward<Args>(_args)...), void())
    {
        obj.mark_begin(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename... Args>
    auto mark_begin_sfinae_impl(Up&, long, Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename Up, typename... Args>
    auto mark_begin_sfinae(Up& obj, Args&&... _args)
        -> decltype(mark_begin_sfinae_impl(obj, 0, std::forward<Args>(_args)...), void())
    {
        mark_begin_sfinae_impl(obj, 0, std::forward<Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct mark_end
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(mark_end)

    template <typename... Args>
    explicit mark_end(type& obj, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        mark_end_sfinae(obj, std::forward<Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto mark_end_sfinae_impl(Up& obj, int, Args&&... _args)
        -> decltype(obj.mark_end(std::forward<Args>(_args)...), void())
    {
        obj.mark_end(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename... Args>
    auto mark_end_sfinae_impl(Up&, long, Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename Up, typename... Args>
    auto mark_end_sfinae(Up& obj, Args&&... _args)
        -> decltype(mark_end_sfinae_impl(obj, 0, std::forward<Args>(_args)...), void())
    {
        mark_end_sfinae_impl(obj, 0, std::forward<Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct store
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(store)

    template <typename... Args>
    explicit store(type& obj, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        store_sfinae(obj, 0, std::forward<Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto store_sfinae(Up& obj, int, Args&&... _args)
        -> decltype(obj.store(std::forward<Args>(_args)...), void())
    {
        obj.store(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename... Args>
    auto store_sfinae(Up&, long, Args&&...) -> decltype(void(), void())
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
template <typename Tp>
struct audit
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(audit)

    template <typename... Args>
    audit(type& obj, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        audit_sfinae(obj, std::forward<Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto audit_sfinae_impl(Up& obj, int, Args&&... _args)
        -> decltype(obj.audit(std::forward<Args>(_args)...), void())
    {
        obj.audit(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename... Args>
    auto audit_sfinae_impl(Up&, long, Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename Up, typename... Args>
    auto audit_sfinae(Up& obj, Args&&... _args)
        -> decltype(audit_sfinae_impl(obj, 0, std::forward<Args>(_args)...), void())
    {
        audit_sfinae_impl(obj, 0, std::forward<Args>(_args)...);
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

    TIMEMORY_OPERATION_DEFAULT(compose)

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

    template <typename _Func, typename... Args>
    static RetType generate(const lhs_base_type& _lhs, const rhs_base_type& _rhs,
                            const _Func& _func, Args&&... _args)
    {
        RetType _ret(std::forward<Args>(_args)...);
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
template <typename Tp>
struct plus
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(plus)

    template <typename Up = Tp, enable_if_t<(trait::record_max<Up>::value), int> = 0,
              enable_if_t<(has_data<Up>::value), char> = 0>
    plus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        obj.base_type::plus(rhs);
        obj = std::max(obj, rhs);
    }

    template <typename Up = Tp, enable_if_t<!(trait::record_max<Up>::value), int> = 0,
              enable_if_t<(has_data<Up>::value), char> = 0>
    plus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        obj.base_type::plus(rhs);
        obj += rhs;
    }

    template <typename _Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    plus(type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::minus
///
/// \brief Define subtraction operations
///
template <typename Tp>
struct minus
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(minus)

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    minus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        // ensures update to laps
        obj.base_type::minus(rhs);
        obj -= rhs;
    }

    template <typename _Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    minus(type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct multiply
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(multiply)

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    multiply(type& obj, const int64_t& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        obj *= rhs;
    }

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    multiply(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        obj *= rhs;
    }

    template <typename _Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    multiply(type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct divide
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(divide)

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    divide(type& obj, const int64_t& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        obj /= rhs;
    }

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    divide(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl_overload;
        obj /= rhs;
    }

    template <typename _Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    divide(type&, const _Vt&)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::get
///
/// \brief The purpose of this operation class is to provide a non-template hook to get
/// the object itself
///
template <typename Tp>
struct get
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(get)

    //----------------------------------------------------------------------------------//
    //
    get(const type& _obj, void*& _ptr, size_t _hash) { get_sfinae(_obj, 0, _ptr, _hash); }

private:
    template <typename U = type>
    auto get_sfinae(const U& _obj, int, void*& _ptr, size_t _hash)
        -> decltype(_obj.get(_ptr, _hash), void())
    {
        if(!_ptr)
            _obj.get(_ptr, _hash);
    }

    template <typename U = type>
    void get_sfinae(const U& _obj, long, void*& _ptr, size_t _hash)
    {
        if(!_ptr)
            static_cast<const base_type&>(_obj).get(_ptr, _hash);
    }
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::get_data
///
/// \brief The purpose of this operation class is to combine the output types from the
/// "get()" member function for multiple components -- this is specifically used in the
/// Python interface to provide direct access to the results
///
template <typename Tp>
struct get_data
{
    using type            = Tp;
    using DataType        = decltype(std::declval<type>().get());
    using LabeledDataType = std::tuple<std::string, decltype(std::declval<type>().get())>;

    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(get_data)

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    get_data(const type& _obj, DataType& _dst)
    {
        _dst = _obj.get();
    }

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    get_data(const type& _obj, LabeledDataType& _dst)
    {
        _dst = LabeledDataType(type::get_label(), _obj.get());
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename U = Tp, typename Dp, enable_if_t<!(has_data<U>::value), char> = 0>
    get_data(U&, Dp&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp, typename _Archive>
struct serialization
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(serialization)

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value), char> = 0>
    serialization(const base_type& obj, _Archive& ar, const unsigned int)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        auto _data = static_cast<const type&>(obj).get();
        ar(cereal::make_nvp("is_transient", obj.is_transient),
           cereal::make_nvp("laps", obj.laps), cereal::make_nvp("repr_data", _data),
           cereal::make_nvp("value", obj.value), cereal::make_nvp("accum", obj.accum));
    }

    template <typename Up = Tp, enable_if_t<!(is_enabled<Up>::value), char> = 0>
    serialization(const base_type&, _Archive&, const unsigned int)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct copy
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(copy)

    template <typename Up = Tp, enable_if_t<(trait::is_available<Up>::value), char> = 0>
    copy(Up& obj, const Up& rhs)
    {
        obj = Up(rhs);
    }

    template <typename Up = Tp, enable_if_t<(trait::is_available<Up>::value), char> = 0>
    copy(Up*& obj, const Up* rhs)
    {
        if(rhs)
        {
            if(!obj)
                obj = new type(*rhs);
            else
                *obj = type(*rhs);
        }
    }

    template <typename Up = Tp, enable_if_t<!(trait::is_available<Up>::value), char> = 0>
    copy(Up&, const Up&)
    {}

    template <typename Up = Tp, enable_if_t<!(trait::is_available<Up>::value), char> = 0>
    copy(Up*&, const Up*)
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
template <typename Tp, typename Op>
struct pointer_operator
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(pointer_operator)

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            Op(*obj, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(type* obj, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            Op(*obj, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, base_type* rhs, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(type* obj, type* rhs, Args&&... _args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(_args)...);
    }

    // if the type is not available, never do anything
    template <typename Up                                         = Tp, typename... Args,
              enable_if_t<!(trait::is_available<Up>::value), int> = 0>
    pointer_operator(Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct pointer_deleter
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(pointer_deleter)

    explicit pointer_deleter(type*& obj) { delete obj; }
    explicit pointer_deleter(base_type*& obj) { delete static_cast<type*&>(obj); }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct pointer_counter
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(pointer_counter)

    explicit pointer_counter(type* obj, uint64_t& count)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            ++count;
    }
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::generic_operator
///
/// \brief This operation class is similar to pointer_operator but can handle non-pointer
/// types
///
template <typename Tp, typename Op>
struct generic_operator
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(generic_operator)

    //----------------------------------------------------------------------------------//
    //
    //      Pointers
    //
    //----------------------------------------------------------------------------------//

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(base_type* obj, Args&&... _args)
    {
        if(obj)
            Op(*obj, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(type* obj, Args&&... _args)
    {
        if(obj)
            Op(*obj, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(base_type* obj, base_type* rhs, Args&&... _args)
    {
        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(type* obj, type* rhs, Args&&... _args)
    {
        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      References
    //
    //----------------------------------------------------------------------------------//

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(base_type& obj, Args&&... _args)
    {
        Op(obj, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(type& obj, Args&&... _args)
    {
        Op(obj, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(base_type& obj, base_type& rhs, Args&&... _args)
    {
        Op(obj, rhs, std::forward<Args>(_args)...);
    }

    template <typename Up                                        = Tp, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit generic_operator(type& obj, type& rhs, Args&&... _args)
    {
        Op(obj, rhs, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      Not available
    //
    //----------------------------------------------------------------------------------//

    // if the type is not available, never do anything
    template <typename Up                                         = Tp, typename... Args,
              enable_if_t<!(trait::is_available<Up>::value), int> = 0>
    generic_operator(Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct generic_deleter
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(generic_deleter)

    template <typename Up = Tp, enable_if_t<(std::is_pointer<Up>::value), int> = 0>
    explicit generic_deleter(Up& obj)
    {
        delete static_cast<type*&>(obj);
    }

    template <typename Up = Tp, enable_if_t<!(std::is_pointer<Up>::value), int> = 0>
    explicit generic_deleter(Up&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct generic_counter
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_OPERATION_DEFAULT(generic_counter)

    template <typename Up = Tp, enable_if_t<(std::is_pointer<Up>::value), int> = 0>
    explicit generic_counter(const Up& obj, uint64_t& count)
    {
        count += (trait::runtime_enabled<type>::get() && obj) ? 1 : 0;
    }

    template <typename Up = Tp, enable_if_t<!(std::is_pointer<Up>::value), int> = 0>
    explicit generic_counter(const Up&, uint64_t& count)
    {
        count += (trait::runtime_enabled<type>::get()) ? 1 : 0;
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
    extern template struct get<COMPONENT_NAME>;                                          \
    extern template struct get_data<COMPONENT_NAME>;                                     \
    extern template struct copy<COMPONENT_NAME>;                                         \
    extern template struct echo_measurement<COMPONENT_NAME,                              \
                                            trait::echo_enabled<COMPONENT_NAME>::value>; \
    extern template struct finalize::get<COMPONENT_NAME, HAS_DATA>;                      \
    extern template struct finalize::mpi_get<COMPONENT_NAME, HAS_DATA>;                  \
    extern template struct finalize::upc_get<COMPONENT_NAME, HAS_DATA>;                  \
    extern template struct finalize::dmp_get<COMPONENT_NAME, HAS_DATA>;                  \
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
    template struct get<COMPONENT_NAME>;                                                 \
    template struct get_data<COMPONENT_NAME>;                                            \
    template struct copy<COMPONENT_NAME>;                                                \
    template struct echo_measurement<COMPONENT_NAME,                                     \
                                     trait::echo_enabled<COMPONENT_NAME>::value>;        \
    template struct finalize::get<COMPONENT_NAME, HAS_DATA>;                             \
    template struct finalize::mpi_get<COMPONENT_NAME, HAS_DATA>;                         \
    template struct finalize::upc_get<COMPONENT_NAME, HAS_DATA>;                         \
    template struct finalize::dmp_get<COMPONENT_NAME, HAS_DATA>;                         \
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
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::user_mode_time, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::kernel_mode_time, true)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::current_peak_rss, true)

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
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::likwid_marker, false)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::likwid_nvmarker, false)
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

//======================================================================================//
//  gotcha
//
#    if defined(TIMEMORY_USE_GOTCHA)
TIMEMORY_DECLARE_EXTERN_OPERATIONS(component::malloc_gotcha, true)
#    endif

#endif

//--------------------------------------------------------------------------------------//
