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
#include "timemory/components/gotcha/backends.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/utility/serializer.hpp"

// this file needs to be able to see the full definition of components
#include "timemory/components.hpp"

#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

//======================================================================================//

#if !defined(SFINAE_WARNING)
#    if defined(DEBUG)
#        define SFINAE_WARNING(TYPE)                                                     \
            if(::tim::trait::is_available<TYPE>::value)                                  \
            {                                                                            \
                fprintf(stderr, "[%s@%s:%i]> Warning! SFINAE disabled for %s\n",         \
                        __FUNCTION__, __FILE__, __LINE__,                                \
                        ::tim::demangle<TYPE>().c_str());                                \
            }
#    else
#        define SFINAE_WARNING(...)
#    endif

#endif

//======================================================================================//

namespace tim
{
namespace operation
{
//--------------------------------------------------------------------------------------//

struct non_vexing
{};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct init_storage
{
    using type            = Tp;
    using value_type      = typename type::value_type;
    using base_type       = typename type::base_type;
    using string_t        = std::string;
    using storage_type    = storage<type>;
    using this_type       = init_storage<Tp>;
    using gotcha_suppress = component::gotcha_suppression;

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
            gotcha_suppress::auto_toggle suppress_lock(gotcha_suppress::get());
            auto                         main_inst = storage_type::master_instance();
            auto                         this_inst = storage_type::instance();
            bool                         this_glob = true;
            bool                         this_work = true;
            bool                         this_data = this_inst->data_init();
            return get_type{ main_inst, this_inst, this_glob, this_work, this_data };
        }();
        return _instance;
    }

    template <typename U = base_type, enable_if_t<!(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static thread_local auto _instance = []() {
            gotcha_suppress::auto_toggle suppress_lock(gotcha_suppress::get());
            auto                         main_inst = storage_type::master_instance();
            auto                         this_inst = storage_type::instance();
            return get_type{ main_inst, this_inst, false, false, false };
        }();
        return _instance;
    }

    static void init()
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        gotcha_suppress::auto_toggle suppress_lock(gotcha_suppress::get());
        static thread_local auto     _init = this_type::get();
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

    TIMEMORY_DELETED_OBJECT(construct)

    template <typename Arg, typename... Args>
    construct(type& obj, Arg&& arg, Args&&... args)
    {
        sfinae(obj, 0, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    template <typename... Args, enable_if_t<(sizeof...(Args) == 0), int> = 0>
    construct(type&, Args&&...)
    {}

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, Args&&... args)
        -> decltype(Up(std::forward<Args>(args)...), void())
    {
        obj = Up(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args
    //
    template <typename Up, typename... Args>
    auto sfinae(Up&, long, Args&&...) -> decltype(void(), void())
    {}
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

    TIMEMORY_DELETED_OBJECT(set_prefix)

    template <typename Up = Tp, enable_if_t<(trait::requires_prefix<Up>::value), int> = 0>
    set_prefix(type& obj, const string_t& prefix)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj.set_prefix(prefix);
    }

    template <typename Up                                            = Tp,
              enable_if_t<!(trait::requires_prefix<Up>::value), int> = 0>
    set_prefix(type& obj, const string_t& prefix)
    {
        sfinae(obj, 0, prefix);
    }

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a set_prefix(const string_t&) member function
    //
    template <typename U = type>
    auto sfinae(U& obj, int, const string_t& prefix)
        -> decltype(obj.set_prefix(prefix), void())
    {
        if(!trait::runtime_enabled<U>::get())
            return;

        obj.set_prefix(prefix);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a set_prefix(const string_t&) member function
    //
    template <typename U = type>
    auto sfinae(U&, long, const string_t&) -> decltype(void(), void())
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

    TIMEMORY_DELETED_OBJECT(set_flat_profile)

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

    TIMEMORY_DELETED_OBJECT(insert_node)

    //----------------------------------------------------------------------------------//
    //  has run-time optional flat storage implementation
    //
    template <typename Up = base_type, typename T = type,
              enable_if_t<!(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(Up::implements_storage_v), int>        = 0>
    explicit insert_node(base_type& obj, const uint64_t& nhash, bool flat)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        if(flat)
            obj.insert_node(scope::flat{}, nhash);
        else
            obj.insert_node(scope::tree{}, nhash);
    }

    //----------------------------------------------------------------------------------//
    //  has compile-time fixed flat storage implementation
    //
    template <typename Up = base_type, typename T = type,
              enable_if_t<(trait::flat_storage<T>::value), char> = 0,
              enable_if_t<(Up::implements_storage_v), int>       = 0>
    explicit insert_node(base_type& obj, const uint64_t& nhash, bool)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        init_storage<Tp>::init();
        obj.insert_node(scope::flat{}, nhash);
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

    TIMEMORY_DELETED_OBJECT(pop_node)

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

    TIMEMORY_DELETED_OBJECT(record)

    //----------------------------------------------------------------------------------//
    // helper
    //
    template <typename T, typename V = value_type>
    struct check_record_type
    {
        static constexpr bool value =
            (!std::is_same<V, void>::value && is_enabled<T>::value &&
             std::is_same<
                 V, typename function_traits<decltype(&T::record)>::result_type>::value);
    };

    //----------------------------------------------------------------------------------//
    // constructors
    //
    template <typename T                                       = type, typename... Args,
              enable_if_t<(check_record_type<T>::value), char> = 0>
    explicit record(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        sfinae<type, value_type>(obj, 0, 0, std::forward<Args>(args)...);
    }

    record(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        sfinae(obj, rhs, 0, 0);
    }

    template <typename T                                        = type, typename... Args,
              enable_if_t<!(check_record_type<T>::value), char> = 0>
    explicit record(base_type&, Args&&...)
    {}

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename Up, typename Vp, typename T, typename... Args,
              enable_if_t<(check_record_type<Up, Vp>::value), int> = 0>
    auto sfinae(T& obj, int, int, Args&&... args)
        -> decltype((obj.value = obj.record(std::forward<Args>(args)...)), void())
    {
        obj.value = obj.record(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename Vp, typename T, typename... Args,
              enable_if_t<(check_record_type<Up, Vp>::value), int> = 0>
    auto sfinae(T& obj, int, long, Args&&...)
        -> decltype((obj.value = obj.record()), void())
    {
        obj.value = obj.record();
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename Vp, typename T, typename... Args,
              enable_if_t<(check_record_type<Up, Vp>::value), int> = 0>
    auto sfinae(T&, long, long, Args&&...) -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename Up, typename Vp, typename T, typename... Args,
              enable_if_t<!(check_record_type<Up, Vp>::value), int> = 0>
    auto sfinae(T&, long, long, Args&&...) -> decltype(void(), void())
    {
        SFINAE_WARNING(type);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename T, enable_if_t<(trait::record_max<T>::value), int> = 0>
    auto sfinae(T& obj, const T& rhs, int, int) -> decltype(std::max(obj, rhs), void())
    {
        obj = std::max(obj, rhs);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename T, enable_if_t<!(trait::record_max<T>::value), int> = 0>
    auto sfinae(T& obj, const T& rhs, int, long) -> decltype((obj += rhs), void())
    {
        obj += rhs;
    }

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename T>
    void sfinae(T&, const T&, long, long)
    {
        SFINAE_WARNING(type);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct reset
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(reset)

    template <typename... Args>
    explicit reset(base_type& obj, Args&&... args)
    {
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.reset(std::forward<Args>(args)...), void())
    {
        obj.reset(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.reset(), void())
    {
        obj.reset();
    }

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct measure
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(measure)

    template <typename... Args>
    explicit measure(type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.measure(std::forward<Args>(args)...), void())
    {
        init_storage<Tp>::init();
        obj.measure(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.measure(), void())
    {
        init_storage<Tp>::init();
        obj.measure();
    }

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }
    //
    //----------------------------------------------------------------------------------//
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

    TIMEMORY_DELETED_OBJECT(sample)

    template <typename Up, typename... Args,
              enable_if_t<(std::is_same<Up, this_type>::value), int> = 0>
    explicit sample(type& obj, Up data, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(sfinae(obj, 0, 0, std::forward<Args>(args)...))
        {
            data.value = obj.get();
            obj.add_sample(std::move(data));
        }
    }

    template <typename Up, typename... Args,
              enable_if_t<!(std::is_same<Up, this_type>::value), int> = 0>
    explicit sample(type&, Up, Args&&...)
    {}

    data_type value;

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.sample(std::forward<Args>(args)...), bool())
    {
        init_storage<Tp>::init();
        obj.sample(std::forward<Args>(args)...);
        return true;
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.sample(), bool())
    {
        init_storage<Tp>::init();
        obj.sample();
        return true;
    }

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    bool sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
        return false;
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(start)

    template <typename... Args>
    explicit start(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        init_storage<Tp>::init();
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

    template <typename... Args>
    explicit start(base_type& obj, non_vexing&&, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        init_storage<Tp>::init();
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.start(std::forward<Args>(args)...), void())
    {
        obj.start(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.start(), void())
    {
        obj.start();
    }

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct priority_start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(priority_start)

    template <typename... Args>
    explicit priority_start(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using sfinae_type =
            conditional_t<(trait::start_priority<Tp>::value < 0), true_type, false_type>;
        sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        start<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct standard_start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(standard_start)

    template <typename... Args>
    explicit standard_start(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using sfinae_type =
            conditional_t<(trait::start_priority<Tp>::value == 0), true_type, false_type>;
        sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        start<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct delayed_start
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(delayed_start)

    template <typename... Args>
    explicit delayed_start(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using sfinae_type =
            conditional_t<(trait::start_priority<Tp>::value > 0), true_type, false_type>;
        sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        start<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(stop)

    template <typename... Args>
    explicit stop(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

    template <typename... Args>
    explicit stop(base_type& obj, non_vexing&&, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition and accepts arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.stop(std::forward<Args>(args)...), void())
    {
        obj.stop(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition but does not accept arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.stop(), void())
    {
        obj.stop();
    }

    //----------------------------------------------------------------------------------//
    //  no member function or does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct priority_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(priority_stop)

    template <typename... Args>
    explicit priority_stop(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using sfinae_type =
            conditional_t<(trait::stop_priority<Tp>::value < 0), true_type, false_type>;
        sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        stop<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct standard_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(standard_stop)

    template <typename... Args>
    explicit standard_stop(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using sfinae_type =
            conditional_t<(trait::stop_priority<Tp>::value == 0), true_type, false_type>;
        sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        stop<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct delayed_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(delayed_stop)

    template <typename... Args>
    explicit delayed_stop(base_type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using sfinae_type =
            conditional_t<(trait::stop_priority<Tp>::value > 0), true_type, false_type>;
        sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  satisfies mpl condition
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        stop<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  does not satisfy mpl condition
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
    //
    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct mark_begin
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(mark_begin)

    template <typename... Args>
    explicit mark_begin(type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.mark_begin(std::forward<Args>(args)...), void())
    {
        obj.mark_begin(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Member function is provided
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.mark_begin(), void())
    {
        obj.mark_begin();
    }

    //----------------------------------------------------------------------------------//
    //  No member function
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {}
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

    TIMEMORY_DELETED_OBJECT(mark_end)

    template <typename... Args>
    explicit mark_end(type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.mark_end(std::forward<Args>(args)...), void())
    {
        obj.mark_end(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Member function is provided
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.mark_end(), void())
    {
        obj.mark_end();
    }

    //----------------------------------------------------------------------------------//
    //  No member function
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {}
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

    TIMEMORY_DELETED_OBJECT(store)

    template <typename... Args>
    explicit store(type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.store(std::forward<Args>(args)...), void())
    {
        obj.store(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.store(), void())
    {
        obj.store();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
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

    TIMEMORY_DELETED_OBJECT(audit)

    template <typename... Args>
    audit(type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  The equivalent of supports_args and an implementation provided
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.audit(std::forward<Args>(args)...), void())
    {
        obj.audit(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename Arg, typename... Args>
    auto sfinae(Up& obj, int, long, Arg&& arg, Args&&...)
        -> decltype(obj.audit(std::forward<Arg>(arg)), void())
    {
        obj.audit(std::forward<Arg>(arg));
    }

    //----------------------------------------------------------------------------------//
    //  The equivalent of !supports_args and no implementation provided
    //
    template <typename Up, typename... Args>
    auto sfinae(Up&, long, long, Args&&...) -> decltype(void(), void())
    {}
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

    TIMEMORY_DELETED_OBJECT(compose)

    static_assert(std::is_same<ret_value_type, lhs_value_type>::value,
                  "Value types of RetType and LhsType are different!");

    static_assert(std::is_same<lhs_value_type, rhs_value_type>::value,
                  "Value types of LhsType and RhsType are different!");

    static RetType generate(const lhs_base_type& lhs, const rhs_base_type& rhs)
    {
        RetType ret;
        ret.is_running   = false;
        ret.is_on_stack  = false;
        ret.is_transient = (lhs.is_transient && rhs.is_transient);
        ret.laps         = std::min(lhs.laps, rhs.laps);
        ret.value        = (lhs.value + rhs.value);
        ret.accum        = (lhs.accum + rhs.accum);
        return ret;
    }

    template <typename Func, typename... Args>
    static RetType generate(const lhs_base_type& lhs, const rhs_base_type& rhs,
                            const Func& func, Args&&... args)
    {
        RetType ret(std::forward<Args>(args)...);
        ret.is_running   = false;
        ret.is_on_stack  = false;
        ret.is_transient = (lhs.is_transient && rhs.is_transient);
        ret.laps         = std::min(lhs.laps, rhs.laps);
        ret.value        = func(lhs.value, rhs.value);
        ret.accum        = func(lhs.accum, rhs.accum);
        return ret;
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

    TIMEMORY_DELETED_OBJECT(plus)

    template <typename Up = Tp, enable_if_t<(trait::record_max<Up>::value), int> = 0,
              enable_if_t<(has_data<Up>::value), char> = 0>
    plus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj.base_type::plus(rhs);
        obj = std::max(obj, rhs);
    }

    template <typename Up = Tp, enable_if_t<!(trait::record_max<Up>::value), int> = 0,
              enable_if_t<(has_data<Up>::value), char> = 0>
    plus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj.base_type::plus(rhs);
        obj += rhs;
    }

    template <typename Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    plus(type&, const Vt&)
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

    TIMEMORY_DELETED_OBJECT(minus)

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    minus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        // ensures update to laps
        obj.base_type::minus(rhs);
        obj -= rhs;
    }

    template <typename Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    minus(type&, const Vt&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct multiply
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(multiply)

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    multiply(type& obj, const int64_t& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj *= rhs;
    }

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    multiply(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj *= rhs;
    }

    template <typename Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    multiply(type&, const Vt&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct divide
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(divide)

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    divide(type& obj, const int64_t& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj /= rhs;
    }

    template <typename Up = Tp, enable_if_t<(has_data<Up>::value), char> = 0>
    divide(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj /= rhs;
    }

    template <typename Vt, typename Up = Tp,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    divide(type&, const Vt&)
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

    TIMEMORY_DELETED_OBJECT(get)

    //----------------------------------------------------------------------------------//
    //
    get(const type& obj, void*& ptr, size_t nhash) { get_sfinae(obj, 0, 0, ptr, nhash); }

private:
    template <typename U = type>
    auto get_sfinae(const U& obj, int, int, void*& ptr, size_t nhash)
        -> decltype(obj.get(ptr, nhash), void())
    {
        if(!ptr)
            obj.get(ptr, nhash);
    }

    template <typename U = type>
    auto get_sfinae(const U& obj, int, long, void*& ptr, size_t nhash)
        -> decltype(static_cast<const base_type&>(obj).get(ptr, nhash), void())
    {
        if(!ptr)
            static_cast<const base_type&>(obj).get(ptr, nhash);
    }

    template <typename U = type>
    void get_sfinae(const U&, long, long, void*&, size_t)
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
template <typename Tp>
struct get_data
{
    using type      = Tp;
    using data_type = decltype(std::declval<type>().get());

    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(get_data)

    //----------------------------------------------------------------------------------//
    // SFINAE
    //
    template <typename Dp, typename... Args>
    get_data(const type& obj, Dp& dst, Args&&... args)
    {
        static_assert(std::is_same<Dp, data_type>::value, "Error! Dp != type::get()");
        sfinae(obj, 0, 0, dst, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<(has_data<Up>::value), char> = 0>
    auto sfinae(const Up& obj, int, int, Dp& dst, Args&&... args)
        -> decltype(obj.get(std::forward<Args>(args)...), void())
    {
        dst = obj.get(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<(has_data<Up>::value), char> = 0>
    auto sfinae(const Up& obj, int, long, Dp& dst, Args&&...)
        -> decltype(obj.get(), void())
    {
        dst = obj.get();
    }

    //----------------------------------------------------------------------------------//
    // component is available but no "get" function
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<(has_data<Up>::value), char> = 0>
    void sfinae(const Up&, long, long, Dp&, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    // nothing if component is not available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    void sfinae(const Up&, long, long, Dp&, Args&&...)
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
template <typename Tp>
struct get_labeled_data
{
    using type      = Tp;
    using data_type = std::tuple<std::string, decltype(std::declval<type>().get())>;

    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(get_labeled_data)

    //----------------------------------------------------------------------------------//
    // SFINAE
    //
    template <typename Dp, typename... Args>
    get_labeled_data(const type& obj, Dp& dst, Args&&... args)
    {
        static_assert(std::is_same<Dp, data_type>::value,
                      "Error! Dp != tuple<string, type::get()>");
        sfinae(obj, 0, 0, dst, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<(has_data<Up>::value), char> = 0>
    auto sfinae(const Up& obj, int, int, Dp& dst, Args&&... args)
        -> decltype(obj.get(std::forward<Args>(args)...), void())
    {
        dst = data_type(type::get_label(), obj.get(std::forward<Args>(args)...));
    }

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<(has_data<Up>::value), char> = 0>
    auto sfinae(const Up& obj, int, long, Dp& dst, Args&&...)
        -> decltype(obj.get(), void())
    {
        dst = data_type(type::get_label(), obj.get());
    }

    //----------------------------------------------------------------------------------//
    // component is available but no "get" function
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<(has_data<Up>::value), char> = 0>
    void sfinae(const Up&, long, long, Dp&, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    // nothing if component is not available
    //
    template <typename Up, typename Dp, typename... Args,
              enable_if_t<!(has_data<Up>::value), char> = 0>
    void sfinae(const Up&, long, long, Dp&, Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct copy
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(copy)

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

    TIMEMORY_DELETED_OBJECT(pointer_operator)

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(Up* obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            Op(*obj, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(type* obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj)
            Op(*obj, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, base_type* rhs, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value), int> = 0>
    explicit pointer_operator(type* obj, type* rhs, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(args)...);
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

    TIMEMORY_DELETED_OBJECT(pointer_deleter)

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

    TIMEMORY_DELETED_OBJECT(pointer_counter)

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

    TIMEMORY_DELETED_OBJECT(generic_operator)

    template <typename Up>
    static void check()
    {
        using U = std::decay_t<std::remove_pointer_t<Up>>;
        static_assert(std::is_same<U, type>::value || std::is_same<U, base_type>::value,
                      "Error! Up != (type || base_type)");
    }

    //----------------------------------------------------------------------------------//
    //
    //      Pointers
    //
    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Args&&... args)
    {
        check<Up>();
        if(obj)
        {
            Op tmp(*obj, std::forward<Args>(args)...);
            consume_parameters(tmp);
        }
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Up& rhs, Args&&... args)
    {
        check<Up>();
        if(obj && rhs)
            Op(*obj, *rhs, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      References
    //
    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && !std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Args&&... args)
    {
        check<Up>();
        Op tmp(obj, std::forward<Args>(args)...);
        consume_parameters(tmp);
    }

    template <typename Up, typename... Args,
              enable_if_t<(trait::is_available<Up>::value && !std::is_pointer<Up>::value),
                          int> = 0>
    explicit generic_operator(Up& obj, Up& rhs, Args&&... args)
    {
        check<Up>();
        Op(obj, rhs, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      Not available
    //
    //----------------------------------------------------------------------------------//

    // if the type is not available, never do anything
    template <typename Up, typename... Args,
              enable_if_t<!(trait::is_available<Up>::value), int> = 0>
    generic_operator(Up&, Args&&...)
    {
        check<Up>();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct generic_deleter
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(generic_deleter)

    template <typename Up, enable_if_t<(std::is_pointer<Up>::value), int> = 0>
    explicit generic_deleter(Up& obj)
    {
        static_assert(std::is_same<Up, type>::value || std::is_same<Up, base_type>::value,
                      "Error! Up != (type || base_type)");
        delete static_cast<type*&>(obj);
    }

    template <typename Up, enable_if_t<!(std::is_pointer<Up>::value), int> = 0>
    explicit generic_deleter(Up&)
    {
        static_assert(std::is_same<Up, type>::value || std::is_same<Up, base_type>::value,
                      "Error! Up != (type || base_type)");
    }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct generic_counter
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    TIMEMORY_DELETED_OBJECT(generic_counter)

    template <typename Up, enable_if_t<(std::is_pointer<Up>::value), int> = 0>
    explicit generic_counter(const Up& obj, uint64_t& count)
    {
        static_assert(std::is_same<Up, type>::value || std::is_same<Up, base_type>::value,
                      "Error! Up != (type || base_type)");
        count += (trait::runtime_enabled<type>::get() && obj) ? 1 : 0;
    }

    template <typename Up, enable_if_t<!(std::is_pointer<Up>::value), int> = 0>
    explicit generic_counter(const Up&, uint64_t& count)
    {
        static_assert(std::is_same<Up, type>::value || std::is_same<Up, base_type>::value,
                      "Error! Up != (type || base_type)");
        count += (trait::runtime_enabled<type>::get()) ? 1 : 0;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
storage_initializer
storage_initializer::get()
{
    using storage_type = storage<T>;

    static auto _master = []() {
        auto _instance = storage_type::master_instance();
        if(_instance)
            _instance->initialize();
        return storage_initializer{};
    }();

    static thread_local auto _worker = []() {
        auto _instance = storage_type::instance();
        if(_instance)
            _instance->initialize();
        return storage_initializer{};
    }();

    consume_parameters(_master);
    return _worker;
}
//
//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

inline tim::component::cpu_clock
operator+(const tim::component::user_clock&   cuser,
          const tim::component::system_clock& csys)
{
    return tim::operation::compose<tim::component::cpu_clock, tim::component::user_clock,
                                   tim::component::system_clock>::generate(cuser, csys);
}

//--------------------------------------------------------------------------------------//

#include "timemory/mpl/bits/operations.hpp"
