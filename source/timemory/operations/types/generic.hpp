// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/operations/types/generic.hpp
 * \brief Definition for various functions for generic in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

#include <type_traits>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::generic_operator
/// \brief This operation class is similar to pointer_operator but can handle non-pointer
/// types
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Op, typename Tag>
struct generic_operator
{
    using type       = std::remove_pointer_t<Tp>;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(generic_operator)

    template <typename Up>
    TIMEMORY_ALWAYS_INLINE static void check()
    {
        using U = std::decay_t<std::remove_pointer_t<Up>>;
        static_assert(std::is_same<U, type>::value, "Error! Up != type");
    }

    //----------------------------------------------------------------------------------//
    //
    //      Pointers
    //
    //----------------------------------------------------------------------------------//
public:
    template <typename Up, typename... Args, typename Rp = type,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<std::is_pointer<Up>::value, int>     = 0>
    TIMEMORY_ALWAYS_INLINE explicit generic_operator(Up obj, Args&&... args)
    {
        check<Up>();
        if(obj)
            sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args, typename Rp = type,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<std::is_pointer<Up>::value, int>     = 0>
    TIMEMORY_ALWAYS_INLINE explicit generic_operator(Up obj, Up rhs, Args&&... args)
    {
        check<Up>();
        if(obj && rhs)
            sfinae(obj, rhs, 0, 0, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
private:
    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto pointer_sfinae(Up obj, int, int, Args&&... args)
        -> decltype(Op(*obj, std::forward<Args>(args)...), void())
    {
        Op _tmp(*obj, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto pointer_sfinae(Up obj, int, long, Args&&...)
        -> decltype(Op{ *obj }, void())
    {
        Op _tmp{ *obj };
    }

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE void pointer_sfinae(Up, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto pointer_sfinae(Up obj, Up rhs, int, int, Args&&... args)
        -> decltype(Op(*obj, *rhs, std::forward<Args>(args)...), void())
    {
        Op(*obj, *rhs, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto pointer_sfinae(Up obj, Up rhs, int, long, Args&&...)
        -> decltype(Op(*obj, *rhs), void())
    {
        Op(*obj, *rhs);
    }

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE void pointer_sfinae(Up, Up, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //
    //      References
    //
    //----------------------------------------------------------------------------------//
public:
    template <typename Up, typename... Args, typename Rp = Tp,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<!std::is_pointer<Up>::value, int>    = 0>
    TIMEMORY_ALWAYS_INLINE explicit generic_operator(Up& obj, Args&&... args)
    {
        check<Up>();
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args, typename Rp = Tp,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<!std::is_pointer<Up>::value, int>    = 0>
    TIMEMORY_ALWAYS_INLINE explicit generic_operator(Up& obj, Up& rhs, Args&&... args)
    {
        check<Up>();
        sfinae(obj, rhs, 0, 0, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
private:
    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(Op(obj, std::forward<Args>(args)...), void())
    {
        Op _tmp(obj, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto sfinae(Up& obj, int, long, Args&&...)
        -> decltype(Op{ obj }, void())
    {
        Op _tmp{ obj };
    }

    template <typename Up, typename... Args,
              enable_if_t<std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_ALWAYS_INLINE void sfinae(Up& obj, long, long, Args&&... args)
    {
        // some operations want a raw pointer, e.g. generic_deleter
        pointer_sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<!std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_ALWAYS_INLINE void sfinae(Up&, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto sfinae(Up& obj, Up& rhs, int, int, Args&&... args)
        -> decltype(Op(obj, rhs, std::forward<Args>(args)...), void())
    {
        Op(obj, rhs, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_ALWAYS_INLINE auto sfinae(Up& obj, Up& rhs, int, long, Args&&...)
        -> decltype(Op(obj, rhs), void())
    {
        Op(obj, rhs);
    }

    template <typename Up, typename... Args,
              enable_if_t<std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_ALWAYS_INLINE void sfinae(Up& obj, Up& rhs, long, long, Args&&... args)
    {
        // some operations want a raw pointer, e.g. generic_deleter
        pointer_sfinae(obj, rhs, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<!std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_ALWAYS_INLINE void sfinae(Up&, Up&, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //
    //      Not available
    //
    //----------------------------------------------------------------------------------//
public:
    // if the type is not available, never do anything
    template <typename Up, typename... Args, typename Rp = Tp,
              enable_if_t<!trait::is_available<Rp>::value, int> = 0>
    TIMEMORY_ALWAYS_INLINE generic_operator(Up&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct generic_deleter
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(generic_deleter)

    explicit generic_deleter(type*& obj)
    {
        DEBUG_PRINT_HERE("%s %s :: %p", "deleting pointer lvalue",
                         demangle<type>().c_str(), (void*) obj);
        delete obj;
        obj = nullptr;
    }

    template <typename Up, enable_if_t<std::is_pointer<Up>::value, int> = 0>
    explicit generic_deleter(Up&& obj)
    {
        DEBUG_PRINT_HERE("%s %s :: %p", "deleting pointer rvalue",
                         demangle<type>().c_str(), (void*) obj);
        delete obj;
        std::ref(std::forward<Up>(obj)).get() = nullptr;
    }

    template <typename... Deleter>
    explicit generic_deleter(std::unique_ptr<type, Deleter...>& obj)
    {
        DEBUG_PRINT_HERE("%s %s :: %p", "deleting unique_ptr", demangle<type>().c_str(),
                         (void*) obj.get());
        obj.reset();
    }

    explicit generic_deleter(std::shared_ptr<type> obj)
    {
        DEBUG_PRINT_HERE("%s %s :: %p", "deleting shared_ptr", demangle<type>().c_str(),
                         (void*) obj.get());
        obj.reset();
    }

    template <typename Up, enable_if_t<!std::is_pointer<Up>::value, int> = 0>
    explicit generic_deleter(Up&&)
    {
        DEBUG_PRINT_HERE("%s %s", "type is not deleted", demangle<type>().c_str());
    }
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct generic_counter
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(generic_counter)

    template <typename Up, enable_if_t<std::is_pointer<Up>::value, int> = 0>
    explicit generic_counter(const Up& obj, uint64_t& count)
    {
        // static_assert(std::is_same<Up, type>::value, "Error! Up != type");
        count += (trait::runtime_enabled<type>::get() && obj) ? 1 : 0;
    }

    template <typename Up, enable_if_t<!std::is_pointer<Up>::value, int> = 0>
    explicit generic_counter(const Up&, uint64_t& count)
    {
        // static_assert(std::is_same<Up, type>::value, "Error! Up != type");
        count += (trait::runtime_enabled<type>::get()) ? 1 : 0;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
