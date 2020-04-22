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

#pragma once

#include "timemory/settings/declaration.hpp"
#include "timemory/variadic/auto_hybrid.hpp"
#include "timemory/variadic/auto_list.hpp"
#include "timemory/variadic/auto_tuple.hpp"
#include "timemory/variadic/types.hpp"

#if !defined(TIMEMORY_FOLD_EXPRESSION)
#    define TIMEMORY_FOLD_EXPRESSION(...)                                                \
        ::tim::consume_parameters(::std::initializer_list<int>{ (__VA_ARGS__, 0)... })
#endif

namespace tim
{
namespace func_impl
{
//--------------------------------------------------------------------------------------//

template <typename Tuple, typename... Args, size_t... Idx>
void
construct(Tuple&& obj, Args&&... args, index_sequence<Idx...>)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(
            std::get<Idx>(obj).construct(std::forward<Args>(args)...));
}

//--------------------------------------------------------------------------------------//

template <typename Tuple, typename... Args, size_t... Idx>
void
mark_begin(Tuple&& obj, Args&&... args, index_sequence<Idx...>)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(
            std::get<Idx>(obj).mark_begin(std::forward<Args>(args)...));
}

//--------------------------------------------------------------------------------------//

template <typename Tuple, typename... Args, size_t... Idx>
void
mark_end(Tuple&& obj, Args&&... args, index_sequence<Idx...>)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(
            std::get<Idx>(obj).mark_end(std::forward<Args>(args)...));
}

//--------------------------------------------------------------------------------------//

template <typename Tuple, typename... Args, size_t... Idx>
void
store(Tuple&& obj, Args&&... args, index_sequence<Idx...>)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).store(std::forward<Args>(args)...));
}

//--------------------------------------------------------------------------------------//

template <typename Tuple, typename... Args, size_t... Idx>
void
audit(Tuple&& obj, Args&&... args, index_sequence<Idx...>)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).audit(std::forward<Args>(args)...));
}
}  // namespace func_impl

//======================================================================================//

template <typename... Args>
void
start(Args&&... args)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(args.start());
}

//--------------------------------------------------------------------------------------//

template <typename... Args>
void
stop(Args&&... args)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(args.stop());
}

//--------------------------------------------------------------------------------------//

template <typename... Args>
void
print(std::ostream& os, Args&&... args)
{
    if(settings::enabled())
        TIMEMORY_FOLD_EXPRESSION(os << args << "\n");
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... Types, typename... Args>
void
construct(Tuple<Types...>&& obj, Args&&... args)
{
    constexpr auto N = sizeof...(Types);
    if(settings::enabled())
        func_impl::construct(std::forward<Tuple<Types...>>(obj),
                             std::forward<Args>(args)..., make_index_sequence<N>{});
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... Types, typename... Args>
void
mark_begin(Tuple<Types...>&& obj, Args&&... args)
{
    constexpr auto N = sizeof...(Types);
    if(settings::enabled())
        func_impl::mark_begin(std::forward<Tuple<Types...>>(obj),
                              std::forward<Args>(args)..., make_index_sequence<N>{});
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... Types, typename... Args>
void
mark_end(Tuple<Types...>&& obj, Args&&... args)
{
    constexpr auto N = sizeof...(Types);
    if(settings::enabled())
        func_impl::mark_end(std::forward<Tuple<Types...>>(obj),
                            std::forward<Args>(args)..., make_index_sequence<N>{});
}

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... Types, typename... Args>
void
store(Tuple<Types...>&& obj, Args&&... args)
{
    constexpr auto N = sizeof...(Types);
    if(settings::enabled())
        func_impl::store(std::forward<Tuple<Types...>>(obj), std::forward<Args>(args)...,
                         make_index_sequence<N>{});
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
