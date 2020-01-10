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

/** \file mpl/types.hpp
 * \headerfile mpl/types.hpp "timemory/mpl/types.hpp"
 *
 * This is a pre-declaration of all the operation structs.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers.
 *
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include "timemory/utility/types.hpp"

//======================================================================================//
//
#if !defined(TIMEMORY_DEFAULT_STATISTICS_TYPE)
#    if defined(TIMEMORY_USE_STATISTICS)
#        define TIMEMORY_DEFAULT_STATISTICS_TYPE std::true_type
#    else
#        define TIMEMORY_DEFAULT_STATISTICS_TYPE std::false_type
#    endif
#endif

//======================================================================================//
//
namespace tim
{
template <int N>
using priority_constant = std::integral_constant<int, N>;

using true_type                      = std::true_type;
using false_type                     = std::false_type;
using default_record_statistics_type = TIMEMORY_DEFAULT_STATISTICS_TYPE;

//======================================================================================//
// type-traits for customization
//
namespace trait
{
template <typename _Tp>
struct is_available;

template <typename _Tp>
struct record_max;

template <typename _Tp>
struct array_serialization;

template <typename _Tp>
struct requires_prefix;

template <typename _Tp>
struct custom_label_printing;

template <typename _Tp>
struct custom_unit_printing;

template <typename _Tp>
struct custom_laps_printing;

template <typename _Tp>
struct start_priority;

template <typename _Tp>
struct stop_priority;

template <typename _Tp>
struct is_timing_category;

template <typename _Tp>
struct is_memory_category;

template <typename _Tp>
struct uses_timing_units;

template <typename _Tp>
struct uses_memory_units;

template <typename _Tp>
struct requires_json;

template <typename _Tp>
struct is_gotcha;

template <typename _Tp, typename _Tuple>
struct supports_args;

template <typename _Tp>
struct supports_custom_record;

template <typename _Tp>
struct iterable_measurement;

template <typename _Tp>
struct secondary_data;

template <typename _Tp>
struct thread_scope_only;

template <typename _Tp>
struct split_serialization;

template <typename _Tp>
struct record_statistics;

template <typename _Tp>
struct statistics;

template <typename _Tp>
struct sampler;

template <typename _Tp>
struct file_sampler;

template <typename _Tp>
struct units;

template <typename _Tp>
struct echo_enabled;

template <typename _Tp>
struct input_archive;

template <typename _Tp>
struct output_archive;

}  // namespace trait

//======================================================================================//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace operation
{
// operators
template <typename _Tp>
struct init_storage;

template <typename _Tp>
struct construct;

template <typename _Tp>
struct set_prefix;

template <typename _Tp, typename _Scope>
struct insert_node;

template <typename _Tp>
struct pop_node;

template <typename _Tp>
struct record;

template <typename _Tp>
struct reset;

template <typename _Tp>
struct measure;

template <typename _Tp>
struct sample;

template <typename _Ret, typename _Lhs, typename _Rhs>
struct compose;

template <typename _Tp>
struct start;

template <typename _Tp>
struct priority_start;

template <typename _Tp>
struct standard_start;

template <typename _Tp>
struct delayed_start;

template <typename _Tp>
struct stop;

template <typename _Tp>
struct priority_stop;

template <typename _Tp>
struct standard_stop;

template <typename _Tp>
struct delayed_stop;

template <typename _Tp>
struct mark_begin;

template <typename _Tp>
struct mark_end;

template <typename _Tp>
struct audit;

template <typename RetType, typename LhsType, typename RhsType>
struct compose;

template <typename _Tp>
struct plus;

template <typename _Tp>
struct minus;

template <typename _Tp>
struct multiply;

template <typename _Tp>
struct divide;

template <typename _Tp>
struct get_data;

template <typename _Tp>
struct base_printer;

template <typename _Tp>
struct print;

template <typename _Tp>
struct print_header;

template <typename _Tp>
struct print_statistics;

template <typename _Tp>
struct print_storage;

template <typename _Tp, typename _Archive>
struct serialization;

// template <typename _Tp>
// struct echo_measurement;
template <typename _Tp, bool _Enabled = trait::echo_enabled<_Tp>::value>
struct echo_measurement;

template <typename _Tp>
struct copy;

template <typename _Tp, typename _Op>
struct pointer_operator;

template <typename _Tp>
struct pointer_deleter;

template <typename _Tp>
struct pointer_counter;

}  // namespace operation

//======================================================================================//
// generic helpers that can/should be inherited from
//
namespace policy
{
template <typename _Tp, bool _WithThreads = true>
struct instance_tracker;

template <typename _Comp, typename _Tp = typename trait::statistics<_Comp>::type>
struct record_statistics;

}  // namespace policy

//--------------------------------------------------------------------------------------//

namespace operation
{
//----------------------------------------------------------------------------------//
// shorthand for available, non-void, using internal output handling
//
template <typename _Up>
struct is_enabled
{
    using _Vp = typename _Up::value_type;
    static constexpr bool value =
        (trait::is_available<_Up>::value && !(std::is_same<_Vp, void>::value));
};

//----------------------------------------------------------------------------------//
// shorthand for non-void, using internal output handling
//
template <typename _Up>
struct has_data
{
    using _Vp                   = typename _Up::value_type;
    static constexpr bool value = (!(std::is_same<_Vp, void>::value));
};

}  // namespace operation

//======================================================================================//
//
///     \class type_list
///     \brief lightweight tuple-alternative for meta-programming logic
//
//======================================================================================//

template <typename... _Tp>
struct type_list
{};

//======================================================================================//
//
//  Pre-C++11 tuple expansion
//
//======================================================================================//

// for pre-C++14 tuple expansion to arguments
namespace impl
{
//--------------------------------------------------------------------------------------//
// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
template <size_t... _Indexes>
struct Index_tuple
{};

//--------------------------------------------------------------------------------------//
// Concatenates two Index_tuples.
template <typename _Itup1, typename _Itup2>
struct Itup_cat;

//--------------------------------------------------------------------------------------//

template <size_t... _Ind1, size_t... _Ind2>
struct Itup_cat<Index_tuple<_Ind1...>, Index_tuple<_Ind2...>>
{
    using __type = Index_tuple<_Ind1..., (_Ind2 + sizeof...(_Ind1))...>;
};

//--------------------------------------------------------------------------------------//
// Builds an Index_tuple<0, 1, 2, ..., _Num-1>.
template <size_t _Num, size_t _Off = 0>
struct Build_index_tuple
: Itup_cat<typename Build_index_tuple<_Num / 2, _Off>::__type,
           typename Build_index_tuple<_Num - _Num / 2, _Off>::__type>
{};

//--------------------------------------------------------------------------------------//

template <size_t _Off>
struct Build_index_tuple<1, _Off>
{
    using __type = Index_tuple<0 + _Off>;
};

//--------------------------------------------------------------------------------------//

template <size_t _Off>
struct Build_index_tuple<0, _Off>
{
    using __type = Index_tuple<>;
};

//--------------------------------------------------------------------------------------//
/// Class template integer_sequence
template <typename _Tp, _Tp... _Idx>
struct integer_sequence
{
    using value_type = _Tp;
    static constexpr size_t size() noexcept { return sizeof...(_Idx); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, _Tp _Num,
          typename _ISeq = typename Build_index_tuple<_Num>::__type>
struct Make_integer_sequence;

//--------------------------------------------------------------------------------------//

template <typename _Tp, _Tp _Num, size_t... _Idx>
struct Make_integer_sequence<_Tp, _Num, Index_tuple<_Idx...>>
{
    static_assert(_Num >= 0, "Cannot make integer sequence of negative length");
    using __type = integer_sequence<_Tp, static_cast<_Tp>(_Idx)...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
struct tuple_concat
{
    using type = std::tuple<Types...>;
};

//--------------------------------------------------------------------------------------//

template <>
struct tuple_concat<>
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts>
struct tuple_concat<std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts0, typename... Ts1, typename... Rest>
struct tuple_concat<std::tuple<Ts0...>, std::tuple<Ts1...>, Rest...>
: tuple_concat<std::tuple<Ts0..., Ts1...>, Rest...>
{};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//======================================================================================//

/// Alias template make_integer_sequence
template <typename _Tp, _Tp _Num>
using make_integer_sequence = typename impl::Make_integer_sequence<_Tp, _Num>::__type;

/// Alias template index_sequence
template <size_t... _Idx>
using index_sequence = impl::integer_sequence<size_t, _Idx...>;

/// Alias template make_index_sequence
template <size_t _Num>
using make_index_sequence = make_integer_sequence<size_t, _Num>;

/// Alias template index_sequence_for
template <typename... _Types>
using index_sequence_for = make_index_sequence<sizeof...(_Types)>;

/// Alias template for enable_if
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <typename T>
using decay_t = typename std::decay<T>::type;

template <bool _Val, typename _Lhs, typename _Rhs>
using conditional_t = typename std::conditional<_Val, _Lhs, _Rhs>::type;

template <typename... Ts>
using tuple_concat_t = typename impl::tuple_concat<Ts...>::type;

//======================================================================================//

///
/// get the index of a type in expansion
///
template <typename _Tp, typename Type>
struct index_of;

template <typename _Tp, template <typename...> class _Tuple, typename... Types>
struct index_of<_Tp, _Tuple<_Tp, Types...>>
{
    static constexpr size_t value = 0;
};

template <typename _Tp, typename Head, template <typename...> class _Tuple,
          typename... Tail>
struct index_of<_Tp, _Tuple<Head, Tail...>>
{
    static constexpr size_t value = 1 + index_of<_Tp, _Tuple<Tail...>>::value;
};

//======================================================================================//

namespace impl
{
template <typename In, typename Out>
struct convert
{
    using type = Out;

    using input_type  = In;
    using output_type = Out;

    static output_type apply(const input_type& _in)
    {
        return static_cast<output_type>(_in);
    }
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class InTuple, typename... In,
          template <typename...> class OutTuple, typename... Out>
struct convert<InTuple<In...>, OutTuple<Out...>>
{
    using type = OutTuple<In...>;

    using input_type     = InTuple<In...>;
    using output_type    = OutTuple<Out...>;
    using init_list_type = std::initializer_list<int>;

    static output_type apply(const input_type& _in)
    {
        output_type _out{};
        auto&&      ret = init_list_type{ (
            std::get<index_of<Out, output_type>::value>(_out) =
                static_cast<Out>(std::get<index_of<In, input_type>::value>(_in)),
            0)... };
        consume_parameters(ret);
        return _out;
    }
};

}  // namespace impl

//======================================================================================//

template <typename T, typename U>
using convert_t = typename impl::convert<T, U>::type;

//======================================================================================//

namespace mpl
{
template <typename _Out, typename _In>
auto
convert(const _In& _in) -> decltype(impl::convert<_In, _Out>::apply(_in))
{
    return impl::convert<_In, _Out>::apply(_in);
}

}  // namespace mpl

//======================================================================================//

}  // namespace tim

//======================================================================================//

#define TIMEMORY_STATISTICS_TYPE(COMPONENT, TYPE)                                        \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct statistics<COMPONENT>                                                         \
    {                                                                                    \
        using type = TYPE;                                                               \
    };                                                                                   \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_TEMPLATE_STATISTICS_TYPE(COMPONENT, TYPE, TEMPLATE_TYPE)                \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <TEMPLATE_TYPE T>                                                           \
    struct statistics<COMPONENT<T>>                                                      \
    {                                                                                    \
        using type = TYPE;                                                               \
    };                                                                                   \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_VARIADIC_STATISTICS_TYPE(COMPONENT, TYPE, TEMPLATE_TYPE)                \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <TEMPLATE_TYPE... T>                                                        \
    struct statistics<COMPONENT<T...>>                                                   \
    {                                                                                    \
        using type = TYPE;                                                               \
    };                                                                                   \
    }                                                                                    \
    }

//======================================================================================//

#define TIMEMORY_DEFINE_CONCRETE_TRAIT(TRAIT, COMPONENT, VALUE)                          \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct TRAIT<COMPONENT> : VALUE                                                      \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_TEMPLATE_TRAIT(TRAIT, COMPONENT, VALUE, TYPE)                    \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <TYPE T>                                                                    \
    struct TRAIT<COMPONENT<T>> : VALUE                                                   \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_VARIADIC_TRAIT(TRAIT, COMPONENT, VALUE, TYPE)                    \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <TYPE... T>                                                                 \
    struct TRAIT<COMPONENT<T...>> : VALUE                                                \
    {};                                                                                  \
    }                                                                                    \
    }

//======================================================================================//
