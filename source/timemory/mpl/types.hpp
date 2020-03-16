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
 * This is a declaration of all the operation structs.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers.
 *
 */

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "timemory/api.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/types.hpp"

//======================================================================================//
//
#if !defined(TIMEMORY_FOLD_EXPRESSION)
#    define TIMEMORY_FOLD_EXPRESSION(...)                                                \
        ::tim::consume_parameters(::std::initializer_list<int>{ (__VA_ARGS__, 0)... })
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DELETED_OBJECT)
#    define TIMEMORY_DELETED_OBJECT(NAME)                                                \
        NAME()            = delete;                                                      \
        NAME(const NAME&) = delete;                                                      \
        NAME(NAME&&)      = delete;                                                      \
        NAME& operator=(const NAME&) = delete;                                           \
        NAME& operator=(NAME&&) = delete;
#endif

//======================================================================================//
//
#if !defined(TIMEMORY_DEFAULT_OBJECT)
#    define TIMEMORY_DEFAULT_OBJECT(NAME)                                                \
        NAME()            = default;                                                     \
        NAME(const NAME&) = default;                                                     \
        NAME(NAME&&)      = default;                                                     \
        NAME& operator=(const NAME&) = default;                                          \
        NAME& operator=(NAME&&) = default;
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

///
/// \class storage_initializer
///
/// \brief This provides an object that can initialize the storage opaquely, e.g.
/// \code
/// namespace
/// {
///     tim::storage_initializer storage = tim::storage_initalizer::get<T>();
/// }
///
struct storage_initializer
{
    TIMEMORY_DEFAULT_OBJECT(storage_initializer)

    template <typename T>
    static storage_initializer get();
};

//======================================================================================//
// type-traits for customization
//
namespace trait
{
template <typename T>
struct base_has_accum;

template <typename T>
struct base_has_last;

template <typename T>
struct is_available;

template <typename T>
struct data;

template <typename T>
struct runtime_enabled;

template <typename T>
struct record_max;

template <typename T>
struct array_serialization;

template <typename T>
struct requires_prefix;

template <typename T>
struct custom_label_printing;

template <typename T>
struct custom_unit_printing;

template <typename T>
struct custom_laps_printing;

template <typename T>
struct start_priority;

template <typename T>
struct stop_priority;

template <typename T>
struct is_timing_category;

template <typename T>
struct is_memory_category;

template <typename T>
struct uses_timing_units;

template <typename T>
struct uses_memory_units;

template <typename T>
struct requires_json;

template <typename T>
struct is_gotcha;

template <typename T>
struct is_user_bundle;

template <typename T>
struct collects_data;

template <typename T, typename Tuple>
struct supports_args;

template <typename T>
struct supports_custom_record;

template <typename T>
struct iterable_measurement;

template <typename T>
struct secondary_data;

template <typename T>
struct thread_scope_only;

template <typename T>
struct custom_serialization;

template <typename T>
struct record_statistics;

template <typename T>
struct statistics;

template <typename T>
struct permissive_statistics;

template <typename T>
struct sampler;

template <typename T>
struct file_sampler;

template <typename T>
struct units;

template <typename T>
struct echo_enabled;

template <typename Api = api::native_tag>
struct api_input_archive;

template <typename Api = api::native_tag>
struct api_output_archive;

template <typename T, typename Api = api::native_tag>
struct input_archive;

template <typename T, typename Api = api::native_tag>
struct output_archive;

template <typename T>
struct pretty_json;

template <typename T>
struct flat_storage;

template <typename T>
struct report_sum;

template <typename T>
struct report_mean;

template <typename T>
struct report_values;

template <typename T>
struct omp_tools;

//--------------------------------------------------------------------------------------//
//
//                              ALIASES
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
using input_archive_t = typename input_archive<T, TIMEMORY_API>::type;

template <typename T>
using output_archive_t = typename output_archive<T, TIMEMORY_API>::type;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait

template <typename T, typename V = typename trait::data<T>::value_type>
struct generates_output;

template <typename T, typename V = typename trait::data<T>::value_type>
struct implements_storage;

//======================================================================================//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace operation
{
// operators
template <typename T>
struct init_storage;

template <typename T>
struct construct;

template <typename T>
struct set_prefix;

template <typename T>
struct set_flat_profile;

template <typename T>
struct set_timeline_profile;

template <typename T>
struct insert_node;

template <typename T>
struct pop_node;

template <typename T>
struct record;

template <typename T>
struct reset;

template <typename T>
struct measure;

template <typename T>
struct sample;

template <typename Ret, typename Lhs, typename Rhs>
struct compose;

template <typename T>
struct start;

template <typename T>
struct priority_start;

template <typename T>
struct standard_start;

template <typename T>
struct delayed_start;

template <typename T>
struct stop;

template <typename T>
struct priority_stop;

template <typename T>
struct standard_stop;

template <typename T>
struct delayed_stop;

template <typename T>
struct mark_begin;

template <typename T>
struct mark_end;

template <typename T>
struct store;

template <typename T>
struct audit;

template <typename RetType, typename LhsType, typename RhsType>
struct compose;

template <typename T>
struct plus;

template <typename T>
struct minus;

template <typename T>
struct multiply;

template <typename T>
struct divide;

template <typename T>
struct get;

template <typename T>
struct get_data;

template <typename T>
struct get_labeled_data;

template <typename T>
struct base_printer;

template <typename T>
struct print;

template <typename T>
struct print_header;

template <typename T>
struct print_statistics;

template <typename T>
struct print_storage;

template <typename T>
struct add_secondary;

template <typename T>
struct add_statistics;

template <typename T>
struct serialization;

template <typename T, bool Enabled = trait::echo_enabled<T>::value>
struct echo_measurement;

template <typename T>
struct copy;

template <typename T, typename Op>
struct pointer_operator;

template <typename T>
struct pointer_deleter;

template <typename T>
struct pointer_counter;

template <typename T, typename Op>
struct generic_operator;

template <typename T>
struct generic_deleter;

template <typename T>
struct generic_counter;

namespace finalize
{
template <typename Type, bool has_data>
struct get;

template <typename Type, bool has_data>
struct mpi_get;

template <typename Type, bool has_data>
struct upc_get;

template <typename Type, bool has_data>
struct dmp_get;

//======================================================================================//

template <typename Type>
struct get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    get(storage_type&, result_type&);
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct mpi_get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    mpi_get(storage_type&, distrib_type&);
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct upc_get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    upc_get(storage_type&, distrib_type&);
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct dmp_get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    dmp_get(storage_type&, distrib_type&);
};

//======================================================================================//

template <typename Type>
struct get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    get(storage_type&) {}
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct mpi_get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    mpi_get(storage_type&) {}
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct upc_get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    upc_get(storage_type&) {}
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct dmp_get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    dmp_get(storage_type&) {}
};

//======================================================================================//

}  // namespace finalize
}  // namespace operation

//======================================================================================//
// generic helpers that can/should be inherited from
//
namespace policy
{
template <typename T, bool WithThreads = true>
struct instance_tracker;

template <typename _Comp, typename T = typename trait::statistics<_Comp>::type>
struct record_statistics;

template <typename Archive, typename Api = api::native_tag>
struct input_archive;

template <typename Archive, typename Api = api::native_tag>
struct output_archive;

template <typename T, typename Toolset>
struct omp_tools;

//--------------------------------------------------------------------------------------//
//
//                              ALIASES
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
using input_archive_t = input_archive<trait::input_archive_t<T>, TIMEMORY_API>;

template <typename T>
using output_archive_t = output_archive<trait::output_archive_t<T>, TIMEMORY_API>;
//
//--------------------------------------------------------------------------------------//
//

}  // namespace policy

//--------------------------------------------------------------------------------------//

namespace operation
{
//--------------------------------------------------------------------------------------//
// shorthand for available, non-void
//
template <typename Up>
struct is_enabled
{
    using Vp = typename Up::value_type;
    static constexpr bool value =
        (trait::is_available<Up>::value && !(std::is_same<Vp, void>::value));
};

template <typename U>
using is_enabled_t = typename is_enabled<U>::type;

//--------------------------------------------------------------------------------------//
// shorthand for non-void
//
template <typename Up>
struct has_data
{
    using Vp                    = typename Up::value_type;
    static constexpr bool value = (!std::is_same<Vp, void>::value);
};

template <typename U>
using has_data_t = typename has_data<U>::type;

}  // namespace operation

//======================================================================================//
//
///     \class type_list
///     \brief lightweight tuple-alternative for meta-programming logic
//
//======================================================================================//

template <typename... Tp>
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
//
//          type_list concatenation
//
//--------------------------------------------------------------------------------------//

template <typename... Types>
struct type_concat
{
    using type = type_list<Types...>;
};

//--------------------------------------------------------------------------------------//

template <>
struct type_concat<>
{
    using type = type_list<>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts>
struct type_concat<type_list<Ts...>>
{
    using type = type_list<Ts...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts0, typename... Ts1, typename... Rest>
struct type_concat<type_list<Ts0...>, type_list<Ts1...>, Rest...>
: type_concat<type_list<Ts0..., Ts1...>, Rest...>
{};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//======================================================================================//

/// Alias template make_integer_sequence
template <typename Tp, Tp Num>
using make_integer_sequence = std::make_integer_sequence<Tp, Num>;

/// Alias template index_sequence
template <size_t... Idx>
using index_sequence = std::integer_sequence<size_t, Idx...>;

/// Alias template make_index_sequence
template <size_t Num>
using make_index_sequence = std::make_integer_sequence<size_t, Num>;

/// Alias template index_sequence_for
template <typename... Types>
using index_sequence_for = std::make_index_sequence<sizeof...(Types)>;

/// Alias template for enable_if
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <typename T>
using decay_t = typename std::decay<T>::type;

template <bool Val, typename Lhs, typename Rhs>
using conditional_t = typename std::conditional<(Val), Lhs, Rhs>::type;

template <typename... Ts>
using tuple_concat_t = typename impl::tuple_concat<Ts...>::type;

template <typename... Ts>
using type_concat_t = typename impl::type_concat<Ts...>::type;

template <typename U>
using remove_pointer_t = typename std::remove_pointer<U>::type;

template <typename U>
using add_pointer_t = conditional_t<(std::is_pointer<U>::value), U, U*>;

//======================================================================================//

///
/// get the index of a type in expansion
///
template <typename Tp, typename Type>
struct index_of;

template <typename Tp, template <typename...> class Tuple, typename... Types>
struct index_of<Tp, Tuple<Tp, Types...>>
{
    static constexpr size_t value = 0;
};

template <typename Tp, typename Head, template <typename...> class Tuple,
          typename... Tail>
struct index_of<Tp, Tuple<Head, Tail...>>
{
    static constexpr size_t value = 1 + index_of<Tp, Tuple<Tail...>>::value;
};

//======================================================================================//

namespace impl
{
//--------------------------------------------------------------------------------------//

template <typename T>
struct unwrapper
{
    using type = T;
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... T>
struct unwrapper<Tuple<T...>>
{
    using type = Tuple<T...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... T>
struct unwrapper<Tuple<Tuple<T...>>>
{
    using type = conditional_t<(std::is_same<Tuple<>, std::tuple<>>::value ||
                                std::is_same<Tuple<>, type_list<>>::value),
                               typename unwrapper<Tuple<T...>>::type, Tuple<Tuple<T...>>>;
};

//--------------------------------------------------------------------------------------//

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

    using input_type  = InTuple<In...>;
    using output_type = OutTuple<Out...>;

    static output_type apply(const input_type& _in)
    {
        output_type _out{};
        TIMEMORY_FOLD_EXPRESSION(
            std::get<index_of<Out, output_type>::value>(_out) =
                static_cast<Out>(std::get<index_of<In, input_type>::value>(_in)));
        return _out;
    }
};

//--------------------------------------------------------------------------------------//

template <typename In, typename Out>
struct pointer_convert
{
    using type = add_pointer_t<Out>;
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class InTuple, typename... In,
          template <typename...> class OutTuple, typename... Out>
struct pointer_convert<InTuple<In...>, OutTuple<Out...>>
{
    using type = OutTuple<add_pointer_t<In>...>;
};

}  // namespace impl

//======================================================================================//

namespace impl
{
template <typename T>
struct wrapper_index_sequence;
template <typename T>
struct nonwrapper_index_sequence;

template <template <typename...> class Tuple, typename... Types>
struct wrapper_index_sequence<Tuple<Types...>>
{
    static constexpr auto size  = sizeof...(Types);
    static constexpr auto value = make_index_sequence<size>{};
    using type                  = decltype(make_index_sequence<size>{});
};

template <template <typename...> class Tuple, typename... Types>
struct nonwrapper_index_sequence<Tuple<Types...>>
{
    static constexpr auto size  = sizeof...(Types);
    static constexpr auto value = std::tuple<>{};
    using type                  = std::tuple<>;
};
}  // namespace impl

template <typename Tp>
struct get_index_sequence
{
    static constexpr auto size  = 0;
    static constexpr auto value = std::tuple<>{};
    using type                  = std::tuple<>;
};

template <typename Lhs, typename Rhs>
struct get_index_sequence<std::pair<Lhs, Rhs>>
{
    static constexpr auto size  = 2;
    static constexpr auto value = index_sequence<0, 1>{};
    using type                  = index_sequence<0, 1>;
};

template <typename... Types>
struct get_index_sequence<std::tuple<Types...>>
{
    static constexpr auto size  = std::tuple_size<std::tuple<Types...>>::value;
    static constexpr auto value = make_index_sequence<size>{};
    using type                  = decltype(make_index_sequence<size>{});
};

template <template <typename...> class Tuple, typename... Types>
struct get_index_sequence<Tuple<Types...>>
{
    using base_type = conditional_t<(concepts ::is_variadic<Tuple<Types...>>::value),
                                    impl::wrapper_index_sequence<Tuple<Types...>>,
                                    impl::nonwrapper_index_sequence<Tuple<Types...>>>;
    static constexpr auto size  = base_type::size;
    static constexpr auto value = base_type::value;
    using type                  = typename base_type::type;
};

template <typename Tp>
using get_index_sequence_t = typename get_index_sequence<decay_t<Tp>>::type;

//======================================================================================//

template <typename T, typename U>
using convert_t = typename impl::convert<T, U>::type;

template <typename T, typename U>
using pointer_convert_t = typename impl::pointer_convert<T, U>::type;

template <typename T>
using unwrap_t = typename impl::unwrapper<T>::type;

//======================================================================================//

namespace mpl
{
//--------------------------------------------------------------------------------------//

template <typename _Out, typename _In>
auto
convert(const _In& _in) -> decltype(impl::convert<_In, _Out>::apply(_in))
{
    return impl::convert<_In, _Out>::apply(_in);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Func, typename End = std::function<void()>>
auto
iterate(Tp& _val, Func&& _func, End&& _end = []() {}) -> decltype(std::begin(_val), Tp())
{
    for(auto itr = std::begin(_val); itr != std::end(_val); ++itr)
        _func(*itr);
    _end();
    return _val;
}

template <typename Tp, typename Func, typename End = std::function<void()>>
auto
iterate(Tp& _val, Func&& _func, End&& _end = []() {})
    -> decltype(_func(_val), std::vector<Tp>())
{
    _func(_val);
    _end();
    return std::vector<Tp>({ _val });
}

//--------------------------------------------------------------------------------------//

template <typename Tp,
          typename std::enable_if<(std::is_arithmetic<Tp>::value), int>::type = 0>
constexpr auto
get_size(const Tp&, std::tuple<>) -> size_t
{
    return 1;
}

template <typename Tp>
auto
get_size(const Tp& _val, std::tuple<>) -> decltype(_val.size(), size_t())
{
    return _val.size();
}

template <typename Tp, size_t... Idx>
constexpr auto
get_size(const Tp& _val, index_sequence<Idx...>) -> decltype(std::get<0>(_val), size_t())
{
    return std::tuple_size<Tp>::value;
}

template <typename Tp>
auto
get_size(const Tp& _val)
    -> decltype(get_size(_val, get_index_sequence<decay_t<Tp>>::value))
{
    return get_size(_val, get_index_sequence<decay_t<Tp>>::value);
}

template <typename Tp>
struct get_tuple_size
{
    static constexpr size_t value = get_index_sequence<decay_t<Tp>>::size;
};

//--------------------------------------------------------------------------------------//

template <typename T>
auto
resize(T&, ...) -> void
{}

template <typename T>
auto
resize(T& _targ, size_t _n) -> decltype(_targ.resize(_n), void())
{
    _targ.resize(_n);
}

//--------------------------------------------------------------------------------------//

template <typename T>
struct identity
{
    using type = T;
};

template <typename T>
using identity_t = typename identity<T>::type;

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
assign(Tp& _targ, const Tp& _val, ...)
{
    _targ = _val;
}

template <typename Tp, typename Vp, typename ValueType = typename Tp::value_type>
auto
assign(Tp& _targ, const Vp& _val, std::tuple<>) -> decltype(_targ[0], void())
{
    auto _n = get_size(_val);
    resize(_targ, _n);
    for(decltype(_n) i = 0; i < _n; ++i)
        assign(_targ[i], *(_val.begin() + i),
               get_index_sequence<decay_t<ValueType>>::value);
}

template <typename Tp, size_t... Idx>
auto
assign(Tp& _targ, const Tp& _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), void())
{
    TIMEMORY_FOLD_EXPRESSION(
        assign(std::get<Idx>(_targ), std::get<Idx>(_val),
               get_index_sequence<decay_t<decltype(std::get<Idx>(_targ))>>::value));
}

template <typename Tp, typename Vp, size_t... Idx,
          enable_if_t<!(std::is_same<Tp, Vp>::value), int> = 0>
auto
assign(Tp& _targ, const Vp& _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_targ) = *std::begin(_val), void())
{
    TIMEMORY_FOLD_EXPRESSION(
        assign(std::get<Idx>(_targ), *(std::begin(_val) + Idx),
               get_index_sequence<decay_t<decltype(std::get<Idx>(_targ))>>::value));
}

template <typename Tp, typename Vp>
void
assign(Tp& _targ, const Vp& _val)
{
    assign(_targ, _val, get_index_sequence<decay_t<Tp>>::value);
}

}  // namespace mpl

//======================================================================================//

}  // namespace tim
