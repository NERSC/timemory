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

template <typename _Tp, typename _Vp = typename _Tp::value_type>
struct generates_output;
template <typename _Tp, typename _Vp = typename _Tp::value_type>
struct implements_storage;

//======================================================================================//
// type-traits for customization
//
namespace trait
{
template <typename _Tp>
struct is_available;

template <typename _Tp>
struct runtime_enabled;

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

template <typename _Tp>
struct pretty_json;

template <typename _Tp>
struct flat_storage;

template <typename _Tp>
struct report_sum;

template <typename _Tp>
struct report_mean;
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

template <typename _Tp>
struct set_flat_profile;

template <typename _Tp>
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
struct store;

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

template <typename _Tp>
struct add_secondary;

template <typename _Tp, typename _Archive>
struct serialization;

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

namespace finalize
{
namespace storage
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

}  // namespace storage
}  // namespace finalize
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

template <typename _Tp>
struct get_index_sequence
{
    static constexpr auto size  = 0;
    static constexpr auto value = std::tuple<>{};
    using type                  = std::tuple<>;
};

template <typename _Lhs, typename _Rhs>
struct get_index_sequence<std::pair<_Lhs, _Rhs>>
{
    static constexpr auto size  = 2;
    static constexpr auto value = index_sequence<0, 1>{};
    using type                  = index_sequence<0, 1>;
};

template <typename... _Types>
struct get_index_sequence<std::tuple<_Types...>>
{
    static constexpr auto size  = std::tuple_size<std::tuple<_Types...>>::value;
    static constexpr auto value = make_index_sequence<size>{};
    using type                  = decltype(make_index_sequence<size>{});
};

template <typename _Tp>
using get_index_sequence_t = typename get_index_sequence<decay_t<_Tp>>::type;

//======================================================================================//

template <typename T, typename U>
using convert_t = typename impl::convert<T, U>::type;

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

template <typename _Tp, typename _Func, typename _End = std::function<void()>>
auto
iterate(_Tp& _val, _Func&& _func, _End&& _end = []() {})
    -> decltype(std::begin(_val), _Tp())
{
    for(auto itr = std::begin(_val); itr != std::end(_val); ++itr)
        _func(*itr);
    _end();
    return _val;
}

template <typename _Tp, typename _Func, typename _End = std::function<void()>>
auto
iterate(_Tp& _val, _Func&& _func, _End&& _end = []() {})
    -> decltype(_func(_val), std::vector<_Tp>())
{
    _func(_val);
    _end();
    return std::vector<_Tp>({ _val });
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Func>
auto
transform(_Tp& _val, _Func&& _func, std::tuple<>) -> decltype(_func(_val), void())
{
    _func(_val);
}

template <typename _Tp, typename _Func>
auto
transform(_Tp& _val, _Func&& _func, std::tuple<>) -> decltype(std::begin(_val), void())
{
    for(auto& itr : _val)
        transform(itr, std::forward<_Func>(_func),
                  get_index_sequence<decay_t<decltype(itr)>>::value);
}

template <typename _Tp, typename _Func>
_Tp
transform(_Tp _val, _Func&& _func)
{
    auto index = get_index_sequence<decay_t<_Tp>>::value;
    transform(_val, std::forward<_Func>(_func), index);
    return _val;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Func>
auto
unary_op(const _Tp& _lhs, _Func&& _func, std::tuple<>) -> decltype(_func(_lhs), _Tp())
{
    return _func(_lhs);
}

template <typename _Tp, typename _Func>
auto
unary_op(const _Tp& _lhs, _Func&& _func, std::tuple<>)
    -> decltype(std::begin(_lhs), _Tp())
{
    auto _n = get_size(_lhs);
    _Tp  _ret{};
    resize(_ret, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = unary_op(*litr, std::forward<_Func>(_func),
                        get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename _Tp, typename _Func>
_Tp
unary_op(const _Tp& _lhs, _Func&& _func)
{
    auto index = get_index_sequence<decay_t<_Tp>>::value;
    return unary_op(_lhs, std::forward<_Func>(_func), index);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Func>
auto
binary_op(const _Tp& _lhs, const _Tp& _rhs, _Func&& _func, std::tuple<>)
    -> decltype(_func(_lhs, _rhs), _Tp())
{
    return _func(_lhs, _rhs);
}

template <typename _Tp, typename _Func>
auto
binary_op(const _Tp& _lhs, const _Tp& _rhs, _Func&& _func, std::tuple<>)
    -> decltype(std::begin(_lhs), _Tp())
{
    auto _nl    = get_size(_lhs);
    auto _nr    = get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    _Tp  _ret{};
    resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = binary_op(*litr, *ritr, std::forward<_Func>(_func),
                         get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename _Tp, typename _Func>
_Tp
binary_op(const _Tp& _lhs, const _Tp& _rhs, _Func&& _func)
{
    auto index = get_index_sequence<decay_t<_Tp>>::value;
    return binary_op(_lhs, _rhs, std::forward<_Func>(_func), index);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Func>
auto
binary_set(_Tp& _lhs, const _Tp& _rhs, _Func&& _func, std::tuple<>)
    -> decltype(_func(_lhs, _rhs), void())
{
    _func(_lhs, _rhs);
}

template <typename _Tp, typename _Func>
auto
binary_set(_Tp& _lhs, const _Tp& _rhs, _Func&& _func, std::tuple<>)
    -> decltype(std::begin(_lhs), void())
{
    auto _n = get_size(_rhs);
    resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        binary_set(*litr, *ritr, std::forward<_Func>(_func),
                   get_index_sequence<decay_t<decltype(*litr)>>::value);
    }
}

template <typename _Tp, typename _Func>
void
binary_set(_Tp& _lhs, const _Tp& _rhs, _Func&& _func)
{
    auto index = get_index_sequence<decay_t<_Tp>>::value;
    binary_set(_lhs, _rhs, std::forward<_Func>(_func), index);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(std::is_arithmetic<_Tp>::value), int>::type = 0>
constexpr auto
get_size(const _Tp&, std::tuple<>) -> size_t
{
    return 1;
}

template <typename _Tp>
auto
get_size(const _Tp& _val, std::tuple<>) -> decltype(_val.size(), size_t())
{
    return _val.size();
}

template <typename _Tp, size_t... _Idx>
constexpr auto
get_size(const _Tp& _val, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_val), size_t())
{
    return std::tuple_size<_Tp>::value;
}

template <typename _Tp>
auto
get_size(const _Tp& _val)
    -> decltype(get_size(_val, get_index_sequence<decay_t<_Tp>>::value))
{
    return get_size(_val, get_index_sequence<decay_t<_Tp>>::value);
}

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

template <typename _Tp>
void
assign(_Tp& _targ, const _Tp& _val, ...)
{
    _targ = _val;
}

template <typename _Tp, typename _Vp, typename ValueType = typename _Tp::value_type>
auto
assign(_Tp& _targ, const _Vp& _val, std::tuple<>) -> decltype(_targ[0], void())
{
    auto _n = get_size(_val);
    resize(_targ, _n);
    for(decltype(_n) i = 0; i < _n; ++i)
        assign(_targ[i], *(_val.begin() + i),
               get_index_sequence<decay_t<ValueType>>::value);
}

template <typename _Tp, size_t... _Idx>
auto
assign(_Tp& _targ, const _Tp& _val, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_val), void())
{
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (assign(std::get<_Idx>(_targ), std::get<_Idx>(_val),
                  get_index_sequence<decay_t<decltype(std::get<_Idx>(_targ))>>::value),
           0)... });
    consume_parameters(tmp);
}

template <typename _Tp, typename _Vp, size_t... _Idx,
          enable_if_t<!(std::is_same<_Tp, _Vp>::value), int> = 0>
auto
assign(_Tp& _targ, const _Vp& _val, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_targ) = *std::begin(_val), void())
{
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (assign(std::get<_Idx>(_targ), *(std::begin(_val) + _Idx),
                  get_index_sequence<decay_t<decltype(std::get<_Idx>(_targ))>>::value),
           0)... });
    consume_parameters(tmp);
}

template <typename _Tp, typename _Vp>
void
assign(_Tp& _targ, const _Vp& _val)
{
    assign(_targ, _val, get_index_sequence<decay_t<_Tp>>::value);
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
