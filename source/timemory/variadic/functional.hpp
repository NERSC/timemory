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

/** \file "timemory/variadic/functional.hpp"
 * This provides function-based forms of the variadic bundlers
 *
 */

#include "timemory/api.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/operations/types/cache.hpp"
#include "timemory/operations/types/generic.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/utility/types.hpp"

#include <type_traits>

namespace tim
{
namespace invoke
{
namespace invoke_impl
{
//
template <typename Tag, template <typename> class Op, typename... Tp>
struct OperatorT
{
    using type = std::tuple<operation::generic_operator<Tp, Op<Tp>, Tag>...>;
};
//
template <typename Tag, template <typename> class Op, typename... Tp>
struct OperatorT<Tag, Op, std::tuple<Tp...>> : OperatorT<Tag, Op, Tp...>
{};
//
template <typename Tag, template <typename> class Op, typename TupleT>
using operation_t = typename OperatorT<Tag, Op, TupleT>::type;
//
template <typename Tag, template <typename, typename> class Op, typename... Tp>
struct OperatorTT
{
    using type = std::tuple<operation::generic_operator<Tp, Op<Tp, Tag>, Tag>...>;
};
//
template <typename Tag, template <typename, typename> class Op, typename... Tp>
struct OperatorTT<Tag, Op, std::tuple<Tp...>> : OperatorTT<Tag, Op, Tp...>
{};
//
template <typename Tag, template <typename, typename> class Op, typename TupleT>
using operation_tt = typename OperatorTT<Tag, Op, TupleT>::type;
//
//--------------------------------------------------------------------------------------//
//
template <template <typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(TupleT<Tp...>& _obj, Args&&... _args)
{
    using data_type = std::tuple<Tp...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<std::remove_pointer_t<Tp>,
                                    OpT<std::remove_pointer_t<Tp>>, Tag>(
            std::get<index_of<Tp, data_type>::value>(_obj),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename, typename> class OpT, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke(TupleT<Tp...>& _obj, Args&&... _args)
{
    using data_type = std::tuple<Tp...>;
    TIMEMORY_FOLD_EXPRESSION(
        operation::generic_operator<std::remove_pointer_t<Tp>,
                                    OpT<std::remove_pointer_t<Tp>, Tag>, Tag>(
            std::get<index_of<Tp, data_type>::value>(_obj),
            std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename> class OpT, typename OpTupleT, size_t Idx, typename Tag,
          template <typename...> class TupleT, typename... Tp, typename... Args>
void
invoke_out_of_order(TupleT<Tp...>& _obj, Args&&... _args)
{
    using OperT = operation_t<Tag, OpT, OpTupleT>;
    apply<void>::out_of_order<OperT, OpTupleT, Idx>(_obj, std::forward<Args>(_args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename, typename> class OpT, typename OpTupleT, size_t Idx,
          typename Tag, template <typename...> class TupleT, typename... Tp,
          typename... Args>
void
invoke_out_of_order(TupleT<Tp...>& _obj, Args&&... _args)
{
    using OperT = operation_tt<Tag, OpT, OpTupleT>;
    apply<void>::out_of_order<OperT, OpTupleT, Idx>(_obj, std::forward<Args>(_args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
construct(TupleT<Tp...>& _obj, Args&&... _args)
{
    using data_type = std::tuple<Tp...>;
    TIMEMORY_FOLD_EXPRESSION(
        std::get<index_of<Tp, data_type>::value>(_obj) =
            operation::construct<Tp>::get(std::forward<Args>(_args)...));
}
//
}  // namespace invoke_impl
//
//======================================================================================//
//
template <typename... Args>
void
print(std::ostream& os, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(os << args << "\n");
}
//
template <typename... Args>
void
print(std::ostream& os, const std::string& delim, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(os << args << delim);
}
//
template <template <typename...> class OpT, typename ApiT,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
invoke(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<OpT, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class OpT, template <typename...> class TupleT,
          typename... Tp, typename... Args>
TIMEMORY_HOT void
invoke(TupleT<Tp...>& obj, Args&&... args)
{
    invoke<OpT, TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename TupleT, typename ApiT, typename... Args>
TIMEMORY_HOT auto
construct(Args&&... args)
{
    IF_CONSTEXPR(trait::is_available<ApiT>::value)
    {
        if(settings::enabled())
        {
            TupleT obj;
            invoke_impl::construct(std::ref(obj).get(), std::forward<Args>(args)...);
            return obj;
        }
    }
    return TupleT{};
}
//
//
template <typename TupleT, typename... Args>
TIMEMORY_HOT auto
construct(Args&&... args)
{
    return construct<TupleT, TIMEMORY_API>(std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
TIMEMORY_HOT auto
destroy(TupleT<Tp...>& obj)
{
    invoke_impl::invoke<operation::generic_deleter, ApiT>(obj);
}
//
template <template <typename...> class TupleT, typename... Tp>
TIMEMORY_HOT auto
destroy(TupleT<Tp...>& obj)
{
    destroy<TIMEMORY_API>(obj);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
start(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
    {
        using data_type        = std::tuple<remove_pointer_t<decay_t<Tp>>...>;
        using priority_types_t = filter_false_t<negative_start_priority, data_type>;
        using priority_tuple_t = mpl::sort<trait::start_priority, priority_types_t>;
        using delayed_types_t  = filter_false_t<positive_start_priority, data_type>;
        using delayed_tuple_t  = mpl::sort<trait::start_priority, delayed_types_t>;

        // start high priority components
        invoke_impl::invoke_out_of_order<operation::priority_start, priority_tuple_t, 1,
                                         ApiT>(obj, std::forward<Args>(args)...);
        // start non-prioritized components
        invoke_impl::invoke<operation::standard_start, ApiT>(obj,
                                                             std::forward<Args>(args)...);
        // start low prioritized components
        invoke_impl::invoke_out_of_order<operation::delayed_start, delayed_tuple_t, 1,
                                         ApiT>(obj, std::forward<Args>(args)...);
    }
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
start(TupleT<Tp...>& obj, Args&&... args)
{
    start<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
stop(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
    {
        using data_type        = std::tuple<remove_pointer_t<decay_t<Tp>>...>;
        using priority_types_t = filter_false_t<negative_stop_priority, data_type>;
        using priority_tuple_t = mpl::sort<trait::stop_priority, priority_types_t>;
        using delayed_types_t  = filter_false_t<positive_stop_priority, data_type>;
        using delayed_tuple_t  = mpl::sort<trait::stop_priority, delayed_types_t>;

        // stop high priority components
        invoke_impl::invoke_out_of_order<operation::priority_stop, priority_tuple_t, 1,
                                         ApiT>(obj, std::forward<Args>(args)...);
        // stop non-prioritized components
        invoke_impl::invoke<operation::standard_stop, ApiT>(obj,
                                                            std::forward<Args>(args)...);
        // stop low prioritized components
        invoke_impl::invoke_out_of_order<operation::delayed_stop, delayed_tuple_t, 1,
                                         ApiT>(obj, std::forward<Args>(args)...);
    }
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
stop(TupleT<Tp...>& obj, Args&&... args)
{
    stop<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
mark_begin(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::mark_begin, ApiT>(obj,
                                                         std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
mark_begin(TupleT<Tp...>& obj, Args&&... args)
{
    mark_begin<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
mark_end(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::mark_end, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
mark_end(TupleT<Tp...>& obj, Args&&... args)
{
    mark_end<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
store(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::store, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
store(TupleT<Tp...>& obj, Args&&... args)
{
    store<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
reset(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::reset, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
reset(TupleT<Tp...>& obj, Args&&... args)
{
    reset<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
record(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::record, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
record(TupleT<Tp...>& obj, Args&&... args)
{
    record<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
measure(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::measure, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
measure(TupleT<Tp...>& obj, Args&&... args)
{
    measure<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
push(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::push_node, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
push(TupleT<Tp...>& obj, Args&&... args)
{
    push<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
pop(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::pop_node, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
pop(TupleT<Tp...>& obj, Args&&... args)
{
    pop<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
set_prefix(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::set_prefix, ApiT>(obj,
                                                         std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
set_prefix(TupleT<Tp...>& obj, Args&&... args)
{
    set_prefix<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
set_scope(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::set_scope, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
set_scope(TupleT<Tp...>& obj, Args&&... args)
{
    set_scope<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
assemble(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::assemble, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
assemble(TupleT<Tp...>& obj, Args&&... args)
{
    assemble<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
derive(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::derive, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
derive(TupleT<Tp...>& obj, Args&&... args)
{
    derive<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
audit(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::audit, ApiT>(obj, std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
audit(TupleT<Tp...>& obj, Args&&... args)
{
    audit<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT void
add_secondary(TupleT<Tp...>& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::invoke<operation::add_secondary, ApiT>(obj,
                                                            std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
add_secondary(TupleT<Tp...>& obj, Args&&... args)
{
    add_secondary<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT auto
get(TupleT<Tp...>& obj, Args&&... args)
{
    using data_type         = TupleT<std::remove_pointer_t<Tp>...>;
    using data_collect_type = get_data_type_t<data_type>;
    using data_value_type   = get_data_value_t<data_type>;

    data_value_type _data{};
    invoke_impl::invoke_out_of_order<operation::get_data, data_collect_type, 2, ApiT>(
        obj, _data, std::forward<Args>(args)...);
    return _data;
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT auto
get(TupleT<Tp...>& obj, Args&&... args)
{
    return ::tim::invoke::get<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_HOT auto
get_labeled(TupleT<Tp...>& obj, Args&&... args)
{
    using data_type         = TupleT<std::remove_pointer_t<Tp>...>;
    using data_collect_type = get_data_type_t<data_type>;
    using data_label_type   = get_data_label_t<data_type>;

    data_label_type _data{};
    invoke_impl::invoke_out_of_order<operation::get_labeled_data, data_collect_type, 2,
                                     ApiT>(obj, _data, std::forward<Args>(args)...);
    return _data;
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT auto
get_labeled(TupleT<Tp...>& obj, Args&&... args)
{
    return get_labeled<TIMEMORY_API>(obj, std::forward<Args>(args)...);
}
//
template <typename... BundleT>
TIMEMORY_HOT auto
get_cache()
{
    operation::construct_cache<std::tuple<BundleT...>> tmp{};
    return tmp();
}
//
//--------------------------------------------------------------------------------------//
//
//          Forwarded bundles
//
//--------------------------------------------------------------------------------------//
//
namespace disjoint
{
//
namespace invoke_impl
{
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
start(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).start(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
stop(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).stop(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
mark_begin(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).mark_begin(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
mark_end(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).mark_end(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
store(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).store(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
reset(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).reset(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
record(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).record(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
measure(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).measure(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
push(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).push(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
pop(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).pop(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
set_prefix(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).set_prefix(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
set_scope(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).set_scope(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
assemble(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).assemble(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
derive(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).derive(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
audit(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(obj).audit(std::forward<Args>(args)...));
}
//
template <template <typename...> class TupleT, typename... Tp, size_t... Idx,
          typename... Args>
void
add_secondary(TupleT<Tp...>&& obj, index_sequence<Idx...>, Args&&... args)
{
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(obj).add_secondary(std::forward<Args>(args)...));
}
//
}  // namespace invoke_impl
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
start(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::start(std::forward<TupleT<Tp...>>(obj),
                           std::make_index_sequence<sizeof...(Tp)>{},
                           std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
stop(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::stop(std::forward<TupleT<Tp...>>(obj),
                          std::make_index_sequence<sizeof...(Tp)>{},
                          std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
mark_begin(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::mark_begin(std::forward<TupleT<Tp...>>(obj),
                                std::make_index_sequence<sizeof...(Tp)>{},
                                std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
mark_end(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::mark_end(std::forward<TupleT<Tp...>>(obj),
                              std::make_index_sequence<sizeof...(Tp)>{},
                              std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
store(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::store(std::forward<TupleT<Tp...>>(obj),
                           std::make_index_sequence<sizeof...(Tp)>{},
                           std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
reset(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::reset(std::forward<TupleT<Tp...>>(obj),
                           std::make_index_sequence<sizeof...(Tp)>{},
                           std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
record(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::record(std::forward<TupleT<Tp...>>(obj),
                            std::make_index_sequence<sizeof...(Tp)>{},
                            std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
measure(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::measure(std::forward<TupleT<Tp...>>(obj),
                             std::make_index_sequence<sizeof...(Tp)>{},
                             std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
push(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::push(std::forward<TupleT<Tp...>>(obj),
                          std::make_index_sequence<sizeof...(Tp)>{},
                          std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
pop(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::pop(std::forward<TupleT<Tp...>>(obj),
                         std::make_index_sequence<sizeof...(Tp)>{},
                         std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
set_prefix(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::set_prefix(std::forward<TupleT<Tp...>>(obj),
                                std::make_index_sequence<sizeof...(Tp)>{},
                                std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
set_scope(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::set_scope(std::forward<TupleT<Tp...>>(obj),
                               std::make_index_sequence<sizeof...(Tp)>{},
                               std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
assemble(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::assemble(std::forward<TupleT<Tp...>>(obj),
                              std::make_index_sequence<sizeof...(Tp)>{},
                              std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
derive(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::derive(std::forward<TupleT<Tp...>>(obj),
                            std::make_index_sequence<sizeof...(Tp)>{},
                            std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
audit(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::audit(std::forward<TupleT<Tp...>>(obj),
                           std::make_index_sequence<sizeof...(Tp)>{},
                           std::forward<Args>(args)...);
}
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_HOT void
add_secondary(TupleT<Tp...>&& obj, Args&&... args)
{
    if(settings::enabled())
        invoke_impl::add_secondary(std::forward<TupleT<Tp...>>(obj),
                                   std::make_index_sequence<sizeof...(Tp)>{},
                                   std::forward<Args>(args)...);
}
}  // namespace disjoint
//
//--------------------------------------------------------------------------------------//
//
}  // namespace invoke
}  // namespace tim
