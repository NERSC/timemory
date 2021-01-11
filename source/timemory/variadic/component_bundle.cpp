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

#include "timemory/variadic/component_bundle.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
typename component_bundle<Tag, Types...>::initializer_type&
component_bundle<Tag, Types...>::get_initializer()
{
    static initializer_type _instance = [](this_type&) {};
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>::component_bundle()
{
    apply_v::set_value(m_data, nullptr);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename... T, typename Func>
component_bundle<Tag, Types...>::component_bundle(const string_t&     _key,
                                                  quirk::config<T...> _config,
                                                  const Func&         _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, true_type{}, _config))
, m_data(invoke::construct<data_type, Tag>(_key, _config))
{
    apply_v::set_value(m_data, nullptr);
    bundle_type::init(type_list_type{}, *this, m_data, _init_func, _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename... T, typename Func>
component_bundle<Tag, Types...>::component_bundle(const captured_location_t& _loc,
                                                  quirk::config<T...>        _config,
                                                  const Func&                _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, true_type{}, _config))
, m_data(invoke::construct<data_type, Tag>(_loc, _config))
{
    apply_v::set_value(m_data, nullptr);
    bundle_type::init(type_list_type{}, *this, m_data, _init_func, _config);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(size_t _hash, bool _store,
                                                  scope::config _scope,
                                                  const Func&   _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, _store, _scope))
, m_data(invoke::construct<data_type, Tag>(_hash, m_scope))
{
    // apply_v::set_value(m_data, nullptr);
    bundle_type::init(type_list_type{}, *this, m_data, _init_func);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(const string_t& _key, bool _store,
                                                  scope::config _scope,
                                                  const Func&   _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, _store, _scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, _init_func);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(const captured_location_t& _loc,
                                                  bool _store, scope::config _scope,
                                                  const Func& _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, _store, _scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, _init_func);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(size_t _hash, scope::config _scope,
                                                  const Func& _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _hash, true_type{}, _scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, _init_func);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(const string_t& _key,
                                                  scope::config   _scope,
                                                  const Func&     _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _key, true_type{}, _scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, _init_func);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(const captured_location_t& _loc,
                                                  scope::config              _scope,
                                                  const Func&                _init_func)
: bundle_type(bundle_type::handle(type_list_type{}, _loc, true_type{}, _scope))
{
    bundle_type::init(type_list_type{}, *this, m_data, _init_func);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>::~component_bundle()
{
    IF_CONSTEXPR(!quirk_config<quirk::explicit_stop>::value)
    {
        if(m_is_active())
            stop();
    }
    DEBUG_PRINT_HERE("%s", "deleting components");
    invoke::destroy<Tag>(m_data);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>::component_bundle(const this_type& rhs)
: bundle_type(rhs)
{
    apply_v::set_value(m_data, nullptr);
    apply_v::access2<operation_t<operation::copy>>(m_data, rhs.m_data);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>&
component_bundle<Tag, Types...>::operator=(const this_type& rhs)
{
    if(this != &rhs)
    {
        bundle_type::operator=(rhs);
        invoke::destroy<Tag>(m_data);
        apply_v::access<operation_t<operation::copy>>(m_data);
    }
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>
component_bundle<Tag, Types...>::clone(bool _store, scope::config _scope)
{
    component_bundle tmp(*this);
    tmp.m_store(_store);
    tmp.m_scope = _scope;
    return tmp;
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::push()
{
    if(!m_is_pushed())
    {
        // reset the data
        invoke::reset<Tag>(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(true);
        // insert node or find existing node
        invoke::push<Tag>(m_data, m_scope, m_hash);
    }
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::pop()
{
    if(m_is_pushed())
    {
        // set the current node to the parent node
        invoke::pop<Tag>(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(false);
    }
}

//--------------------------------------------------------------------------------------//
// measure functions
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::measure(Args&&... args)
{
    if(!trait::runtime_enabled<Tag>::get())
        return;
    invoke::measure<Tag>(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// sample functions
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::sample(Args&&... args)
{
    invoke::invoke<operation::sample, Tag>(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::start(mpl::lightweight, Args&&... args)
{
    if(!trait::runtime_enabled<Tag>::get())
        return;
    assemble(*this);
    invoke::start<Tag>(m_data, std::forward<Args>(args)...);
    m_is_active(true);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::stop(mpl::lightweight, Args&&... args)
{
    if(!trait::runtime_enabled<Tag>::get())
        return;
    invoke::stop<Tag>(m_data, std::forward<Args>(args)...);
    ++m_laps;
    derive(*this);
    m_is_active(false);
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename Tag, typename... Types>
template <typename... Tp, typename... Args>
void
component_bundle<Tag, Types...>::start(mpl::piecewise_select<Tp...>, Args&&... args)
{
    using standard_tuple_t = mpl::sort<trait::start_priority, std::tuple<Tp...>>;
    using standard_start_t = operation_t<operation::standard_start, standard_tuple_t>;

    TIMEMORY_FOLD_EXPRESSION(
        operation::reset<Tp>(std::get<index_of<Tp, data_type>::value>(m_data)));
    TIMEMORY_FOLD_EXPRESSION(operation::push_node<Tp>(
        std::get<index_of<Tp, data_type>::value>(m_data), m_scope, m_hash));

    // start components
    apply_v::out_of_order<standard_start_t, standard_tuple_t, 1>(
        m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename... Tp, typename... Args>
void
component_bundle<Tag, Types...>::stop(mpl::piecewise_select<Tp...>, Args&&... args)
{
    using standard_tuple_t = mpl::sort<trait::stop_priority, std::tuple<Tp...>>;
    using standard_stop_t  = operation_t<operation::standard_stop, standard_tuple_t>;

    // stop components
    apply_v::out_of_order<standard_stop_t, standard_tuple_t, 1>(
        m_data, std::forward<Args>(args)...);

    TIMEMORY_FOLD_EXPRESSION(
        operation::pop_node<Tp>(std::get<index_of<Tp, data_type>::value>(m_data)));
}

//--------------------------------------------------------------------------------------//
// start/stop functions
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::start(Args&&... args)
{
    if(!trait::runtime_enabled<Tag>::get())
        return;

    // push components into the call-stack
    if(m_store())
        push();

    // start components
    start(mpl::lightweight{}, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::stop(Args&&... args)
{
    if(!trait::runtime_enabled<Tag>::get())
        return;

    // stop components
    stop(mpl::lightweight{}, std::forward<Args>(args)...);

    // pop components off of the call-stack stack
    if(m_store())
        pop();
}

//--------------------------------------------------------------------------------------//
// recording
//
template <typename Tag, typename... Types>
template <typename... Args>
component_bundle<Tag, Types...>&
component_bundle<Tag, Types...>::record(Args&&... args)
{
    if(!trait::runtime_enabled<Tag>::get())
        return *this;

    ++m_laps;
    invoke::record<Tag>(m_data, std::forward<Args>(args)...);
    return *this;
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename Tag, typename... Types>
template <typename... Args>
void
component_bundle<Tag, Types...>::reset(Args&&... args)
{
    invoke::reset<Tag>(m_data, std::forward<Args>(args)...);
    m_laps = 0;
}

//--------------------------------------------------------------------------------------//
// get data
//
template <typename Tag, typename... Types>
template <typename... Args>
auto
component_bundle<Tag, Types...>::get(Args&&... args) const
{
    return invoke::get<Tag>(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// get labeled data
//
template <typename Tag, typename... Types>
template <typename... Args>
auto
component_bundle<Tag, Types...>::get_labeled(Args&&... args) const
{
    return invoke::get_labeled<Tag>(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// this_type operators
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>&
component_bundle<Tag, Types...>::operator-=(const this_type& rhs)
{
    apply_v::access2<operation_t<operation::minus>>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>&
component_bundle<Tag, Types...>::operator-=(this_type& rhs)
{
    apply_v::access2<operation_t<operation::minus>>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>&
component_bundle<Tag, Types...>::operator+=(const this_type& rhs)
{
    apply_v::access2<operation_t<operation::plus>>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>&
component_bundle<Tag, Types...>::operator+=(this_type& rhs)
{
    apply_v::access2<operation_t<operation::plus>>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::print_storage()
{
    apply_v::type_access<operation::print_storage, data_type>();
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
typename component_bundle<Tag, Types...>::data_type&
component_bundle<Tag, Types...>::data()
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
const typename component_bundle<Tag, Types...>::data_type&
component_bundle<Tag, Types...>::data() const
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename T>
void
component_bundle<Tag, Types...>::set_scope(T* obj) const
{
    using PrefixOpT = operation::generic_operator<T, operation::set_scope<T>, Tag>;
    PrefixOpT(obj, m_scope);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::set_scope(scope::config val)
{
    if(!trait::runtime_enabled<Tag>::get())
        return;
    m_scope = val;
    invoke::set_scope<Tag>(m_data, m_scope);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename T>
void
component_bundle<Tag, Types...>::set_prefix(T* obj) const
{
    using PrefixOpT = operation::generic_operator<T, operation::set_prefix<T>, Tag>;
    auto _key       = get_hash_identifier_fast(m_hash);
    PrefixOpT(obj, m_hash, _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::set_prefix(const string_t& _key) const
{
    if(!trait::runtime_enabled<Tag>::get())
        return;
    invoke::set_prefix<Tag>(m_data, m_hash, _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::set_prefix(size_t _hash) const
{
    if(!trait::runtime_enabled<Tag>::get())
        return;
    auto itr = get_hash_ids()->find(_hash);
    if(itr != get_hash_ids()->end())
        invoke::set_prefix<Tag>(m_data, _hash, itr->second);
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
void
component_bundle<Tag, Types...>::init_storage()
{
    static thread_local bool _once = []() {
        apply_v::type_access<operation::init_storage, reference_type>();
        return true;
    }();
    consume_parameters(_once);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
