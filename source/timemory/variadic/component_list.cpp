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

/** \file timemory/variadic/component_list.cpp
 * \brief Implementation for various component_list member functions
 *
 */

#include "timemory/variadic/component_list.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/utility/utility.hpp"
#include "timemory/variadic/functional.hpp"

#include <string>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
typename component_list<Types...>::initializer_type&
component_list<Types...>::get_initializer()
{
    static auto env_enum = enumerate_components(
        tim::delimit(tim::get_env<std::string>("TIMEMORY_COMPONENT_LIST_INIT", "")));

    static initializer_type _instance = [=](this_type& cl) {
        ::tim::initialize(cl, env_enum);
    };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>::component_list()
{
    if(settings::enabled())
        init_storage();
    apply_v::set_value(m_data, nullptr);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename FuncT>
component_list<Types...>::component_list(const string_t& key, const bool& store,
                                         scope::config _scope, const FuncT& _func)
: bundle_type((settings::enabled()) ? add_hash_id(key) : 0, store, _scope)
, m_data(data_type{})
{
    apply_v::set_value(m_data, nullptr);
    if(settings::enabled())
    {
        init_storage();
        _func(*this);
        set_prefix(get_hash_ids()->find(m_hash)->second);
        invoke::set_scope(m_data, m_scope);
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename FuncT>
component_list<Types...>::component_list(const captured_location_t& loc,
                                         const bool& store, scope::config _scope,
                                         const FuncT& _func)
: bundle_type(loc.get_hash(), store, _scope)
, m_data(data_type{})
{
    apply_v::set_value(m_data, nullptr);
    if(settings::enabled())
    {
        init_storage();
        _func(*this);
        set_prefix(loc.get_hash());
        invoke::set_scope(m_data, m_scope);
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename FuncT>
component_list<Types...>::component_list(size_t _hash, const bool& store,
                                         scope::config _scope, const FuncT& _func)
: bundle_type(_hash, store, _scope)
, m_data(data_type{})
{
    apply_v::set_value(m_data, nullptr);
    if(settings::enabled())
    {
        init_storage();
        _func(*this);
        set_prefix(_hash);
        invoke::set_scope(m_data, m_scope);
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>::~component_list()
{
    if(m_is_active())
        stop();
    // DEBUG_PRINT_HERE("%s", "deleting components");
    apply_v::access<operation_t<operation::generic_deleter>>(m_data);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>::component_list(const this_type& rhs)
: bundle_type(rhs)
{
    apply_v::set_value(m_data, nullptr);
    apply_v::access2<operation_t<operation::copy>>(m_data, rhs.m_data);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>&
component_list<Types...>::operator=(const this_type& rhs)
{
    if(this != &rhs)
    {
        bundle_type::operator=(rhs);
        invoke::destroy(m_data);
        apply_v::access<operation_t<operation::copy>>(m_data);
    }
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>
component_list<Types...>::clone(bool store, scope::config _scope)
{
    component_list tmp(*this);
    tmp.m_store(store);
    tmp.m_scope = _scope;
    return tmp;
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
void
component_list<Types...>::push()
{
    uint64_t count = 0;
    invoke::invoke<operation::generic_counter>(m_data, std::ref(count));
    if(!m_is_pushed() && count > 0)
    {
        // reset data
        invoke::reset(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(true);
        // insert node or find existing node
        invoke::push(m_data, m_scope, m_hash);
    }
}

//--------------------------------------------------------------------------------------//
// pop out of grapsh
//
template <typename... Types>
void
component_list<Types...>::pop()
{
    if(m_is_pushed())
    {
        // set the current node to the parent node
        invoke::pop(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(false);
    }
}

//--------------------------------------------------------------------------------------//
// measure functions
//
template <typename... Types>
template <typename... Args>
void
component_list<Types...>::measure(Args&&... args)
{
    invoke::measure(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// sample functions
//
template <typename... Types>
template <typename... Args>
void
component_list<Types...>::sample(Args&&... args)
{
    sample_type _samples{};
    apply_v::access2<operation_t<operation::sample>>(m_data, _samples,
                                                     std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// start/stop functions
//
template <typename... Types>
template <typename... Args>
void
component_list<Types...>::start(Args&&... args)
{
    // push components into the call-stack
    if(m_store())
        push();
    assemble(*this);
    invoke::start(m_data, std::forward<Args>(args)...);
    m_is_active(true);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... Args>
void
component_list<Types...>::stop(Args&&... args)
{
    invoke::stop(m_data, std::forward<Args>(args)...);
    ++m_laps;
    derive(*this);
    if(m_store())
        pop();
    m_is_active(false);
}

//--------------------------------------------------------------------------------------//
// recording
//
template <typename... Types>
template <typename... Args>
component_list<Types...>&
component_list<Types...>::record(Args&&... args)
{
    ++m_laps;
    invoke::record(m_data, std::forward<Args>(args)...);
    return *this;
}

//--------------------------------------------------------------------------------------//
// reset to zero
//
template <typename... Types>
template <typename... Args>
void
component_list<Types...>::reset(Args&&... args)
{
    invoke::reset(m_data, std::forward<Args>(args)...);
    m_laps = 0;
}

//--------------------------------------------------------------------------------------//
// get data
//
template <typename... Types>
template <typename... Args>
auto
component_list<Types...>::get(Args&&... args) const
{
    return invoke::get(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
template <typename... Args>
auto
component_list<Types...>::get_labeled(Args&&... args) const
{
    return invoke::get_labeled(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// this_type operators
//
template <typename... Types>
component_list<Types...>&
component_list<Types...>::operator-=(const this_type& rhs)
{
    apply_v::access2<operation_t<operation::minus>>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>&
component_list<Types...>::operator-=(this_type& rhs)
{
    apply_v::access2<operation_t<operation::minus>>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>&
component_list<Types...>::operator+=(const this_type& rhs)
{
    apply_v::access2<operation_t<operation::plus>>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_list<Types...>&
component_list<Types...>::operator+=(this_type& rhs)
{
    apply_v::access2<operation_t<operation::plus>>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_list<Types...>::print_storage()
{
    apply_v::type_access<operation::print_storage, reference_type>();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
typename component_list<Types...>::data_type&
component_list<Types...>::data()
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
const typename component_list<Types...>::data_type&
component_list<Types...>::data() const
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_list<Types...>::set_scope(scope::config val)
{
    m_scope = val;
    invoke::set_scope(m_data, m_scope);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename T>
void
component_list<Types...>::set_prefix(T* obj) const
{
    using PrefixOpT =
        operation::generic_operator<T, operation::set_prefix<T>, TIMEMORY_API>;
    auto itr = get_hash_ids()->find(m_hash);
    if(itr != get_hash_ids()->end())
        PrefixOpT(obj, m_hash, itr->second);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_list<Types...>::set_prefix(const string_t& _key) const
{
    invoke::set_prefix(m_data, m_hash, _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_list<Types...>::set_prefix(size_t _hash) const
{
    auto itr = get_hash_ids()->find(_hash);
    if(itr != get_hash_ids()->end())
        invoke::set_prefix(m_data, _hash, itr->second);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_list<Types...>::init_storage()
{
    static thread_local bool _once = []() {
        apply_v::type_access<operation::init_storage, reference_type>();
        return true;
    }();
    consume_parameters(_once);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
