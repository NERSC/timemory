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

/** \file timemory/variadic/component_bundle.cpp
 * \brief Implementation for various component_bundle member functions
 *
 */

#include "timemory/variadic/component_bundle.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/runtime/types.hpp"
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
    static initializer_type _instance = [](this_type& cl) {
        static auto env_enum = []() {
            auto _tag = demangle<Tag>();
            for(const auto& itr : { string_t("tim::"), string_t("api::") })
            {
                auto _pos = _tag.find(itr);
                do
                {
                    if(_pos != std::string::npos)
                        _tag = _tag.erase(_pos, itr.length());
                    _pos = _tag.find(itr);
                } while(_pos != std::string::npos);
            }

            for(const auto& itr : { string_t("::"), string_t("<"), string_t(">"),
                                    string_t(" "), string_t("__") })
            {
                auto _pos = _tag.find(itr);
                do
                {
                    if(_pos != std::string::npos)
                        _tag = _tag.replace(_pos, itr.length(), "_");
                    _pos = _tag.find(itr);
                } while(_pos != std::string::npos);
            }

            if(_tag.length() > 0 && _tag.at(0) == '_')
                _tag = _tag.substr(1);
            if(_tag.length() > 0 && _tag.at(_tag.size() - 1) == '_')
                _tag = _tag.substr(0, _tag.size() - 1);

            for(auto& itr : _tag)
                itr = toupper(itr);
            auto env_var = string_t("TIMEMORY_") + _tag + "_COMPONENTS";
            if(settings::debug() || settings::verbose() > 0)
                PRINT_HERE("%s is using environment variable: '%s'",
                           demangle<this_type>().c_str(), env_var.c_str());

            // get environment variable
            return enumerate_components(
                tim::delimit(tim::get_env<string_t>(env_var, "")));
        }();
        ::tim::initialize(cl, env_enum);
    };
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
component_bundle<Tag, Types...>::component_bundle(const string_t&     key,
                                                  quirk::config<T...> config,
                                                  const Func&         init_func)
: bundle_type(((settings::enabled()) ? add_hash_id(key) : 0), quirk::config<T...>{})
, m_data(invoke::construct<data_type, Tag>(key, config))
{
    apply_v::set_value(m_data, nullptr);
    if(m_store())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_init, T...>::value) { init_func(*this); }
        set_prefix(get_hash_ids()->find(m_hash)->second);
        invoke::set_scope<Tag>(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start, T...>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename... T, typename Func>
component_bundle<Tag, Types...>::component_bundle(const captured_location_t& loc,
                                                  quirk::config<T...>        config,
                                                  const Func&                init_func)
: bundle_type(loc.get_hash(), quirk::config<T...>{})
, m_data(invoke::construct<data_type, Tag>(loc, config))
{
    apply_v::set_value(m_data, nullptr);
    if(m_store() && trait::runtime_enabled<Tag>::get())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_init, T...>::value) { init_func(*this); }
        set_prefix(loc.get_hash());
        invoke::set_scope<Tag>(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start, T...>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(const string_t& key, const bool& store,
                                                  scope::config _scope,
                                                  const Func&   init_func)
: bundle_type((settings::enabled()) ? add_hash_id(key) : 0, store,
              _scope + scope::config(quirk_config<quirk::flat_scope>::value,
                                     quirk_config<quirk::timeline_scope>::value,
                                     quirk_config<quirk::tree_scope>::value))
, m_data(invoke::construct<data_type, Tag>(key, store, _scope))
{
    apply_v::set_value(m_data, nullptr);
    if(m_store() && trait::runtime_enabled<Tag>::get())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_init>::value) { init_func(*this); }
        set_prefix(get_hash_ids()->find(m_hash)->second);
        invoke::set_scope<Tag>(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(const captured_location_t& loc,
                                                  const bool& store, scope::config _scope,
                                                  const Func& init_func)
: bundle_type(loc.get_hash(), store,
              _scope + scope::config(quirk_config<quirk::flat_scope>::value,
                                     quirk_config<quirk::timeline_scope>::value,
                                     quirk_config<quirk::tree_scope>::value))
, m_data(invoke::construct<data_type, Tag>(loc, store, _scope))
{
    apply_v::set_value(m_data, nullptr);
    if(m_store() && trait::runtime_enabled<Tag>::get())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_init>::value) { init_func(*this); }
        set_prefix(loc.get_hash());
        invoke::set_scope<Tag>(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
template <typename Func>
component_bundle<Tag, Types...>::component_bundle(size_t hash, const bool& store,
                                                  scope::config _scope,
                                                  const Func&   init_func)
: bundle_type(hash, store,
              _scope + scope::config(quirk_config<quirk::flat_scope>::value,
                                     quirk_config<quirk::timeline_scope>::value,
                                     quirk_config<quirk::tree_scope>::value))
, m_data(invoke::construct<data_type, Tag>(hash, store, _scope))
{
    apply_v::set_value(m_data, nullptr);
    if(m_store() && trait::runtime_enabled<Tag>::get())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_init>::value) { init_func(*this); }
        set_prefix(hash);
        invoke::set_scope<Tag>(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename Tag, typename... Types>
component_bundle<Tag, Types...>::~component_bundle()
{
    if(m_is_active())
        stop();
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
    sample_type _samples{};
    if(trait::runtime_enabled<Tag>::get())
        apply_v::access2<operation_t<operation::sample>>(m_data, _samples,
                                                         std::forward<Args>(args)...);
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
    using PrefixOpT =
        operation::generic_operator<T, operation::set_prefix<T>, TIMEMORY_API>;
    auto _key = get_hash_ids()->find(m_hash)->second;
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
