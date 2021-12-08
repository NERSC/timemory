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

#ifndef TIMEMORY_STORAGE_IMPL_STORAGE_TRUE_CPP_
#define TIMEMORY_STORAGE_IMPL_STORAGE_TRUE_CPP_ 1

#include "timemory/storage/impl_storage_true.hpp"

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/data/stream.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/init.hpp"
#include "timemory/operations/types/node.hpp"
#include "timemory/operations/types/start.hpp"
#include "timemory/operations/types/stop.hpp"
#include "timemory/plotting/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/demangle.hpp"

#include <cstddef>
#include <fstream>
#include <functional>
#include <memory>
#include <regex>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace tim
{
namespace impl
{
template <typename Type>
storage<Type, true>::storage()
: base_type(singleton_t::is_master_thread(), instance_count()++, demangle<Type>())
{
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "constructing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);

    component::state<Type>::has_storage() = true;

    static std::atomic<int32_t> _skip_once(0);
    if(_skip_once++ > 0)
    {
        // make sure all worker instances have a copy of the hash id and aliases
        auto _master = singleton_t::master_instance();
        if(_master)
        {
            hash_map_t       _hash_ids     = *_master->get_hash_ids();
            hash_alias_map_t _hash_aliases = *_master->get_hash_aliases();
            for(const auto& itr : _hash_ids)
            {
                if(m_hash_ids->find(itr.first) == m_hash_ids->end())
                    m_hash_ids->insert({ itr.first, itr.second });
            }
            for(const auto& itr : _hash_aliases)
            {
                if(m_hash_aliases->find(itr.first) == m_hash_aliases->end())
                    m_hash_aliases->insert({ itr.first, itr.second });
            }
        }
    }

    get_shared_manager();
    // m_printer = std::make_shared<printer_t>(Type::get_label(), this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::~storage()
{
    component::state<Type>::has_storage() = false;

    auto _debug = m_settings->get_debug();

    CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> destroying storage", m_label.c_str(),
                           (long) m_instance_id);

    auto _main_instance = singleton_t::master_instance();

    if(!m_is_master)
    {
        if(_main_instance)
        {
            CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> merging into primary instance",
                                   m_label.c_str(), (long) m_instance_id);
            _main_instance->merge(this);
        }
        else
        {
            CONDITIONAL_PRINT_HERE(_debug,
                                   "[%s|%li]> skipping merge into non-existent primary "
                                   "instance",
                                   m_label.c_str(), (long) m_instance_id);
        }
    }

    if(m_graph_data_instance)
    {
        CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> deleting graph data", m_label.c_str(),
                               (long) m_instance_id);
        delete m_graph_data_instance;
    }

    m_graph_data_instance = nullptr;

    CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> storage destroyed", m_label.c_str(),
                           (long) m_instance_id);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::initialize()
{
    if(m_initialized)
        return;
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "initializing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
    m_initialized = true;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::finalize()
{
    if(m_finalized)
        return;

    if(!m_initialized)
        return;

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "finalizing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);

    m_finalized            = true;
    worker_is_finalizing() = true;
    if(m_is_master)
        master_is_finalizing() = true;
    manager::instance()->is_finalizing(true);

    using fini_t = operation::fini<Type>;
    using ValueT = typename trait::collects_data<Type>::type;
    auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);

    if(m_thread_init)
        fini_t(upcast, operation::mode_constant<operation::fini_mode::thread>{});
    if(m_is_master && m_global_init)
        fini_t(upcast, operation::mode_constant<operation::fini_mode::global>{});

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "finalized %s", m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::stack_clear()
{
    if(!m_stack.empty() && m_settings->get_stack_clearing())
    {
        std::unordered_set<Type*> _stack = m_stack;
        for(auto& itr : _stack)
        {
            operation::generic_operator<Type, operation::start<Type>, TIMEMORY_API>{
                *itr
            };
            operation::generic_operator<Type, operation::pop_node<Type>, TIMEMORY_API>{
                *itr
            };
        }
    }
    m_stack.clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::global_init()
{
    if(!m_global_init)
    {
        CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "[%s|%i]> invoking global_init",
                               demangle<Type>().c_str(), (int) m_thread_idx);
        if(!m_is_master && master_instance())
            master_instance()->global_init();
        m_global_init = true;
        operation::init<Type>{ operation::mode_constant<operation::init_mode::global>{} };
    }
    return m_global_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::thread_init()
{
    if(!m_thread_init)
    {
        global_init();
        CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "[%s|%i]> invoking thread_init",
                               demangle<Type>().c_str(), (int) m_thread_idx);
        if(!m_is_master && master_instance())
            master_instance()->thread_init();
        m_thread_init = true;
        operation::init<Type>{ operation::mode_constant<operation::init_mode::thread>{} };
    }
    return m_thread_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::data_init()
{
    if(!m_data_init)
    {
        global_init();
        thread_init();
        CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "[%s|%i]> invoking data_init",
                               demangle<Type>().c_str(), (int) m_thread_idx);
        if(!m_is_master && master_instance())
            master_instance()->data_init();
        m_data_init = true;
        check_consistency();
    }
    return m_data_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename storage<Type, true>::graph_data_t&
storage<Type, true>::data() const
{
    if(!is_finalizing())
    {
        using type_t                   = decay_t<remove_pointer_t<decltype(this)>>;
        static thread_local auto _init = const_cast<type_t*>(this)->data_init();
        consume_parameters(_init);
    }
    return _data();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename storage<Type, true>::graph_t&
storage<Type, true>::graph() const
{
    if(!is_finalizing())
    {
        using type_t                   = decay_t<remove_pointer_t<decltype(this)>>;
        static thread_local auto _init = const_cast<type_t*>(this)->data_init();
        consume_parameters(_init);
    }
    return _data().graph();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
int64_t
storage<Type, true>::depth() const
{
    if(!is_finalizing())
    {
        using type_t                   = decay_t<remove_pointer_t<decltype(this)>>;
        static thread_local auto _init = const_cast<type_t*>(this)->data_init();
        consume_parameters(_init);
    }
    return (is_finalizing()) ? 0 : _data().depth();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_data_t&
storage<Type, true>::data()
{
    if(!is_finalizing())
    {
        static thread_local auto _init = data_init();
        consume_parameters(_init);
    }
    return _data();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_t&
storage<Type, true>::graph()
{
    if(!is_finalizing())
    {
        static thread_local auto _init = data_init();
        consume_parameters(_init);
    }
    return _data().graph();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator&
storage<Type, true>::current()
{
    if(!is_finalizing())
    {
        static thread_local auto _init = data_init();
        consume_parameters(_init);
    }
    return _data().current();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::pop()
{
    return _data().pop_graph();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::stack_pop(Type* obj)
{
    auto itr = m_stack.find(obj);
    if(itr != m_stack.end())
        m_stack.erase(itr);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::check_consistency()
{
    auto* ptr = &_data();
    if(ptr != m_graph_data_instance)
    {
        fprintf(stderr, "[%s]> mismatched graph data on master thread: %p vs. %p\n",
                m_label.c_str(), (void*) ptr, static_cast<void*>(m_graph_data_instance));
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::ensure_init()
{
    global_init();
    thread_init();
    data_init();
    // check this now to ensure everything is initialized
    if(m_node_ids.empty() || m_graph_data_instance == nullptr)
        initialize();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::string
storage<Type, true>::get_prefix(const graph_node& node)
{
    auto _ret = operation::decode<TIMEMORY_API>{}(m_hash_ids, m_hash_aliases, node.id());
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master && singleton_t::master_instance())
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(node);
        }

        return operation::decode<TIMEMORY_API>{}(node.id());
    }

#if defined(TIMEMORY_TESTING) || defined(TIMEMORY_INTERNAL_TESTING)
    if(_ret.empty() || _ret.find("unknown-hash=") == 0)
    {
        TIMEMORY_EXCEPTION("Hash-lookup error!")
    }
#endif

    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::string
storage<Type, true>::get_prefix(const uint64_t& id)
{
    auto _ret = get_hash_identifier(m_hash_ids, m_hash_aliases, id);
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master && singleton_t::master_instance())
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(id);
        }

        return get_hash_identifier(id);
    }

#if defined(TIMEMORY_TESTING) || defined(TIMEMORY_INTERNAL_TESTING)
    if(_ret.empty() || _ret.find("unknown-hash=") == 0)
    {
        TIMEMORY_EXCEPTION("Hash-lookup error!")
    }
#endif

    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_data_t&
storage<Type, true>::_data()
{
    if(m_graph_data_instance == nullptr)
    {
        auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);

        if(!m_is_master && master_instance())
        {
            static thread_local bool _data_init = master_instance()->data_init();
            auto&                    m          = master_instance()->data();
            consume_parameters(_data_init);

            if(!lk.owns_lock())
                lk.lock();

            CONDITIONAL_PRINT_HERE(
                m_settings->get_debug(), "[%s]> Worker: %i, master ptr: %p",
                demangle<Type>().c_str(), (int) m_thread_idx, (void*) &m);
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
                m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
            if(m.current())
            {
                auto         _current = m.current();
                auto         _id      = _current->id();
                auto         _depth   = _current->depth();
                graph_node_t node(_id, operation::dummy<Type>{}(), _depth, m_thread_idx);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, _depth, &m);
                m_graph_data_instance->depth()     = _depth;
                m_graph_data_instance->sea_level() = _depth;
            }
            else
            {
                if(!m_graph_data_instance)
                {
                    graph_node_t node(0, operation::dummy<Type>{}(), 1, m_thread_idx);
                    m_graph_data_instance = new graph_data_t(node, 1, &m);
                }
                m_graph_data_instance->depth()     = 1;
                m_graph_data_instance->sea_level() = 1;
            }
            m_graph_data_instance->set_master(&m);
        }
        else
        {
            if(!lk.owns_lock())
                lk.lock();

            std::string _prefix = "> [tot] total";
            add_hash_id(_prefix);
            graph_node_t node(0, operation::dummy<Type>{}(), 0, m_thread_idx);
            if(!m_graph_data_instance)
                m_graph_data_instance = new graph_data_t(node, 0, nullptr);
            m_graph_data_instance->depth()     = 0;
            m_graph_data_instance->sea_level() = 0;
            CONDITIONAL_PRINT_HERE(m_settings->get_debug(),
                                   "[%s]> Master: %i, master ptr: %p",
                                   demangle<Type>().c_str(), (int) m_thread_idx,
                                   (void*) m_graph_data_instance);
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
                m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
        }

        if(m_node_ids.empty() && m_graph_data_instance)
        {
            m_node_ids.emplace(0, iterator_hash_submap_t{});
            m_node_ids.at(0).emplace(0, m_graph_data_instance->current());
        }
    }

    m_initialized = true;
    return *m_graph_data_instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::merge()
{
    if(!m_is_master || !m_initialized)
        return;

    auto m_children = singleton_t::children();
    if(m_children.empty())
        return;

    for(auto& itr : m_children)
        merge(itr);

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    for(auto& itr : m_children)
        singleton_t::remove(itr);

    // for(auto& itr : m_children)
    // {
    //     if(itr != this)
    //         itr->data().clear();
    // }

    stack_clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::merge(this_type* itr)
{
    if(itr)
        operation::finalize::merge<Type, true>(*this, *itr);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::result_array_t
storage<Type, true>::get()
{
    result_array_t _ret{};
    operation::finalize::get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::get(Tp& _ret)
{
    return operation::finalize::get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::mpi_get()
{
    dmp_result_t _ret{};
    operation::finalize::mpi_get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::mpi_get(Tp& _ret)
{
    return operation::finalize::mpi_get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::upc_get()
{
    dmp_result_t _ret{};
    operation::finalize::upc_get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::upc_get(Tp& _ret)
{
    return operation::finalize::upc_get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::dmp_get()
{
    dmp_result_t _ret{};
    operation::finalize::dmp_get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::dmp_get(Tp& _ret)
{
    return operation::finalize::dmp_get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::internal_print()
{
    base::storage::stop_profiler();

    if(!m_initialized && !m_finalized)
        return;

    auto _is_primary       = singleton_t::is_master(this);
    auto _primary_instance = singleton_t::master_instance();

    if(!_is_primary && !_primary_instance && common_singleton::is_main_thread())
    {
        PRINT_HERE("[%s]> storage instance (%p) on main thread is not designated as the "
                   "primary but there is a nullptr to primary. Designating as primary",
                   m_label.c_str(), (void*) this);
        _is_primary = true;
    }

    if(!_is_primary)
    {
        if(_primary_instance)
            _primary_instance->merge(this);
        finalize();
    }
    else
    {
        merge();
        finalize();

        if(!trait::runtime_enabled<Type>::get())
        {
            instance_count().store(0);
            return;
        }

        // if the graph wasn't ever initialized, exit
        if(!m_graph_data_instance)
        {
            instance_count().store(0);
            return;
        }

        // no entries
        if(_data().graph().size() <= 1)
        {
            instance_count().store(0);
            return;
        }

        // generate output
        if(m_settings->get_auto_output())
        {
            m_printer.reset(new printer_t(Type::get_label(), this, m_settings));

            if(m_manager)
                m_manager->add_entries(this->size());

            m_printer->execute();
        }

        instance_count().store(0);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::get_shared_manager()
{
    using func_t = std::function<void()>;

    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        if(!m_manager)
            return;

        auto       _label = demangle(Type::label());
        std::regex _namespace_re{ "^(tim::[a-z_]+::|tim::)([a-z].*)" };
        if(std::regex_search(_label, _namespace_re))
            _label = std::regex_replace(_label, _namespace_re, "$2");
        // replace spaces with underscores
        auto _pos = std::string::npos;
        while((_pos = _label.find_first_of(" -")) != std::string::npos)
            _label = _label.replace(_pos, 1, "_");
        // convert to upper-case
        for(auto& itr : _label)
            itr = toupper(itr);
        // handle any remaining brackets or colons
        for(auto itr : { ':', '<', '>' })
        {
            while((_pos = _label.find(itr)) != std::string::npos)
                _pos = _label.erase(_pos, 1).find(itr);
        }
        std::stringstream env_var;
        env_var << TIMEMORY_SETTINGS_PREFIX << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        trait::runtime_enabled<Type>::set(_enabled);

        auto _instance_id = m_instance_id;
        bool _is_master   = m_is_master;
        auto _sync        = [&]() {
            if(m_graph_data_instance)
                this->data().sync_sea_level();
        };
        auto _cleanup = [_is_master, _instance_id]() {
            if(_is_master)
                return;
            if(manager::master_instance())
                manager::master_instance()->remove_synchronization(demangle<Type>(),
                                                                   _instance_id);
            if(manager::instance())
                manager::instance()->remove_synchronization(demangle<Type>(),
                                                            _instance_id);
        };
        func_t _finalize = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                auto _debug_v = m_settings->get_debug();
                auto _verb_v  = m_settings->get_verbose();
                if(_debug_v || _verb_v > 1)
                {
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling singleton::reset(this)");
                }
                _instance->reset(this);
                if((m_is_master || common_singleton::is_main_thread()) && _instance)
                {
                    if(_debug_v || _verb_v > 1)
                    {
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling singleton::reset()");
                    }
                    _instance->reset();
                }
            }
            else
            {
                DEBUG_PRINT_HERE("[%s]> %p", demangle<Type>().c_str(), (void*) _instance);
            }
            if(m_is_master)
                trait::runtime_enabled<Type>::set(false);
        };

        if(!m_is_master)
        {
            manager::master_instance()->add_synchronization(
                demangle<Type>(), m_instance_id, std::move(_sync));
            m_manager->add_synchronization(demangle<Type>(), m_instance_id,
                                           std::move(_sync));
        }

        m_manager->add_finalizer(demangle<Type>(), std::move(_cleanup),
                                 std::move(_finalize), m_is_master,
                                 trait::fini_priority<Type>::value);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::noninit_instance()
{
    return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::noninit_master_instance()
{
    return get_singleton() ? get_singleton()->master_instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, true>::master_is_finalizing()
{
    static bool _instance = false;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, true>::worker_is_finalizing()
{
    static thread_local bool _instance = master_is_finalizing();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::is_finalizing()
{
    return worker_is_finalizing() || master_is_finalizing();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::atomic<int64_t>&
storage<Type, true>::instance_count()
{
    static std::atomic<int64_t> _counter{ 0 };
    return _counter;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::reset()
{
    // have the data graph erase all children of the head node
    if(m_graph_data_instance)
        m_graph_data_instance->reset();
    // erase all the cached iterators except for m_node_ids[0][0]
    for(auto& ditr : m_node_ids)
    {
        auto _depth = ditr.first;
        if(_depth != 0)
        {
            ditr.second.clear();
        }
        else
        {
            for(auto itr = ditr.second.begin(); itr != ditr.second.end(); ++itr)
            {
                if(itr->first != 0)
                    ditr.second.erase(itr);
            }
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert(scope::config scope_data, const Type& obj, uint64_t hash_id,
                            int64_t _tid)
{
    ensure_init();

    using force_tree_t = trait::tree_storage<Type>;
    using force_flat_t = trait::flat_storage<Type>;
    using force_time_t = trait::timeline_storage<Type>;

    // if data is all the way up to the zeroth (relative) depth then worker
    // threads should insert a new dummy at the current master thread id and depth.
    // Be aware, this changes 'm_current' inside the data graph
    //
    if(!m_is_master && _data().at_sea_level() &&
       _data().dummy_count() < m_settings->get_max_thread_bookmarks())
        _data().add_dummy();

    if(_tid < 0)
        _tid = m_thread_idx;

    // compute the insertion depth
    auto hash_depth = scope_data.compute_depth<force_tree_t, force_flat_t, force_time_t>(
        _data().depth());

    // compute the insertion key
    auto hash_value = scope_data.compute_hash<force_tree_t, force_flat_t, force_time_t>(
        hash_id, hash_depth, m_timeline_counter);

    // alias the true id with the insertion key
    add_hash_id(hash_id, hash_value);

    // even when flat is combined with timeline, it still inserts at depth of 1
    // so this is easiest check
    if(scope_data.is_flat() || force_flat_t::value)
        return insert_flat(hash_value, obj, hash_depth, _tid);

    // in the case of tree + timeline, timeline will have appropriately modified the
    // depth and hash so it doesn't really matter which check happens first here
    // however, the query for is_timeline() is cheaper so we will check that
    // and fallback to inserting into tree without a check
    // if(scope_data.is_timeline() || force_time_t::value)
    //    return insert_timeline(hash_value, obj, hash_depth);

    // default fall-through if neither flat nor timeline
    return insert_tree(hash_value, obj, hash_depth, _tid);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Vp, enable_if_t<!std::is_same<decay_t<Vp>, Type>::value, int>>
typename storage<Type, true>::iterator
storage<Type, true>::append(const secondary_data_t<Vp>& _secondary)
{
    ensure_init();

    // get the iterator and check if valid
    auto&& _itr = std::get<0>(_secondary);
    if(!_data().graph().is_valid(_itr))
        return nullptr;

    // compute hash of prefix
    auto _hash_id = add_hash_id(std::get<1>(_secondary));
    // compute hash w.r.t. parent iterator (so identical kernels from different
    // call-graph parents do not locate same iterator)
    auto _hash = get_combined_hash_id(_hash_id, _itr->id());
    // unique thread-hash
    auto _uniq_hash = get_combined_hash_id(_hash, _itr->tid());
    // add the hash alias
    add_hash_id(_hash_id, _hash);
    // compute depth
    auto _depth = _itr->depth() + 1;

    // see if depth + hash entry exists already
    auto _nitr = m_node_ids[_depth].find(_uniq_hash);
    if(_nitr != m_node_ids[_depth].end())
    {
        // if so, then update
        auto& _obj = _nitr->second->data();
        _obj += std::get<2>(_secondary);
        _obj.set_laps(_nitr->second->data().get_laps() + 1);
        auto& _stats = _nitr->second->stats();
        operation::add_statistics<Type>(_nitr->second->data(), _stats);
        return _nitr->second;
    }

    // else, create a new entry
    auto&& _tmp = Type{};
    _tmp += std::get<2>(_secondary);
    _tmp.set_laps(_tmp.get_laps() + 1);
    graph_node_t _node{ _hash, _tmp, _depth, static_cast<uint32_t>(_itr->tid()) };
    _node.stats() += _tmp.get();
    auto& _stats = _node.stats();
    operation::add_statistics<Type>(_tmp, _stats);
    auto itr = _data().emplace_child(_itr, std::move(_node));
    operation::set_iterator<Type>{}(itr->data(), itr);
    m_node_ids[_depth][_uniq_hash] = itr;
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Vp, enable_if_t<std::is_same<decay_t<Vp>, Type>::value, int>>
typename storage<Type, true>::iterator
storage<Type, true>::append(const secondary_data_t<Vp>& _secondary)
{
    ensure_init();

    // get the iterator and check if valid
    auto&& _itr = std::get<0>(_secondary);
    if(!_data().graph().is_valid(_itr))
        return nullptr;

    // compute hash of prefix
    auto _hash_id = add_hash_id(std::get<1>(_secondary));
    // compute hash w.r.t. parent iterator (so identical kernels from different
    // call-graph parents do not locate same iterator)
    auto _hash = get_combined_hash_id(_hash_id, _itr->id(), _itr->tid());
    // unique thread-hash
    auto _uniq_hash = get_combined_hash_id(_hash, _itr->tid());
    // add the hash alias
    add_hash_id(_hash_id, _hash);
    // compute depth
    auto _depth = _itr->depth() + 1;

    // see if depth + hash entry exists already
    auto _nitr = m_node_ids[_depth].find(_uniq_hash);
    if(_nitr != m_node_ids[_depth].end())
    {
        _nitr->second->data() += std::get<2>(_secondary);
        return _nitr->second;
    }

    // else, create a new entry
    auto&& _tmp = std::get<2>(_secondary);
    auto   itr  = _data().emplace_child(
        _itr, graph_node_t{ _hash, _tmp, static_cast<int64_t>(_depth),
                            static_cast<uint32_t>(_itr->tid()) });
    operation::set_iterator<Type>{}(itr->data(), itr);
    m_node_ids[_depth][_uniq_hash] = itr;
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_tree(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                                 int64_t _tid)
{
    bool has_head = _data().has_head();
    return insert_hierarchy(hash_id, obj, hash_depth, has_head, _tid);
}

//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_timeline(uint64_t hash_id, const Type& obj,
                                     uint64_t hash_depth, int64_t _tid)
{
    auto _current = _data().current();
    return _data().emplace_child(_current, graph_node_t{ hash_id, obj,
                                                         static_cast<int64_t>(hash_depth),
                                                         static_cast<uint32_t>(_tid) });
}

//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_flat(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                                 int64_t _tid)
{
    static thread_local auto _current = _data().head();
    static thread_local bool _first   = true;
    // unique thread-hash
    auto _uniq_hash = get_combined_hash_id(hash_id, _tid);
    if(_first)
    {
        _first = false;
        if(_current.begin())
        {
            _current = _current.begin();
        }
        else
        {
            auto itr = _data().emplace_child(
                _current, graph_node_t{ hash_id, obj, static_cast<int64_t>(hash_depth),
                                        static_cast<uint32_t>(_tid) });
            m_node_ids[hash_depth][_uniq_hash] = itr;
            _current                           = itr;
            return itr;
        }
    }

    auto _existing = m_node_ids[hash_depth].find(_uniq_hash);
    if(_existing != m_node_ids[hash_depth].end())
        return m_node_ids[hash_depth].find(_uniq_hash)->second;

    auto itr = _data().emplace_child(
        _current, graph_node_t{ hash_id, obj, static_cast<int64_t>(hash_depth),
                                static_cast<uint32_t>(_tid) });
    m_node_ids[hash_depth][_uniq_hash] = itr;
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_hierarchy(uint64_t hash_id, const Type& obj,
                                      uint64_t hash_depth, bool has_head, int64_t _tid)
{
    using id_hash_map_t = typename iterator_hash_map_t::mapped_type;

    auto& m_data = m_graph_data_instance;

    // unique thread-hash
    auto _uniq_hash = get_combined_hash_id(hash_id, _tid);

    // if first instance
    if(!has_head || (m_is_master && m_node_ids.empty()))
    {
        auto itr = m_data->append_child(graph_node_t{ hash_id, obj,
                                                      static_cast<int64_t>(hash_depth),
                                                      static_cast<uint32_t>(_tid) });
        m_node_ids[hash_depth][_uniq_hash] = itr;
        return itr;
    }

    // lambda for updating settings
    auto _update = [&](iterator itr) {
        m_data->depth() = itr->depth();
        return (m_data->current() = itr);
    };

    auto _nitr = m_node_ids[hash_depth].find(_uniq_hash);
    if(_nitr != m_node_ids[hash_depth].end() && _nitr->second->depth() == m_data->depth())
    {
        return _update(_nitr->second);
    }

    using sibling_itr = typename graph_t::sibling_iterator;
    graph_node_t node{ hash_id, obj, m_data->depth(), static_cast<uint32_t>(_tid) };

    // lambda for inserting child
    auto _insert_child = [&]() {
        node.depth() = hash_depth;
        auto itr     = m_data->append_child(std::move(node));
        auto ditr    = m_node_ids.find(hash_depth);
        if(ditr == m_node_ids.end())
            m_node_ids.insert({ hash_depth, id_hash_map_t{} });
        auto hitr = m_node_ids.at(hash_depth).find(_uniq_hash);
        if(hitr == m_node_ids.at(hash_depth).end())
            m_node_ids.at(hash_depth).insert({ _uniq_hash, iterator{} });
        m_node_ids.at(hash_depth).at(_uniq_hash) = itr;
        return itr;
    };

    auto current = m_data->current();
    if(!m_data->graph().is_valid(current))
        _insert_child();  // create valid current, intentional non-return

    // check children first because in general, child match is ideal
    auto fchild = graph_t::child(current, 0);
    if(m_data->graph().is_valid(fchild))
    {
        for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
        {
            if((hash_id) == itr->id() && _tid == itr->tid())
                return _update(itr);
        }
    }

    // occasionally, we end up here because of some of the threading stuff that
    // has to do with the head node. Protected against mis-matches in hierarchy
    // because the actual hash includes the depth so "example" at depth 2
    // has a different hash than "example" at depth 3.
    if((hash_id) == current->id() && _tid == current->tid())
        return current;

    // check siblings
    for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
    {
        // skip if current
        if(itr == current)
            continue;
        // check hash id's
        if((hash_id) == itr->id() && _tid == itr->tid())
            return _update(itr);
    }

    return _insert_child();
}

//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
void
storage<Type, true>::serialize(Archive& ar, const unsigned int version)
{
    auto&& _results = dmp_get();
    operation::serialization<Type>{}(ar, _results);
    consume_parameters(version);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
void
storage<Type, true>::do_serialize(Archive& ar)
{
    if(m_is_master)
        merge();

    auto&& _results = dmp_get();
    operation::serialization<Type>{}(ar, _results);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
}

}  // namespace impl
}  // namespace tim

#endif
