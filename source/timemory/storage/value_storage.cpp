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

#ifndef TIMEMORY_STORAGE_VALUE_STORAGE_CPP_
#define TIMEMORY_STORAGE_VALUE_STORAGE_CPP_ 1

#include "timemory/storage/value_storage.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
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
#include "timemory/storage/declaration.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/types.hpp"

#include <fstream>
#include <memory>
#include <utility>

namespace tim
{
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::atomic<int64_t>&
value_storage<Type>::instance_count()
{
    static std::atomic<int64_t> _counter{ 0 };
    return _counter;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::parent_type&
value_storage<Type>::get_upcast()
{
    return static_cast<parent_type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename value_storage<Type>::parent_type&
value_storage<Type>::get_upcast() const
{
    return static_cast<const parent_type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::parent_type*
value_storage<Type>::get_parent()
{
    return static_cast<parent_type*>(this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename value_storage<Type>::parent_type*
value_storage<Type>::get_parent() const
{
    return static_cast<const parent_type*>(this);
}
//
//======================================================================================//
//
//                              impl::value_storage<Type>
//                                  impl::storage_true
//
//======================================================================================//
//
template <typename Type>
value_storage<Type>::value_storage()
: base_type(parent_type::singleton_type::is_master_thread(), instance_count()++,
            demangle<Type>())
{
    if(m_settings->get_debug())
        printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);

    component::state<Type>::has_storage() = true;
    m_call_stack.set_primary(m_is_master);

    static std::atomic<int32_t> _skip_once{ 0 };
    if(_skip_once++ > 0)
    {
        // make sure all worker instances have a copy of the hash id and aliases
        auto _master = parent_type::singleton_type::master_instance();
        if(_master)
        {
            // _master->data_init();
            _master->m_call_stack.initialize_data();
            m_call_stack.set_parent(&_master->m_call_stack);
            graph_hash_map_t   _hash_ids     = *_master->get_hash_ids();
            graph_hash_alias_t _hash_aliases = *_master->get_hash_aliases();
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
value_storage<Type>::~value_storage()
{
    component::state<Type>::has_storage() = false;

    auto _debug = m_settings->get_debug();

    if(_debug)
        printf("[%s]> destructing @ %i...\n", m_label.c_str(), __LINE__);

    if(!m_is_master)
    {
        if(parent_type::singleton_type::master_instance())
        {
            if(_debug)
                printf("[%s]> merging into master @ %i...\n", m_label.c_str(), __LINE__);
            parent_type::singleton_type::master_instance()->merge(this);
        }
    }

    if(_debug)
        printf("[%s]> deleting graph data @ %i...\n", m_label.c_str(), __LINE__);

    m_call_stack.data().reset();

    if(_debug)
        printf("[%s]> storage destroyed @ %i...\n", m_label.c_str(), __LINE__);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::initialize()
{
    if(m_initialized)
        return;
    if(m_settings->get_debug())
        printf("[%s]> initializing...\n", m_label.c_str());
    m_initialized = true;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::finalize()
{
    if(m_finalized)
        return;

    if(!m_initialized)
        return;

    if(m_settings->get_debug())
        PRINT_HERE("[%s]> finalizing...", m_label.c_str());

    m_finalized            = true;
    worker_is_finalizing() = true;
    if(m_is_master)
        master_is_finalizing() = true;
    manager::instance()->is_finalizing(true);

    if(m_thread_init)
        operation::fini<Type>{ get_parent(),
                               operation::mode_constant<operation::fini_mode::thread>{} };

    if(m_is_master && m_global_init)
        operation::fini<Type>{ get_parent(),
                               operation::mode_constant<operation::fini_mode::global>{} };

    if(m_settings->get_debug())
        PRINT_HERE("[%s]> finalizing...", m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::stack_clear()
{
    if(m_settings->get_stack_clearing())
        m_call_stack.stack_clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
value_storage<Type>::global_init()
{
    if(!m_global_init)
    {
        return [&]() {
            m_global_init = true;
            if(!m_is_master && parent_type::master_instance())
                parent_type::master_instance()->global_init();
            if(m_is_master)
            {
                operation::init<Type>{
                    get_parent(), operation::mode_constant<operation::init_mode::global>{}
                };
            }
            return m_global_init;
        }();
    }
    return m_global_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
value_storage<Type>::thread_init()
{
    if(!m_thread_init && m_thread_tid == std::this_thread::get_id())
    {
        return [&]() {
            bool _global_init = global_init();
            consume_parameters(_global_init);
            m_thread_init = true;
            operation::init<Type>{
                get_parent(), operation::mode_constant<operation::init_mode::thread>{}
            };
            return m_thread_init;
        }();
    }
    return m_thread_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
value_storage<Type>::data_init()
{
    if(!m_data_init)
    {
        return [&]() {
            m_data_init = true;
            if(!m_is_master && parent_type::master_instance())
                parent_type::master_instance()->data_init();
            bool _global_init = global_init();
            bool _thread_init = thread_init();
            consume_parameters(_global_init, _thread_init);
            m_call_stack.initialize_data();
            return m_data_init;
        }();
    }
    return m_data_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::insert_init()
{
    consume_parameters(global_init(), thread_init(), data_init());
    // check this now to ensure everything is initialized
    initialize();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::string
value_storage<Type>::get_prefix(const graph_node& node)
{
    auto _ret = operation::decode<TIMEMORY_API>{}(m_hash_ids, m_hash_aliases, node.id());
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master && parent_type::singleton_type::master_instance())
        {
            auto _master = parent_type::singleton_type::master_instance();
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
value_storage<Type>::get_prefix(uint64_t id)
{
    auto _ret = get_hash_identifier(m_hash_ids, m_hash_aliases, id);
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master && parent_type::singleton_type::master_instance())
        {
            auto _master = parent_type::singleton_type::master_instance();
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
void
value_storage<Type>::merge()
{
    if(!m_is_master || !m_initialized)
        return;

    auto m_children = parent_type::singleton_type::children();
    if(m_children.empty())
        return;

    for(auto& itr : m_children)
        merge(itr);

    // create lock
    auto_lock_t _lk{ parent_type::singleton_type::get_mutex(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();

    for(auto& itr : m_children)
    {
        if(itr != this)
            itr->data().clear();
    }

    stack_clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::merge(this_type* itr)
{
    if(itr)
        operation::finalize::merge<Type, true>{ get_upcast(),
                                                static_cast<parent_type&>(*itr) };
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::result_vector_type
value_storage<Type>::get()
{
    result_vector_type _ret{};
    return get(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
value_storage<Type>::get(Tp& _ret)
{
    return operation::finalize::get<Type, true>{ get_upcast() }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::dmp_result_vector_type
value_storage<Type>::mpi_get()
{
    dmp_result_vector_type _ret{};
    return mpi_get(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
value_storage<Type>::mpi_get(Tp& _ret)
{
    return operation::finalize::mpi_get<Type, true>{ get_upcast() }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::dmp_result_vector_type
value_storage<Type>::upc_get()
{
    dmp_result_vector_type _ret{};
    return upc_get(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
value_storage<Type>::upc_get(Tp& _ret)
{
    return operation::finalize::upc_get<Type, true>{ get_upcast() }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::dmp_result_vector_type
value_storage<Type>::dmp_get()
{
    dmp_result_vector_type _ret;
    operation::finalize::dmp_get<Type, true>{ get_upcast() }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
value_storage<Type>::dmp_get(Tp& _ret)
{
    return operation::finalize::dmp_get<Type, true>{ get_upcast() }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::internal_print()
{
    base::storage::stop_profiler();

    if(!m_initialized && !m_finalized)
        return;

    if(!parent_type::singleton_type::is_master(get_parent()))
    {
        if(parent_type::singleton_type::master_instance())
            parent_type::singleton_type::master_instance()->merge(this);
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
        if(!m_call_stack.data())
        {
            instance_count().store(0);
            return;
        }

        // no entries
        if(m_call_stack.size() <= 1)
        {
            instance_count().store(0);
            return;
        }

        // generate output
        if(m_settings->get_auto_output())
        {
            m_printer.reset(new printer_t{ Type::get_label(), get_parent(), m_settings });

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
value_storage<Type>::get_shared_manager()
{
    using func_t = std::function<void()>;

    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        if(!m_manager)
            return;

        auto   _label = Type::label();
        size_t pos    = std::string::npos;
        // remove the namespaces
        for(auto itr : { "tim::component::", "tim::project::", "tim::tpls::",
                         "tim::api::", "tim::" })
        {
            while((pos = _label.find(itr)) != std::string::npos)
                _label = _label.replace(pos, std::string(itr).length(), "");
        }
        // replace spaces with underscores
        while((pos = _label.find(' ')) != std::string::npos)
            _label = _label.replace(pos, 1, "_");
        // convert to upper-case
        for(auto& itr : _label)
            itr = toupper(itr);
        for(auto itr : { ':', '<', '>' })
        {
            auto _pos = _label.find(itr);
            while(_pos != std::string::npos)
                _pos = _label.erase(_pos, 1).find(itr);
        }
        std::stringstream env_var;
        env_var << "TIMEMORY_" << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        trait::runtime_enabled<Type>::set(_enabled);

        auto _instance_id = m_instance_id;
        bool _is_master   = parent_type::singleton_type::is_master(get_parent());
        auto _sync        = [&]() {
            if(m_call_stack.data() != nullptr)
                data().sync_sea_level();
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
            auto _instance = parent_type::get_singleton();
            if(_instance)
            {
                auto _debug_v = m_settings->get_debug();
                auto _verb_v  = m_settings->get_verbose();
                if(_debug_v || _verb_v > 1)
                {
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling _instance->reset(this)");
                }
                _instance->reset(get_parent());
                // if(_debug_v || _verb_v > 1)
                //    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                //               "calling _instance->smart_instance().reset()");
                // _instance->smart_instance().reset();
                if(_is_master && _instance)
                {
                    if(_debug_v || _verb_v > 1)
                    {
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling _instance->reset()");
                    }
                    _instance->reset();
                    // _instance->smart_master_instance().reset();
                }
            }
            else
            {
                DEBUG_PRINT_HERE("[%s]> %p", demangle<Type>().c_str(), (void*) _instance);
            }
            if(_is_master)
                trait::runtime_enabled<Type>::set(false);
        };

        if(!_is_master)
        {
            manager::master_instance()->add_synchronization(
                demangle<Type>(), m_instance_id, std::move(_sync));
            m_manager->add_synchronization(demangle<Type>(), m_instance_id,
                                           std::move(_sync));
        }
        m_manager->add_finalizer(demangle<Type>(), std::move(_cleanup),
                                 std::move(_finalize), _is_master);
    }
}
//
}  // namespace impl
}  // namespace tim

#endif
