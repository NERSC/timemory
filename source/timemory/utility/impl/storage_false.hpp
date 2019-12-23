// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

/** \file timemory/utility/impl/storage_false.hpp
 * \headerfile utility/impl/storage_false.hpp "timemory/utility/impl/storage_false.hpp"
 * Defines storage implementation when the data type is void
 *
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/base_storage.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace impl
{
//======================================================================================//
//
//              Storage class for types that DO NOT use storage
//
//======================================================================================//

template <typename Type>
class storage<Type, false> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    using base_type     = base::storage;
    using this_type     = storage<Type, false>;
    using string_t      = std::string;
    using smart_pointer = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;

    friend class tim::manager;
    friend struct impl::storage_deleter<this_type>;

public:
    using iterator       = void*;
    using const_iterator = const void*;

    static pointer instance()
    {
        return get_singleton() ? get_singleton()->instance() : nullptr;
    }
    static pointer master_instance()
    {
        return get_singleton() ? get_singleton()->master_instance() : nullptr;
    }
    static pointer noninit_instance()
    {
        return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
    }
    static pointer noninit_master_instance()
    {
        return get_singleton() ? get_singleton()->master_instance_ptr() : nullptr;
    }

public:
    static bool& master_is_finalizing()
    {
        static bool _instance = false;
        return _instance;
    }

    static bool& worker_is_finalizing()
    {
        static thread_local bool _instance = master_is_finalizing();
        return _instance;
    }

    static bool is_finalizing()
    {
        return worker_is_finalizing() || master_is_finalizing();
    }

private:
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }

    static std::atomic<int64_t>& instance_count()
    {
        static std::atomic<int64_t> _counter(0);
        return _counter;
    }

public:
    //----------------------------------------------------------------------------------//
    //
    storage()
    : base_type(singleton_t::is_master_thread(), instance_count()++, Type::label())
    {
        if(settings::debug())
            printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);
        get_shared_manager();
        component::state<Type>::has_storage() = true;
    }

    //----------------------------------------------------------------------------------//
    //
    ~storage()
    {
        if(settings::debug())
            printf("[%s]> destructing @ %i...\n", m_label.c_str(), __LINE__);
    }

    //----------------------------------------------------------------------------------//
    //
    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;

    //----------------------------------------------------------------------------------//
    //
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

public:
    //----------------------------------------------------------------------------------//
    //
    virtual void print() final { finalize(); }

    virtual void cleanup() final { Type::cleanup(); }

    virtual void stack_clear() final
    {
        std::unordered_set<Type*> _stack = m_stack;
        for(auto& itr : _stack)
        {
            itr->stop();
            itr->pop_node();
        }
        m_stack.clear();
    }

    void initialize()
    {
        if(m_initialized)
            return;

        if(settings::debug())
            printf("[%s]> initializing...\n", m_label.c_str());

        m_initialized = true;

        if(!m_is_master)
        {
            Type::thread_init(this);
        }
        else
        {
            Type::global_init(this);
            Type::thread_init(this);
        }
    }

    void finalize() final
    {
        if(m_finalized)
            return;

        if(!m_initialized)
            return;

        if(settings::debug())
            printf("[%s]> finalizing...\n", m_label.c_str());

        m_finalized = true;
        if(!m_is_master)
        {
            worker_is_finalizing() = true;
            Type::thread_finalize(this);
        }
        else
        {
            master_is_finalizing() = true;
            worker_is_finalizing() = true;
            Type::thread_finalize(this);
            Type::global_finalize(this);
        }
    }

public:
    bool          empty() const { return true; }
    inline size_t size() const { return 0; }
    inline size_t depth() const { return 0; }

    iterator pop() { return nullptr; }
    iterator insert(int64_t, const Type&, const string_t&) { return nullptr; }

    template <typename _Archive>
    void serialize(_Archive&, const unsigned int)
    {}

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj)
    {
        auto itr = m_stack.find(obj);
        if(itr != m_stack.end())
        {
            m_stack.erase(itr);
        }
    }

protected:
    void get_shared_manager();

    void merge()
    {
        auto m_children = singleton_t::children();
        if(m_children.size() == 0)
            return;

        for(auto& itr : m_children)
            merge(itr);

        stack_clear();
    }

    void merge(this_type* itr)
    {
        itr->stack_clear();

        // create lock but don't immediately lock
        // auto_lock_t l(type_mutex<this_type>(), std::defer_lock);
        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

        // lock if not already owned
        if(!l.owns_lock())
            l.lock();

        for(const auto& _itr : (*itr->get_hash_ids()))
            if(m_hash_ids->find(_itr.first) == m_hash_ids->end())
                (*m_hash_ids)[_itr.first] = _itr.second;
        for(const auto& _itr : (*itr->get_hash_aliases()))
            if(m_hash_aliases->find(_itr.first) == m_hash_aliases->end())
                (*m_hash_aliases)[_itr.first] = _itr.second;
    }

private:
    template <typename _Archive>
    void _serialize(_Archive&)
    {}

private:
    std::unordered_set<Type*> m_stack;
};

//======================================================================================//

}  // namespace impl

}  // namespace tim
