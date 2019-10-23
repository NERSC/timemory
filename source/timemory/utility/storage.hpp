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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file storage.hpp
 * \headerfile storage.hpp "timemory/utility/storage.hpp"
 * Storage of the call-graph for each component. Each component has a thread-local
 * singleton that hold the call-graph. When a worker thread is deleted, it merges
 * itself back into the master thread storage. When the master thread is deleted,
 * it handles I/O (i.e. text file output, JSON output, stdout output).
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/gperf.hpp"
#include "timemory/backends/mpi.hpp"
#include "timemory/data/accumulators.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/utility/graph.hpp"
#include "timemory/utility/graph_data.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

// this is deprecated mostly because it is not extensible
#include "timemory/utility/type_id.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace tim
{
// clang-format off
namespace details
{
template <typename StorageType> struct storage_deleter;
template <typename ObjectType> class storage;
template <typename _Tp>
using storage_smart_pointer = std::unique_ptr<_Tp, details::storage_deleter<_Tp>>;
template <typename _Tp>
using storage_singleton_t = singleton<_Tp, storage_smart_pointer<_Tp>>;
}  // namespace details
namespace cupti { struct result; }
namespace impl  { template <typename ObjectType, bool IsAvailable> class storage {}; }
class manager;

namespace scope
{
struct flat    {};  // flat-scope storage
struct thread  {};  // thread-scoped storage
struct process {};  // process-scoped storage
} // namespace scope
// clang-format on

//--------------------------------------------------------------------------------------//

template <typename _Tp>
details::storage_singleton_t<_Tp>&
get_storage_singleton()
{
    using _single_t            = details::storage_singleton_t<_Tp>;
    static _single_t _instance = _single_t::instance();
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
details::storage_singleton_t<_Tp>&
get_noninit_storage_singleton()
{
    using _single_t            = details::storage_singleton_t<_Tp>;
    static _single_t _instance = _single_t::instance_ptr();
    return _instance;
}

//--------------------------------------------------------------------------------------//

namespace impl
{
//======================================================================================//
//
//              Storage class for types that implement it
//
//======================================================================================//

template <typename ObjectType>
class storage<ObjectType, true>
{
public:
    //----------------------------------------------------------------------------------//
    //
    using this_type     = storage<ObjectType, true>;
    using string_t      = std::string;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;

    using graph_node_tuple = std::tuple<int64_t, ObjectType, int64_t>;

    class graph_node : public graph_node_tuple
    {
    public:
        using this_type       = graph_node;
        using base_type       = graph_node_tuple;
        using data_value_type = typename ObjectType::value_type;
        using data_base_type  = typename ObjectType::base_type;
        using string_t        = std::string;

        int64_t&    id() { return std::get<0>(*this); }
        ObjectType& obj() { return std::get<1>(*this); }
        int64_t&    depth() { return std::get<2>(*this); }

        const int64_t&    id() const { return std::get<0>(*this); }
        const ObjectType& obj() const { return std::get<1>(*this); }
        const int64_t&    depth() const { return std::get<2>(*this); }

        graph_node()
        : base_type(0, ObjectType(), 0)
        {
        }

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {
        }

        graph_node(const int64_t& _id, const ObjectType& _obj, int64_t _depth)
        : base_type(_id, _obj, _depth)
        {
        }

        ~graph_node() {}
        // explicit graph_node(const this_type&) = default;
        // explicit graph_node(this_type&&)      = default;
        // graph_node& operator=(const this_type&) = default;
        // graph_node& operator=(this_type&&) = default;

        bool operator==(const graph_node& rhs) const
        {
            return (id() == rhs.id() && depth() == rhs.depth());
        }

        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            auto&       _obj = obj();
            const auto& _rhs = rhs.obj();
            static_cast<data_base_type&>(_obj) +=
                static_cast<const data_base_type&>(_rhs);
            return *this;
        }

        size_t data_size() const { return sizeof(ObjectType) + 2 * sizeof(int64_t); }
    };

public:
    using graph_node_t   = graph_node;
    using graph_data_t   = graph_data<graph_node_t>;
    using graph_t        = typename graph_data_t::graph_t;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;

    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }
    static pointer noninit_instance() { return get_noninit_singleton().instance(); }
    static pointer noninit_master_instance()
    {
        return get_noninit_singleton().master_instance();
    }

public:
    //----------------------------------------------------------------------------------//
    //
    storage()
    {
        if(settings::debug())
            printf("[%s]> constructing @ %i...\n", ObjectType::label().c_str(), __LINE__);

        component::properties<ObjectType>::has_storage() = true;
        // check_consistency();
    }

    //----------------------------------------------------------------------------------//
    //
    ~storage()
    {
        if(settings::debug())
            printf("[%s]> destructing @ %i...\n", ObjectType::label().c_str(), __LINE__);

        if(!singleton_t::is_master(this))
            singleton_t::master_instance()->merge(this);
        delete m_graph_data_instance;
        m_graph_data_instance = nullptr;
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
    void initialize()
    {
        if(m_initialized)
            return;

        if(settings::debug())
            printf("[%s]> initializing...\n", ObjectType::label().c_str());

        m_initialized = true;

        // global_init();
    }

    //----------------------------------------------------------------------------------//
    //
    void finalize()
    {
        if(m_finalized)
            return;

        if(!m_initialized)
            return;

        if(settings::debug())
            printf("[%s]> finalizing...\n", ObjectType::label().c_str());

        m_finalized = true;
        if(!singleton_t::is_master(this))
        {
            if(m_thread_init)
                ObjectType::thread_finalize_policy(this);
        }
        else
        {
            if(m_thread_init)
                ObjectType::thread_finalize_policy(this);
            if(m_global_init)
                ObjectType::global_finalize_policy(this);
        }
    }

    bool    is_initialized() const { return m_initialized; }
    int64_t instance_id() const { return m_instance_id; }

    //----------------------------------------------------------------------------------//
    //
    bool global_init()
    {
        static auto _lambda = [&]() {
            if(!singleton_t::is_master(this))
                master_instance()->global_init();
            if(singleton_t::is_master(this))
                ObjectType::global_init_policy(this);
            m_global_init = true;
            return m_global_init;
        };
        if(!m_global_init)
            return _lambda();
        return m_global_init;
    }

    //----------------------------------------------------------------------------------//
    //
    bool thread_init()
    {
        static auto _lambda = [&]() {
            if(!singleton_t::is_master(this))
                master_instance()->thread_init();
            bool _global_init = global_init();
            consume_parameters(_global_init);
            ObjectType::thread_init_policy(this);
            m_thread_init = true;
            return m_thread_init;
        };
        if(!m_thread_init)
            return _lambda();
        return m_thread_init;
    }

    //----------------------------------------------------------------------------------//
    //
    bool data_init()
    {
        static auto _lambda = [&]() {
            if(!singleton_t::is_master(this))
                master_instance()->data_init();
            bool _global_init = global_init();
            bool _thread_init = thread_init();
            consume_parameters(_global_init, _thread_init);
            check_consistency();
            m_data_init = true;
            return m_data_init;
        };
        if(!m_data_init)
            return _lambda();
        return m_data_init;
    }

private:
    void check_consistency()
    {
        // auto_lock_t lk(type_mutex<this_type>(), std::defer_lock);
        // if(!lk.owns_lock())
        //    lk.lock();

        auto* ptr = &_data();
        if(ptr != m_graph_data_instance)
        {
            fprintf(stderr, "[%s]> mismatched graph data on master thread: %p vs. %p\n",
                    ObjectType::label().c_str(), (void*) ptr,
                    (void*) m_graph_data_instance);
        }
    }

public:
    //----------------------------------------------------------------------------------//
    // there is always a head node that should not be counted
    //
    bool empty() const { return (_data().graph().size() <= 1); }

    //----------------------------------------------------------------------------------//
    // there is always a head node that should not be counted
    //
    inline size_t size() const { return _data().graph().size() - 1; }

    //----------------------------------------------------------------------------------//
    //
    iterator pop() { return _data().pop_graph(); }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope  = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::process>::value ||
                           std::is_same<_Scope, scope::thread>::value),
                          int> = 0>
    iterator insert(int64_t hash_id, const ObjectType& obj, int64_t hash_depth)
    {
        // check this now to ensure everything is initialized
        if(m_node_ids.size() == 0 || m_graph_data_instance == nullptr)
            initialize();
        bool _has_head = _data().has_head();

        // if first instance
        if(!_has_head || (this == master_instance() && m_node_ids.size() == 0))
        {
            graph_node_t node(hash_id, obj, hash_depth);
            if(this == master_instance())
            {
                _data()         = graph_data_t(node);
                _data().depth() = 0;
                if(m_node_ids.size() == 0)
                    m_node_ids[0][0] = _data().current();
                m_node_ids[hash_depth][hash_id] = _data().current();
                return _data().current();
            }
            else
            {
                _data()           = graph_data_t(*master_instance()->current());
                _data().head()    = master_instance()->data().current();
                _data().current() = master_instance()->data().current();
                _data().depth()   = master_instance()->data().depth();
                auto itr          = _data().append_child(node);
                m_node_ids[hash_depth][hash_id] = itr;
                return itr;
            }
        }

        // lambda for updating settings
        auto _update = [&](iterator itr) {
            _data().depth() = itr->depth();
            return (_data().current() = itr);
        };

        if(m_node_ids[hash_depth].find(hash_id) != m_node_ids[hash_depth].end() &&
           m_node_ids[hash_depth].find(hash_id)->second->depth() == _data().depth())
        {
            return _update(m_node_ids[hash_depth].find(hash_id)->second);
        }

        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node_t node(hash_id, obj, _data().depth());

        // lambda for inserting child
        auto _insert_child = [&]() {
            node.depth()                    = hash_depth;
            auto itr                        = _data().append_child(node);
            m_node_ids[hash_depth][hash_id] = itr;
            // if(m_node_ids[hash_depth].bucket_count() < m_node_ids[hash_depth].size())
            //    m_node_ids[hash_depth].rehash(m_node_ids[hash_depth].size() + 10);
            return itr;
        };

        auto current = _data().current();
        if(!_data().graph().is_valid(current))
            _insert_child();

        // auto nchildren = graph().number_of_children(current);
        // auto nsiblings = graph().number_of_siblings(current);

        if((hash_id) == current->id())
        {
            return current;
        }
        else if(_data().graph().is_valid(current))
        {
            // check siblings
            for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
            {
                // skip if current
                if(itr == current)
                    continue;
                // check hash id's
                if((hash_id) == itr->id())
                    return _update(itr);
            }

            // check children
            // if(nchildren == 0)
            //    return _insert_child();
            // else
            {
                // check child
                auto fchild = graph_t::child(current, 0);
                if(_data().graph().is_valid(fchild))
                {
                    for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                    {
                        if((hash_id) == itr->id())
                            return _update(itr);
                    }
                }
            }
        }

        return _insert_child();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::flat>::value), int> = 0>
    iterator insert(int64_t hash_id, const ObjectType& obj, int64_t hash_depth)
    {
        // check this now to ensure everything is initialized
        if(m_node_ids.size() == 0 || m_graph_data_instance == nullptr)
            initialize();
        bool _has_head = _data().has_head();

        // if first instance
        if(!_has_head || (this == master_instance() && m_node_ids.size() == 0))
        {
            graph_node_t node(hash_id, obj, hash_depth);
            if(this == master_instance())
            {
                _data()         = graph_data_t(node);
                _data().depth() = 0;
                if(m_node_ids.size() == 0)
                    m_node_ids[0][0] = _data().current();
                m_node_ids[hash_depth][hash_id] = _data().current();
                return _data().current();
            }
            else
            {
                _data()           = graph_data_t(*master_instance()->current());
                _data().head()    = master_instance()->data().current();
                _data().current() = master_instance()->data().current();
                _data().depth()   = master_instance()->data().depth();
                auto itr          = _data().append_child(node);
                m_node_ids[hash_depth][hash_id] = itr;
                return itr;
            }
        }

        // lambda for updating settings
        auto _update = [&](iterator itr) { return itr; };

        if(m_node_ids[hash_depth].find(hash_id) != m_node_ids[hash_depth].end() &&
           m_node_ids[hash_depth].find(hash_id)->second->depth() == _data().depth())
        {
            return _update(m_node_ids[hash_depth].find(hash_id)->second);
        }

        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node_t node(hash_id, obj, hash_depth);

        // lambda for inserting child
        auto _insert_head = [&]() {
            node.depth()                    = hash_depth;
            auto itr                        = _data().append_head(node);
            m_node_ids[hash_depth][hash_id] = itr;
            // if(m_node_ids[hash_depth].bucket_count() < m_node_ids[hash_depth].size())
            //    m_node_ids[hash_depth].rehash(m_node_ids[hash_depth].size() + 10);
            return itr;
        };

        auto current   = _data().head();
        auto nchildren = graph_t::number_of_children(current);

        if(nchildren == 0 && graph().number_of_siblings(current) == 0)
            return _insert_head();
        else if(_data().graph().is_valid(current))
        {
            // check siblings
            for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
            {
                // skip if current
                if(itr == current)
                    continue;
                // check hash id's
                if((hash_id) == itr->id())
                    return _update(itr);
            }

            // check children
            if(nchildren == 0)
                return _insert_head();
            else
            {
                // check child
                auto fchild = graph_t::child(current, 0);
                if(_data().graph().is_valid(fchild))
                {
                    for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                    {
                        if((hash_id) == itr->id())
                            return _update(itr);
                    }
                }
            }
        }
        return _insert_head();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope  = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::process>::value ||
                           std::is_same<_Scope, scope::thread>::value),
                          int> = 0>
    iterator insert(const ObjectType& obj, int64_t hash_id)
    {
        static bool _global_init = global_init();
        static bool _thread_init = thread_init();
        static bool _data_init   = data_init();
        consume_parameters(_global_init, _thread_init, _data_init);

        auto hash_depth = ((_data().depth() >= 0) ? (_data().depth() + 1) : 1);
        auto itr        = insert<_Scope>(hash_id * hash_depth, obj, hash_depth);
        add_hash_id(hash_id, hash_id * hash_depth);
        add_hash_id(hash_id * hash_depth, hash_id);
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::flat>::value), int> = 0>
    iterator insert(const ObjectType& obj, int64_t hash_id)
    {
        static bool _global_init = global_init();
        static bool _thread_init = thread_init();
        static bool _data_init   = data_init();
        consume_parameters(_global_init, _thread_init, _data_init);

        auto itr = insert<_Scope>(hash_id, obj, 1);
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    void print()
    {
        typename trait::external_output_handling<ObjectType>::type type;
        external_print(type);
    }

private:
    string_t get_prefix(const graph_node& node) { return get_hash_identifier(node.id()); }
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }

public:
    //----------------------------------------------------------------------------------//
    //
    const graph_data_t& data() const { return _data(); }
    const graph_t&      graph() const { return _data().graph(); }
    const int64_t&      depth() const { return _data().depth(); }

    //----------------------------------------------------------------------------------//
    //
    graph_data_t& data() { return _data(); }
    iterator&     current() { return _data().current(); }
    graph_t&      graph() { return _data().graph(); }

protected:
    friend struct details::storage_deleter<this_type>;

    void merge()
    {
        if(!singleton_t::is_master(this) || !m_initialized)
            return;

        auto m_children = singleton_t::children();
        if(m_children.size() == 0)
            return;

        for(auto& itr : m_children)
            merge(itr);

        // create lock but don't immediately lock
        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

        // lock if not already owned
        if(!l.owns_lock())
            l.lock();

        for(auto& itr : m_children)
            if(itr != this)
                itr->data().clear();
    }

    void merge(this_type* itr);
    void mpi_reduce();

protected:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Tp>
    struct write_serialization
    {
        template <typename _Up>
        struct is_enabled
        {
            using _Vp = typename _Up::value_type;
            static constexpr bool value =
                (trait::is_available<_Up>::value &&
                 !(trait::external_output_handling<_Up>::value) &&
                 !(std::is_same<_Vp, void>::value));
        };

        using storage_t = this_type;

        template <typename _Archive, typename _Type = ObjectType,
                  typename std::enable_if<(is_enabled<_Type>::value), char>::type = 0>
        static void serialize(storage_t& _obj, _Archive& ar, const unsigned int version)
        {
            typename tim::trait::array_serialization<ObjectType>::type type;
            _obj.serialize_me(type, ar, version);
        }

        template <typename _Archive, typename _Type = ObjectType,
                  typename std::enable_if<!(is_enabled<_Type>::value), char>::type = 0>
        static void serialize(storage_t&, _Archive&, const unsigned int)
        {
        }
    };

    friend struct write_serialization<this_type>;

public:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Archive>
    void serialize(_Archive& ar, const unsigned int version)
    {
        write_serialization<this_type>::serialize(*this, ar, version);
    }

private:
    //----------------------------------------------------------------------------------//
    //
    friend class tim::manager;

    template <typename _Archive>
    void _serialize(_Archive& ar)
    {
        auto _label = ObjectType::label();
        if(singleton_t::is_master(this))
            merge();
        ar(cereal::make_nvp(_label, *this));
    }

private:
    static singleton_t& get_singleton() { return get_storage_singleton<this_type>(); }
    static singleton_t& get_noninit_singleton()
    {
        return get_noninit_storage_singleton<this_type>();
    }

    static std::atomic<int64_t>& instance_count()
    {
        static std::atomic<int64_t> _counter;
        return _counter;
    }

    // tim::trait::array_serialization<ObjectType>::type == TRUE
    template <typename Archive>
    void serialize_me(std::true_type, Archive&, const unsigned int);

    // tim::trait::array_serialization<ObjectType>::type == FALSE
    template <typename Archive>
    void serialize_me(std::false_type, Archive&, const unsigned int);

    // tim::trait::external_output_handling<ObjectType>::type == TRUE
    void external_print(std::true_type);

    // tim::trait::external_output_handling<ObjectType>::type == FALSE
    void external_print(std::false_type);

    graph_data_t& _data()
    {
        if(m_graph_data_instance == nullptr && !singleton_t::is_master(this))
        {
            static bool _data_init = master_instance()->data_init();
            consume_parameters(_data_init);

            m_graph_data_instance = new graph_data_t(*master_instance()->current());
            m_graph_data_instance->head()    = master_instance()->current();
            m_graph_data_instance->current() = master_instance()->current();
            m_graph_data_instance->depth()   = master_instance()->depth();
        }
        else if(m_graph_data_instance == nullptr)
        {
            auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);
            if(!lk.owns_lock())
                lk.lock();
            /*static std::recursive_mutex _init_mutex;
            auto_lock_t lk(_init_mutex, std::defer_lock);
            if(!lk.owns_lock())
                lk.lock();*/

            using base_type     = typename ObjectType::base_type;
            std::string _prefix = "> [tot] total";
            add_hash_id(m_hash_ids, _prefix);
            graph_node_t node(0, base_type::dummy(), 0);
            m_graph_data_instance          = new graph_data_t(node);
            m_graph_data_instance->depth() = 0;
            if(m_node_ids.size() == 0)
                m_node_ids[0][0] = m_graph_data_instance->current();
        }
        return *m_graph_data_instance;
    }

    const graph_data_t& _data() const { return const_cast<this_type*>(this)->_data(); }

    template <typename _Key_t, typename _Mapped_t>
    using uomap_t             = std::unordered_map<_Key_t, _Mapped_t>;
    using iterator_hash_map_t = uomap_t<int64_t, uomap_t<int64_t, iterator>>;

    bool                          m_initialized         = false;
    bool                          m_finalized           = false;
    bool                          m_global_init         = false;
    bool                          m_thread_init         = false;
    bool                          m_data_init           = false;
    bool                          m_node_init           = mpi::is_initialized();
    int32_t                       m_node_rank           = mpi::rank();
    int32_t                       m_node_size           = mpi::size();
    int64_t                       m_instance_id         = instance_count()++;
    ::tim::graph_hash_map_ptr_t   m_hash_ids            = ::tim::get_hash_ids();
    ::tim::graph_hash_alias_ptr_t m_hash_aliases        = ::tim::get_hash_aliases();
    mutable graph_data_t*         m_graph_data_instance = nullptr;
    iterator_hash_map_t           m_node_ids;

public:
    const graph_hash_map_ptr_t&   get_hash_ids() const { return m_hash_ids; }
    const graph_hash_alias_ptr_t& get_hash_aliases() const { return m_hash_aliases; }
    const iterator_hash_map_t     get_node_ids() const { return m_node_ids; }
};

//======================================================================================//
//
//              Storage class for types that DO NOT use storage
//
//======================================================================================//

template <typename ObjectType>
class storage<ObjectType, false>
{
public:
    //----------------------------------------------------------------------------------//
    //
    using this_type     = storage<ObjectType, false>;
    using string_t      = std::string;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;

public:
    using iterator       = void*;
    using const_iterator = const void*;

    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }
    static pointer noninit_instance() { return get_noninit_singleton().instance(); }
    static pointer noninit_master_instance()
    {
        return get_noninit_singleton().master_instance();
    }

public:
    //----------------------------------------------------------------------------------//
    //
    storage()
    {
        if(settings::debug())
            printf("[%s]> constructing @ %i...\n", ObjectType::label().c_str(), __LINE__);

        component::properties<ObjectType>::has_storage() = false;
    }

    //----------------------------------------------------------------------------------//
    //
    ~storage()
    {
        if(settings::debug())
            printf("[%s]> destructing @ %i...\n", ObjectType::label().c_str(), __LINE__);
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
    void initialize()
    {
        if(m_initialized)
            return;

        if(settings::debug())
            printf("[%s]> initializing...\n", ObjectType::label().c_str());

        m_initialized = true;

        if(!singleton_t::is_master(this))
        {
            ObjectType::thread_init_policy(this);
        }
        else
        {
            ObjectType::global_init_policy(this);
            ObjectType::thread_init_policy(this);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    void finalize()
    {
        if(m_finalized)
            return;

        if(!m_initialized)
            return;

        if(settings::debug())
            printf("[%s]> finalizing...\n", ObjectType::label().c_str());

        m_finalized = true;
        if(!singleton_t::is_master(this))
        {
            ObjectType::thread_finalize_policy(this);
        }
        else
        {
            ObjectType::thread_finalize_policy(this);
            ObjectType::global_finalize_policy(this);
        }
    }

    //----------------------------------------------------------------------------------//
    //  query status properties
    //
    bool    is_initialized() const { return m_initialized; }
    bool    is_finalized() const { return m_finalized; }
    int64_t instance_id() const { return m_instance_id; }

public:
    //----------------------------------------------------------------------------------//
    // there is always a head node that should not be counted
    //
    bool empty() const { return true; }

    //----------------------------------------------------------------------------------//
    // there is always a head node that should not be counted
    //
    inline size_t size() const { return 0; }

    //----------------------------------------------------------------------------------//
    //
    iterator pop() { return nullptr; }

    //----------------------------------------------------------------------------------//
    //
    iterator insert(int64_t, const ObjectType&, const string_t&) { return nullptr; }
    void     set_prefix(const string_t&) {}
    void     print()
    {
        if(m_initialized)
        {
            ObjectType::thread_finalize_policy(this);
            if(singleton_t::is_master(this))
                ObjectType::global_finalize_policy(this);
            finalize();
        }
    }

protected:
    friend struct details::storage_deleter<this_type>;
    void merge() {}
    void merge(this_type*) {}

public:
    template <typename _Archive>
    void serialize(_Archive&, const unsigned int)
    {
    }

private:
    friend class tim::manager;

    template <typename _Archive>
    void _serialize(_Archive&)
    {
    }

private:
    static singleton_t& get_singleton() { return get_storage_singleton<this_type>(); }
    static singleton_t& get_noninit_singleton()
    {
        return get_noninit_storage_singleton<this_type>();
    }

    static std::atomic<int64_t>& instance_count()
    {
        static std::atomic<int64_t> _counter;
        return _counter;
    }

    bool    m_initialized = false;
    bool    m_finalized   = false;
    int64_t m_instance_id = instance_count()++;
};

//======================================================================================//

}  // namespace impl

//======================================================================================//
//
//      determines if storage should be implemented
//
//======================================================================================//

template <typename _Tp, typename _Vp = typename _Tp::value_type>
struct implements_storage
{
    static constexpr bool value = (trait::is_available<_Tp>::value &&
                                   !(trait::external_output_handling<_Tp>::value) &&
                                   !(std::is_same<_Vp, void>::value));
};

//======================================================================================//

template <typename _Tp>
class storage
: public impl::storage<_Tp, implements_storage<_Tp, typename _Tp::value_type>::value>
{
    using this_type = storage<_Tp>;
    using base_type =
        impl::storage<_Tp, implements_storage<_Tp, typename _Tp::value_type>::value>;
    using string_t      = std::string;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;
    friend struct details::storage_deleter<this_type>;
    friend class manager;
};

//--------------------------------------------------------------------------------------//
/// args:
///     1) filename
///     2) reference to storage object
///     3) concurrency
///
template <typename _Tp>
void
serialize_storage(const std::string&, const _Tp&, int64_t = 1, int64_t = mpi::rank());

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

template <typename StorageType>
struct tim::details::storage_deleter : public std::default_delete<StorageType>
{
    using Pointer     = std::unique_ptr<StorageType, storage_deleter<StorageType>>;
    using singleton_t = tim::singleton<StorageType, Pointer>;

    void operator()(StorageType* ptr)
    {
        StorageType*    master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        if(ptr && master && ptr != master)
        {
            master->StorageType::merge(ptr);
        }
        else
        {
            if(ptr)
            {
                ptr->StorageType::print();
            }
            else if(master)
            {
                master->StorageType::print();
            }
        }

        if(this_tid == master_tid)
        {
            delete ptr;
        }
        else
        {
            if(master && ptr != master)
            {
                singleton_t::remove(ptr);
            }
            delete ptr;
        }
    }
};

//======================================================================================//
