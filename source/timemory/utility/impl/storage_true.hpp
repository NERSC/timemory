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

/** \file timemory/utility/impl/storage_true.hpp
 * \headerfile utility/impl/storage_true.hpp "timemory/utility/impl/storage_true.hpp"
 * Defines storage implementation when the data type is not void
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/dmp.hpp"
#include "timemory/mpl/bits/operations.hpp"
#include "timemory/mpl/impl/math.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/base_storage.hpp"
#include "timemory/utility/graph.hpp"
#include "timemory/utility/graph_data.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tim
{
namespace impl
{
template <typename StorageType, typename Type,
          typename _HashMap   = typename StorageType::iterator_hash_map_t,
          typename _GraphData = typename StorageType::graph_data_t>
typename StorageType::iterator
insert_heirarchy(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                 _HashMap& m_node_ids, _GraphData*& m_data, bool _has_head,
                 bool _is_master);

//======================================================================================//
//
//              Storage class for types that implement it
//
//======================================================================================//

template <typename Type>
class storage<Type, true> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //  forward decl of some internal types
    //
    struct result_node;
    struct graph_node;
    friend struct result_node;
    friend struct graph_node;

protected:
    template <typename _Tp>
    struct write_serialization;

public:
    //----------------------------------------------------------------------------------//
    //
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, true>;
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using node_tuple_t   = std::tuple<uint64_t, Type, int64_t>;
    using result_array_t = std::vector<result_node>;
    using dmp_result_t   = std::vector<result_array_t>;
    using strvector_t    = std::vector<string_t>;
    using result_tuple_t =
        std::tuple<uint64_t, Type, string_t, int64_t, uint64_t, strvector_t>;

    friend struct impl::storage_deleter<this_type>;
    friend struct write_serialization<this_type>;
    friend class tim::manager;

public:
    // static functions
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

    //----------------------------------------------------------------------------------//
    //
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
    //      Result returned from get()
    //
    //----------------------------------------------------------------------------------//
    struct result_node : public result_tuple_t
    {
        using base_type = result_tuple_t;

        result_node() = default;
        result_node(base_type&& _base)
        : result_tuple_t(std::forward<base_type>(_base))
        {}
        ~result_node()                  = default;
        result_node(const result_node&) = default;
        result_node(result_node&&)      = default;
        result_node& operator=(const result_node&) = default;
        result_node& operator=(result_node&&) = default;

        uint64_t&    hash() { return std::get<0>(*this); }
        Type&        data() { return std::get<1>(*this); }
        string_t&    prefix() { return std::get<2>(*this); }
        int64_t&     depth() { return std::get<3>(*this); }
        uint64_t&    rolling_hash() { return std::get<4>(*this); }
        strvector_t& hierarchy() { return std::get<5>(*this); }

        const uint64_t&    hash() const { return std::get<0>(*this); }
        const Type&        data() const { return std::get<1>(*this); }
        const string_t&    prefix() const { return std::get<2>(*this); }
        const int64_t&     depth() const { return std::get<3>(*this); }
        const uint64_t&    rolling_hash() const { return std::get<4>(*this); }
        const strvector_t& hierarchy() const { return std::get<5>(*this); }
    };

    //----------------------------------------------------------------------------------//
    //
    //      Storage type in graph
    //
    //----------------------------------------------------------------------------------//
    struct graph_node : public node_tuple_t
    {
        using this_type       = graph_node;
        using base_type       = node_tuple_t;
        using data_value_type = typename Type::value_type;
        using data_base_type  = typename Type::base_type;
        using string_t        = std::string;

        uint64_t& id() { return std::get<0>(*this); }
        Type&     obj() { return std::get<1>(*this); }
        int64_t&  depth() { return std::get<2>(*this); }

        const uint64_t& id() const { return std::get<0>(*this); }
        const Type&     obj() const { return std::get<1>(*this); }
        const int64_t&  depth() const { return std::get<2>(*this); }

        string_t get_prefix() const { return master_instance()->get_prefix(*this); }

        graph_node()
        : base_type(0, Type(), 0)
        {}

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {}

        graph_node(const uint64_t& _id, const Type& _obj, int64_t _depth)
        : base_type(_id, _obj, _depth)
        {}

        ~graph_node() {}

        bool operator==(const graph_node& rhs) const
        {
            return (id() == rhs.id() && depth() == rhs.depth());
        }

        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            auto&       _obj = obj();
            const auto& _rhs = rhs.obj();
            _obj += _rhs;
            _obj.plus(_rhs);
            return *this;
        }

        size_t data_size() const { return sizeof(Type) + 2 * sizeof(int64_t); }

        friend std::ostream& operator<<(std::ostream& os, const graph_node& obj)
        {
            std::stringstream ss;
            auto              _prefix = obj.get_prefix();
            static auto       _w      = _prefix.length();
            _w                        = std::max(_w, _prefix.length());
            ss << "id = " << std::setw(24) << obj.id() << ", depth = " << std::setw(4)
               << obj.depth() << ", label = " << std::setw(_w) << std::left
               << obj.get_prefix();
            os << ss.str();
            return os;
        }
    };

public:
    using graph_node_t   = graph_node;
    using graph_data_t   = graph_data<graph_node_t>;
    using graph_t        = typename graph_data_t::graph_t;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;
    template <typename _Vp>
    using secondary_data_t = std::tuple<iterator, const std::string&, _Vp>;
    template <typename _Key_t, typename _Mapped_t>
    using uomap_t             = std::unordered_map<_Key_t, _Mapped_t>;
    using iterator_hash_map_t = uomap_t<int64_t, uomap_t<int64_t, iterator>>;

public:
    //----------------------------------------------------------------------------------//
    //
    storage()
    : base_type(singleton_t::is_master_thread(), instance_count()++, Type::label())
    {
        if(settings::debug())
            printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);

        component::state<Type>::has_storage() = true;

        static std::atomic<int32_t> _skip_once(0);
        if(_skip_once++ > 0)
        {
            // make sure all worker instances have a copy of the hash id and aliases
            auto               _master       = singleton_t::master_instance();
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

        get_shared_manager();
    }

    //----------------------------------------------------------------------------------//
    //
    ~storage()
    {
        if(settings::debug())
            printf("[%s]> destructing @ %i...\n", m_label.c_str(), __LINE__);

        if(!m_is_master)
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

    //----------------------------------------------------------------------------------//
    //  cleanup function for object
    //

public:
    virtual void print() final { internal_print(); }

    virtual void cleanup() final { Type::cleanup(); }

    void get_shared_manager();

    virtual void initialize()
    {
        if(m_initialized)
            return;
        if(settings::debug())
            printf("[%s]> initializing...\n", m_label.c_str());
        m_initialized = true;
    }

    virtual void finalize()
    {
        if(m_finalized)
            return;

        if(!m_initialized)
            return;

        if(settings::debug())
            PRINT_HERE("[%s]> finalizing...", m_label.c_str());

        m_finalized            = true;
        worker_is_finalizing() = true;
        if(m_is_master)
            master_is_finalizing() = true;

        if(m_thread_init)
            Type::thread_finalize(this);

        if(m_is_master && m_global_init)
            Type::global_finalize(this);

        if(settings::debug())
            PRINT_HERE("[%s]> finalizing...", m_label.c_str());
    }

    void stack_clear()
    {
        std::unordered_set<Type*> _stack = m_stack;
        for(auto& itr : _stack)
        {
            itr->stop();
            itr->pop_node();
        }
        m_stack.clear();
    }

    //----------------------------------------------------------------------------------//
    //
    virtual bool global_init() final
    {
        static auto _lambda = [&]() {
            if(!m_is_master)
                master_instance()->global_init();
            if(m_is_master)
                Type::global_init(this);
            m_global_init = true;
            return m_global_init;
        };
        if(!m_global_init)
            return _lambda();
        return m_global_init;
    }

    //----------------------------------------------------------------------------------//
    //
    virtual bool thread_init() final
    {
        static auto _lambda = [&]() {
            if(!m_is_master)
                master_instance()->thread_init();
            bool _global_init = global_init();
            consume_parameters(_global_init);
            Type::thread_init(this);
            m_thread_init = true;
            return m_thread_init;
        };
        if(!m_thread_init)
            return _lambda();
        return m_thread_init;
    }

    //----------------------------------------------------------------------------------//
    //
    virtual bool data_init() final
    {
        static auto _lambda = [&]() {
            if(!m_is_master)
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

    //----------------------------------------------------------------------------------//
    //
    const graph_data_t& data() const { return _data(); }
    const graph_t&      graph() const { return _data().graph(); }
    int64_t             depth() const { return (is_finalizing()) ? 0 : _data().depth(); }

    //----------------------------------------------------------------------------------//
    //
    graph_data_t& data() { return _data(); }
    iterator&     current() { return _data().current(); }
    graph_t&      graph() { return _data().graph(); }

    //----------------------------------------------------------------------------------//
    //
    inline bool     empty() const { return (_data().graph().size() <= 1); }
    inline size_t   size() const { return _data().graph().size() - 1; }
    inline iterator pop() { return _data().pop_graph(); }

    result_array_t get();
    dmp_result_t   mpi_get();
    dmp_result_t   upc_get();
    dmp_result_t   dmp_get()
    {
        auto fallback_get = [&]() { return dmp_result_t(1, get()); };

#if defined(TIMEMORY_USE_UPCXX) && defined(TIMEMORY_USE_MPI)
        return (mpi::is_initialized())
                   ? mpi_get()
                   : ((upc::is_initialized()) ? upc_get() : fallback_get());
#elif defined(TIMEMORY_USE_UPCXX)
        return (upc::is_initialized()) ? upc_get() : fallback_get();
#elif defined(TIMEMORY_USE_MPI)
        return (mpi::is_initialized()) ? mpi_get() : fallback_get();
#else
        return fallback_get();
#endif
    }

    const iterator_hash_map_t get_node_ids() const { return m_node_ids; }

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj)
    {
        auto itr = m_stack.find(obj);
        if(itr != m_stack.end())
            m_stack.erase(itr);
    }

private:
    void check_consistency()
    {
        auto* ptr = &_data();
        if(ptr != m_graph_data_instance)
        {
            fprintf(stderr, "[%s]> mismatched graph data on master thread: %p vs. %p\n",
                    m_label.c_str(), (void*) ptr,
                    static_cast<void*>(m_graph_data_instance));
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope  = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::process>::value ||
                           std::is_same<_Scope, scope::thread>::value),
                          int> = 0>
    iterator insert(uint64_t hash_id, const Type& obj, uint64_t hash_depth)
    {
        // check this now to ensure everything is initialized
        if(m_node_ids.size() == 0 || m_graph_data_instance == nullptr)
            initialize();
        bool _has_head = _data().has_head();

        return insert_heirarchy<this_type, Type>(hash_id, obj, hash_depth, m_node_ids,
                                                 m_graph_data_instance, _has_head,
                                                 m_is_master);
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::flat>::value), int> = 0>
    iterator insert(uint64_t hash_id, const Type& obj, uint64_t hash_depth)
    {
        // check this now to ensure everything is initialized
        if(m_node_ids.size() == 0 || m_graph_data_instance == nullptr)
            initialize();

        static thread_local auto _current = _data().head();
        static thread_local bool _first   = true;
        if(_first)
        {
            _first = false;
            if(_current.begin())
                _current = _current.begin();
            else
            {
                graph_node_t node(hash_id, obj, hash_depth);
                auto         itr                = _data().emplace_child(_current, node);
                m_node_ids[hash_depth][hash_id] = itr;
                _current                        = itr;
                return itr;
            }
        }

        auto _existing = m_node_ids[hash_depth].find(hash_id);
        if(_existing != m_node_ids[hash_depth].end())
            return m_node_ids[hash_depth].find(hash_id)->second;

        graph_node_t node(hash_id, obj, hash_depth);
        auto         itr                = _data().emplace_child(_current, node);
        m_node_ids[hash_depth][hash_id] = itr;
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope  = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::process>::value ||
                           std::is_same<_Scope, scope::thread>::value),
                          int> = 0>
    iterator insert(const Type& obj, uint64_t hash_id)
    {
        static bool _global_init = global_init();
        static bool _thread_init = thread_init();
        static bool _data_init   = data_init();
        consume_parameters(_global_init, _thread_init, _data_init);

        auto hash_depth = ((_data().depth() >= 0) ? (_data().depth() + 1) : 1);
        auto itr        = insert<_Scope>(hash_id * hash_depth, obj, hash_depth);
        add_hash_id(hash_id, hash_id * hash_depth);
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope = scope::process,
              enable_if_t<(std::is_same<_Scope, scope::flat>::value), int> = 0>
    iterator insert(const Type& obj, uint64_t hash_id)
    {
        static bool _global_init = global_init();
        static bool _thread_init = thread_init();
        static bool _data_init   = data_init();
        consume_parameters(_global_init, _thread_init, _data_init);

        // auto hash_depth = ((_data().depth() >= 0) ? (_data().depth() + 1) : 1);
        uint64_t hash_depth = 1;
        auto     itr        = insert<_Scope>(hash_id * hash_depth, obj, hash_depth);
        add_hash_id(hash_id, hash_id * hash_depth);
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Vp>
    void append(const secondary_data_t<_Vp>& _secondary)
    {
        static bool _global_init = global_init();
        static bool _thread_init = thread_init();
        static bool _data_init   = data_init();
        consume_parameters(_global_init, _thread_init, _data_init);

        // get the iterator and check if valid
        auto&& _itr = std::get<0>(_secondary);
        if(!_data().graph().is_valid(_itr))
            return;

        // compute hash of prefix
        auto _hash_id = add_hash_id(std::get<1>(_secondary));
        // compute hash w.r.t. parent iterator (so identical kernels from different
        // call-graph parents do not locate same iterator)
        auto _hash = _hash_id + _itr->id();
        // add the hash alias
        add_hash_id(_hash_id, _hash);
        // compute depth
        auto _depth = _itr->depth() + 1;

        // see if depth + hash entry exists already
        auto _nitr = m_node_ids[_depth].find(_hash);
        if(_nitr != m_node_ids[_depth].end())
        {
            // if so, then update
            _nitr->second->obj() += std::get<2>(_secondary);
            _nitr->second->obj().laps += 1;
        }
        else
        {
            // else, create a new entry
            auto&& _tmp = Type();
            _tmp += std::get<2>(_secondary);
            _tmp.laps = 1;
            graph_node_t _node(_hash, _tmp, _depth);
            m_node_ids[_depth][_hash] = _data().emplace_child(_itr, _node);
        }
    }

protected:
    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }

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
                (trait::is_available<_Up>::value && !(std::is_same<_Vp, void>::value));
        };

        using storage_t = this_type;

        template <typename _Archive, typename _Type = Type,
                  typename std::enable_if<(is_enabled<_Type>::value), char>::type = 0>
        static void serialize(storage_t& _obj, _Archive& ar, const unsigned int version,
                              const result_array_t& result)
        {
            typename tim::trait::array_serialization<Type>::type type;
            _obj.serialize_me(type, ar, version, result);
        }

        template <typename _Archive, typename _Type = Type,
                  typename std::enable_if<!(is_enabled<_Type>::value), char>::type = 0>
        static void serialize(storage_t&, _Archive&, const unsigned int,
                              const result_array_t&)
        {}
    };

public:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Archive>
    void serialize(_Archive& ar, const unsigned int version)
    {
        using serial_write_t = write_serialization<this_type>;

        auto   num_instances = instance_count().load();
        auto&& _results      = dmp_get();
        for(uint64_t i = 0; i < _results.size(); ++i)
        {
            ar.startNode();
            ar(cereal::make_nvp("rank", i));
            ar(cereal::make_nvp("concurrency", num_instances));
            serial_write_t::serialize(*this, ar, 1, _results.at(i));
            ar.finishNode();
        }
        consume_parameters(version);
    }

private:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Archive>
    void _serialize(_Archive& ar)
    {
        auto _label = m_label;
        if(m_is_master)
            merge();
        ar(cereal::make_nvp(_label, *this));
    }

private:
    // tim::trait::array_serialization<Type>::type == TRUE
    template <typename Archive>
    void serialize_me(std::true_type, Archive&, const unsigned int,
                      const result_array_t&);

    // tim::trait::array_serialization<Type>::type == FALSE
    template <typename Archive>
    void serialize_me(std::false_type, Archive&, const unsigned int,
                      const result_array_t&);

    void internal_print();

    graph_data_t&       _data();
    const graph_data_t& _data() const { return const_cast<this_type*>(this)->_data(); }

private:
    mutable graph_data_t*     m_graph_data_instance = nullptr;
    iterator_hash_map_t       m_node_ids;
    std::unordered_set<Type*> m_stack;
};

//--------------------------------------------------------------------------------------//
//
//              Storage functions for implemented types
//
//--------------------------------------------------------------------------------------//

template <typename Type>
std::string
storage<Type, true>::get_prefix(const graph_node& node)
{
    auto _ret = get_hash_identifier(m_hash_ids, m_hash_aliases, node.id());
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master)
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(node);
        }
        else
        {
            return get_hash_identifier(node.id());
        }
    }
    return _ret;
}

//======================================================================================//

template <typename Type>
typename storage<Type, true>::graph_data_t&
storage<Type, true>::_data()
{
    using object_base_t = typename Type::base_type;

    if(m_graph_data_instance == nullptr && !m_is_master)
    {
        static bool _data_init = master_instance()->data_init();
        consume_parameters(_data_init);

        auto         m = *master_instance()->current();
        graph_node_t node(m.id(), object_base_t::dummy(), m.depth());
        m_graph_data_instance          = new graph_data_t(node);
        m_graph_data_instance->depth() = m.depth();
        if(m_node_ids.size() == 0)
            m_node_ids[0][0] = m_graph_data_instance->current();
    }
    else if(m_graph_data_instance == nullptr)
    {
        auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();

        std::string _prefix = "> [tot] total";
        add_hash_id(_prefix);
        graph_node_t node(0, object_base_t::dummy(), 0);
        m_graph_data_instance          = new graph_data_t(node);
        m_graph_data_instance->depth() = 0;
        if(m_node_ids.size() == 0)
            m_node_ids[0][0] = m_graph_data_instance->current();
    }

    m_initialized = true;
    return *m_graph_data_instance;
}

//======================================================================================//

template <typename Type>
void
storage<Type, true>::merge()
{
    if(!m_is_master || !m_initialized)
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

    stack_clear();
}

//======================================================================================//

template <typename Type>
void
storage<Type, true>::merge(this_type* itr)
{
    using pre_order_iterator = typename graph_t::pre_order_iterator;
    using sibling_iterator   = typename graph_t::sibling_iterator;

    // don't merge self
    if(itr == this)
        return;

    // if merge was not initialized return
    if(itr && !itr->is_initialized())
        return;

    itr->stack_clear();

    // create lock but don't immediately lock
    // auto_lock_t l(type_mutex<this_type>(), std::defer_lock);
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

    // lock if not already owned
    if(!l.owns_lock())
        l.lock();

    auto _copy_hash_ids = [&]() {
        for(const auto& _itr : (*itr->get_hash_ids()))
            if(m_hash_ids->find(_itr.first) == m_hash_ids->end())
                (*m_hash_ids)[_itr.first] = _itr.second;
        for(const auto& _itr : (*itr->get_hash_aliases()))
            if(m_hash_aliases->find(_itr.first) == m_hash_aliases->end())
                (*m_hash_aliases)[_itr.first] = _itr.second;
    };

    // if self is not initialized but itr is, copy data
    if(itr && itr->is_initialized() && !this->is_initialized())
    {
        PRINT_HERE("[%s]> Warning! master is not initialized! Segmentation fault likely",
                   Type::label().c_str());
        graph().insert_subgraph_after(_data().head(), itr->data().head());
        m_initialized = itr->m_initialized;
        m_finalized   = itr->m_finalized;
        _copy_hash_ids();
        return;
    }
    else
    {
        _copy_hash_ids();
    }

    if(itr->size() == 0 || !itr->data().has_head())
        return;

    bool _merged = false;
    for(auto _titr = graph().begin(); _titr != graph().end(); ++_titr)
    {
        if(_titr && itr->data().has_head() && *_titr == *itr->data().head())
        {
            typename graph_t::pre_order_iterator _nitr(itr->data().head());
            if(graph().is_valid(_nitr.begin()) && _nitr.begin())
            {
                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> worker is merging %i records into %i records",
                               Type::label().c_str(), (int) itr->size(),
                               (int) this->size());
                pre_order_iterator _pos   = _titr;
                sibling_iterator   _other = _nitr;
                for(auto sitr = _other.begin(); sitr != _other.end(); ++sitr)
                {
                    pre_order_iterator pitr = sitr;
                    graph().append_child(_pos, pitr);
                }
                _merged = true;
                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> master has %i records", Type::label().c_str(),
                               (int) this->size());
                break;
            }

            if(!_merged)
            {
                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> worker is not merged!", Type::label().c_str());
                ++_nitr;
                if(graph().is_valid(_nitr) && _nitr)
                {
                    graph().append_child(_titr, _nitr);
                    _merged = true;
                    break;
                }
            }
        }
    }

    if(!_merged)
    {
        if(settings::debug() || settings::verbose() > 2)
            PRINT_HERE("[%s]> worker is not merged!", Type::label().c_str());
        pre_order_iterator _nitr(itr->data().head());
        ++_nitr;
        if(!graph().is_valid(_nitr))
            _nitr = pre_order_iterator(itr->data().head());
        graph().append_child(_data().head(), _nitr);
    }

    itr->data().clear();
}

//======================================================================================//

template <typename Type>
typename storage<Type, true>::result_array_t
storage<Type, true>::get()
{
    //------------------------------------------------------------------------------//
    //
    //  Compute the node prefix
    //
    //------------------------------------------------------------------------------//
    auto _get_node_prefix = [&]() {
        if(!m_node_init)
            return std::string(">>> ");

        // prefix spacing
        static uint16_t width = 1;
        if(m_node_size > 9)
            width = std::max(width, (uint16_t)(log10(m_node_size) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << m_node_rank << ">>> ";
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //
    //  Compute the indentation
    //
    //------------------------------------------------------------------------------//
    // fix up the prefix based on the actual depth
    auto _compute_modified_prefix = [&](const graph_node& itr) {
        std::string _prefix      = get_prefix(itr);
        std::string _indent      = "";
        std::string _node_prefix = _get_node_prefix();

        int64_t _depth = itr.depth() - 1;
        if(_depth > 0)
        {
            for(int64_t ii = 0; ii < _depth - 1; ++ii)
                _indent += "  ";
            _indent += "|_";
        }

        return _node_prefix + _indent + _prefix;
    };

    // convert graph to a vector
    auto convert_graph = [&]() {
        result_array_t _list;
        {
            // the head node should always be ignored
            int64_t _min = std::numeric_limits<int64_t>::max();
            for(const auto& itr : graph())
                _min = std::min<int64_t>(_min, itr.depth());

            for(auto itr = graph().begin(); itr != graph().end(); ++itr)
            {
                if(itr->depth() > _min)
                {
                    auto        _depth   = itr->depth() - (_min + 1);
                    auto        _prefix  = _compute_modified_prefix(*itr);
                    auto        _rolling = itr->id();
                    auto        _parent  = graph_t::parent(itr);
                    strvector_t _hierarchy;
                    if(_parent && _parent->depth() > _min)
                    {
                        while(_parent)
                        {
                            _hierarchy.push_back(get_prefix(*_parent));
                            _rolling += _parent->id();
                            _parent = graph_t::parent(_parent);
                            if(!_parent || !(_parent->depth() > _min))
                                break;
                        }
                    }
                    if(_hierarchy.size() > 1)
                        std::reverse(_hierarchy.begin(), _hierarchy.end());
                    _hierarchy.push_back(get_prefix(*itr));
                    result_node _entry(result_tuple_t{ itr->id(), itr->obj(), _prefix,
                                                       _depth, _rolling, _hierarchy });
                    _list.push_back(_entry);
                }
            }
        }

        bool _thread_scope_only = trait::thread_scope_only<Type>::value;
        if(!settings::collapse_threads() || _thread_scope_only)
            return _list;

        result_array_t _combined;

        //--------------------------------------------------------------------------//
        //
        auto _equiv = [&](const result_node& _lhs, const result_node& _rhs) {
            return (std::get<0>(_lhs) == std::get<0>(_rhs) &&
                    std::get<2>(_lhs) == std::get<2>(_rhs) &&
                    std::get<3>(_lhs) == std::get<3>(_rhs) &&
                    std::get<4>(_lhs) == std::get<4>(_rhs));
        };

        //--------------------------------------------------------------------------//
        //
        auto _exists = [&](const result_node& _lhs) {
            for(auto itr = _combined.begin(); itr != _combined.end(); ++itr)
            {
                if(_equiv(_lhs, *itr))
                    return itr;
            }
            return _combined.end();
        };

        //--------------------------------------------------------------------------//
        //  collapse duplicates
        //
        for(const auto& itr : _list)
        {
            auto citr = _exists(itr);
            if(citr == _combined.end())
            {
                _combined.push_back(itr);
            }
            else
            {
                std::get<1>(*citr) += std::get<1>(itr);
                std::get<1>(*citr).plus(std::get<1>(itr));
            }
        }
        return _combined;
    };

    return convert_graph();
}

//======================================================================================//

template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::mpi_get()
{
#if !defined(TIMEMORY_USE_MPI)
    if(settings::debug())
        PRINT_HERE("%s", "timemory not using MPI");

    return dmp_result_t(1, get());
#else
    if(settings::debug())
        PRINT_HERE("%s", "timemory using MPI");

    // not yet implemented
    // auto comm =
    //    (settings::mpi_output_per_node()) ? mpi::get_node_comm() : mpi::comm_world_v;
    auto comm = mpi::comm_world_v;
    mpi::barrier(comm);

    int mpi_rank = mpi::rank(comm);
    int mpi_size = mpi::size(comm);
    // int mpi_rank = m_node_rank;
    // int mpi_size = m_node_size;

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const result_array_t& src) {
        std::stringstream ss;
        {
            auto space = cereal::JSONOutputArchive::Options::IndentChar::space;
            cereal::JSONOutputArchive::Options opt(16, space, 0);
            cereal::JSONOutputArchive oa(ss);
            oa(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [&](const std::string& src) {
        result_array_t ret;
        std::stringstream ss;
        ss << src;
        {
            cereal::JSONInputArchive ia(ss);
            ia(cereal::make_nvp("data", ret));
            if(settings::debug())
                printf("[RECV: %i]> data size: %lli\n", mpi_rank,
                       (long long int) ret.size());
        }
        return ret;
    };

    dmp_result_t results(mpi_size);

    auto ret = get();
    auto str_ret = send_serialize(ret);

    if(mpi_rank == 0)
    {
        for(int i = 1; i < mpi_size; ++i)
        {
            std::string str;
            if(settings::debug())
                printf("[RECV: %i]> starting %i\n", mpi_rank, i);
            mpi::recv(str, i, 0, comm);
            if(settings::debug())
                printf("[RECV: %i]> completed %i\n", mpi_rank, i);
            results[i] = recv_serialize(str);
        }
        results[mpi_rank] = ret;
    }
    else
    {
        if(settings::debug())
            printf("[SEND: %i]> starting\n", mpi_rank);
        mpi::send(str_ret, 0, 0, comm);
        if(settings::debug())
            printf("[SEND: %i]> completed\n", mpi_rank);
        return dmp_result_t(1, ret);
    }

    return results;
#endif
}

//======================================================================================//

template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::upc_get()
{
#if !defined(TIMEMORY_USE_UPCXX)
    if(settings::debug())
        PRINT_HERE("%s", "timemory not using UPC++");

    return dmp_result_t(1, get());
#else
    if(settings::debug())
        PRINT_HERE("%s", "timemory using UPC++");

    upc::barrier();

    int upc_rank = upc::rank();
    int upc_size = upc::size();

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [=](const result_array_t& src) {
        std::stringstream ss;
        {
            auto space = cereal::JSONOutputArchive::Options::IndentChar::space;
            cereal::JSONOutputArchive::Options opt(16, space, 0);
            cereal::JSONOutputArchive oa(ss);
            oa(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [=](const std::string& src) {
        result_array_t ret;
        std::stringstream ss;
        ss << src;
        {
            cereal::JSONInputArchive ia(ss);
            ia(cereal::make_nvp("data", ret));
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Function executed on remote node
    //
    auto remote_serialize = [=]() {
        return send_serialize(this_type::master_instance()->get());
    };

    dmp_result_t results(upc_size);

    //------------------------------------------------------------------------------//
    //  Combine on master rank
    //
    if(upc_rank == 0)
    {
        for(int i = 1; i < upc_size; ++i)
        {
            upcxx::future<std::string> fut = upcxx::rpc(i, remote_serialize);
            while(!fut.ready())
                upcxx::progress();
            fut.wait();
            results[i] = recv_serialize(fut.result());
        }
        results[upc_rank] = get();
    }

    upcxx::barrier(upcxx::world());

    if(upc_rank != 0)
        return dmp_result_t(1, get());
    else
        return results;
#endif
}

//======================================================================================//

template <typename Type>
void
storage<Type, true>::internal_print()
{
    base::storage::stop_profiler();

    if(!m_initialized && !m_finalized)
        return;

    auto                  requires_json = trait::requires_json<Type>::value;
    auto                  label         = Type::label();
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        finalize();
    }
    else if(settings::auto_output())
    {
        merge();
        finalize();

        bool _json_forced = requires_json;
        bool _file_output = settings::file_output();
        bool _cout_output = settings::cout_output();
        bool _json_output = settings::json_output() || _json_forced;
        bool _text_output = settings::text_output();

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

        if(!_file_output && !_cout_output && !_json_forced)
        {
            instance_count().store(0);
            return;
        }

        dmp::barrier();
        auto _results     = this->get();
        auto _dmp_results = this->dmp_get();
        dmp::barrier();

        if(settings::debug())
            printf("[%s]|%i> dmp results size: %i\n", label.c_str(), m_node_rank,
                   (int) _dmp_results.size());

        // bool return_nonzero_mpi = (dmp::using_mpi() && !settings::mpi_output_per_node()
        // &&
        //                           !settings::mpi_output_per_rank());

        if(_dmp_results.size() > 0)
        {
            if(m_node_rank != 0)
                return;
            else
            {
                _results.clear();
                for(const auto& sitr : _dmp_results)
                {
                    for(const auto& ritr : sitr)
                        _results.push_back(ritr);
                }
            }
        }

#if defined(DEBUG)
        if(tim::settings::debug() && tim::settings::verbose() > 3)
        {
            printf("\n");
            size_t w = 0;
            for(const auto& itr : _results)
                w = std::max<size_t>(w, itr.prefix().length());
            for(const auto& itr : _results)
            {
                std::cout << std::setw(w) << std::left << itr.prefix() << " : "
                          << itr.data();
                auto _hierarchy = itr.hierarchy();
                for(size_t i = 0; i < _hierarchy.size(); ++i)
                {
                    if(i == 0)
                        std::cout << " :: ";
                    std::cout << _hierarchy[i];
                    if(i + 1 < _hierarchy.size())
                        std::cout << "/";
                }
                std::cout << std::endl;
            }
            printf("\n");
        }
#endif

        int64_t _width     = Type::get_width();
        int64_t _max_depth = 0;
        int64_t _max_laps  = 0;
        // find the max width
        for(const auto mitr : _dmp_results)
        {
            for(const auto& itr : mitr)
            {
                const auto& itr_obj    = itr.data();
                const auto& itr_prefix = itr.prefix();
                const auto& itr_depth  = itr.depth();
                if(itr_depth < 0 || itr_depth > settings::max_depth())
                    continue;
                int64_t _len = itr_prefix.length();
                _width       = std::max(_len, _width);
                _max_depth   = std::max<int64_t>(_max_depth, itr_depth);
                _max_laps    = std::max<int64_t>(_max_laps, itr_obj.nlaps());
            }
        }

        int64_t              _width_laps  = std::log10(_max_laps) + 1;
        int64_t              _width_depth = std::log10(_max_depth) + 1;
        std::vector<int64_t> _widths      = { _width, _width_laps, _width_depth };

        // return type of get() function
        using get_return_type = decltype(std::declval<const Type>().get());

        auto_lock_t flk(type_mutex<std::ofstream>(), std::defer_lock);
        auto_lock_t slk(type_mutex<decltype(std::cout)>(), std::defer_lock);

        if(!flk.owns_lock())
            flk.lock();

        if(!slk.owns_lock())
            slk.lock();

        std::ofstream*       fout = nullptr;
        decltype(std::cout)* cout = nullptr;

        //--------------------------------------------------------------------------//
        // output to json file
        //
        if((_file_output && _json_output) || _json_forced)
        {
            printf("\n");
            auto jname = settings::compose_output_filename(label, ".json");
            if(jname.length() > 0)
            {
                printf("[%s]|%i> Outputting '%s'...\n", label.c_str(), m_node_rank,
                       jname.c_str());
                add_json_output(label, jname);
                {
                    using serial_write_t        = write_serialization<this_type>;
                    auto          num_instances = instance_count().load();
                    std::ofstream ofs(jname.c_str());
                    if(ofs)
                    {
                        // ensure json write final block during destruction
                        // before the file is closed
                        //  Option args: precision, spacing, indent size
                        cereal::JSONOutputArchive::Options opts(12, spacing, 2);
                        cereal::JSONOutputArchive          oa(ofs, opts);
                        oa.setNextName("timemory");
                        oa.startNode();
                        oa.setNextName("ranks");
                        oa.startNode();
                        oa.makeArray();
                        for(uint64_t i = 0; i < _dmp_results.size(); ++i)
                        {
                            oa.startNode();
                            oa(cereal::make_nvp("rank", i));
                            oa(cereal::make_nvp("concurrency", num_instances));
                            serial_write_t::serialize(*this, oa, 1, _dmp_results.at(i));
                            oa.finishNode();
                        }
                        oa.finishNode();
                        oa.finishNode();
                    }
                    if(ofs)
                        ofs << std::endl;
                    ofs.close();
                }
            }
        }
        else if(_file_output && _text_output)
        {
            printf("\n");
        }

        //--------------------------------------------------------------------------//
        // output to text file
        //
        if(_file_output && _text_output)
        {
            auto fname = settings::compose_output_filename(label, ".txt");
            if(fname.length() > 0)
            {
                fout = new std::ofstream(fname.c_str());
                if(fout && *fout)
                {
                    printf("[%s]|%i> Outputting '%s'...\n", label.c_str(), m_node_rank,
                           fname.c_str());
                    add_text_output(label, fname);
                }
                else
                {
                    delete fout;
                    fout = nullptr;
                    fprintf(stderr, "[storage<%s>::%s @ %i]|%i> Error opening '%s'...\n",
                            label.c_str(), __FUNCTION__, __LINE__, m_node_rank,
                            fname.c_str());
                }
            }
        }

        //--------------------------------------------------------------------------//
        // output to cout
        //
        if(_cout_output)
        {
            cout = &std::cout;
            printf("\n");
        }

        for(auto itr = _results.begin(); itr != _results.end(); ++itr)
        {
            auto& itr_obj    = std::get<1>(*itr);
            auto& itr_prefix = std::get<2>(*itr);
            auto& itr_depth  = std::get<3>(*itr);

            if(itr_depth < 0 || itr_depth > settings::max_depth())
                continue;
            std::stringstream _pss;
            // if we are not at the bottom of the call stack (i.e. completely
            // inclusive)
            if(itr_depth < _max_depth)
            {
                // get the next iteration
                auto eitr = itr;
                std::advance(eitr, 1);
                // counts the number of non-exclusive values
                int64_t nexclusive = 0;
                // the sum of the exclusive values
                get_return_type exclusive_values;
                // continue while not at end of graph until first sibling is
                // encountered
                if(eitr != _results.end())
                {
                    auto eitr_depth = std::get<3>(*eitr);
                    while(eitr_depth != itr_depth)
                    {
                        auto& eitr_obj = std::get<1>(*eitr);

                        // if one level down, this is an exclusive value
                        if(eitr_depth == itr_depth + 1)
                        {
                            // if first exclusive value encountered: assign; else:
                            // combine
                            if(nexclusive == 0)
                                exclusive_values = eitr_obj.get();
                            else
                                math::combine(exclusive_values, eitr_obj.get());
                            // increment. beyond 0 vs. 1, this value plays no role
                            ++nexclusive;
                        }
                        // increment iterator for next while check
                        ++eitr;
                        if(eitr == _results.end())
                            break;
                        eitr_depth = std::get<3>(*eitr);
                    }
                    // if there were exclusive values encountered
                    if(nexclusive > 0 && trait::is_available<Type>::value)
                    {
                        math::print_percentage(
                            _pss,
                            math::compute_percentage(exclusive_values, itr_obj.get()));
                    }
                }
            }

            auto _laps = itr_obj.nlaps();

            std::stringstream _oss;
            operation::print<Type>(itr_obj, _oss, itr_prefix, _laps, itr_depth, _widths,
                                   true, _pss.str());
            // for(const auto& itr : itr->hierarchy())
            //    _oss << itr << "//";
            // _oss << "\n";

            if(cout != nullptr)
                *cout << _oss.str() << std::flush;
            if(fout != nullptr)
                *fout << _oss.str() << std::flush;
        }

        if(fout)
        {
            fout->close();
            delete fout;
            fout = nullptr;
        }

        bool _dart_output = settings::dart_output();

        // if only a specific type should be echoed
        if(settings::dart_type().length() > 0)
        {
            auto dtype = settings::dart_type();
            if(operation::echo_measurement<Type>::lowercase(dtype) !=
               operation::echo_measurement<Type>::lowercase(label))
                _dart_output = false;
        }

        if(_dart_output)
        {
            printf("\n");
            uint64_t _nitr = 0;
            for(auto& itr : _results)
            {
                auto& itr_depth = itr.depth();

                if(itr_depth < 0 || itr_depth > settings::max_depth())
                    continue;

                // if only a specific number of measurements should be echoed
                if(settings::dart_count() > 0 && _nitr >= settings::dart_count())
                    continue;

                auto& itr_obj       = itr.data();
                auto& itr_hierarchy = itr.hierarchy();
                operation::echo_measurement<Type>(itr_obj, itr_hierarchy);
                ++_nitr;
            }
        }
        instance_count().store(0);
    }
    else
    {
        if(singleton_t::is_master(this))
        {
            instance_count().store(0);
        }
    }
}

//======================================================================================//

template <typename Type>
template <typename Archive>
void
storage<Type, true>::serialize_me(std::false_type, Archive& ar,
                                  const unsigned int    version,
                                  const result_array_t& graph_list)
{
    if(graph_list.size() == 0)
        return;

    ar(cereal::make_nvp("type", Type::label()),
       cereal::make_nvp("description", Type::description()),
       cereal::make_nvp("unit_value", Type::unit()),
       cereal::make_nvp("unit_repr", Type::display_unit()));
    Type::extra_serialization(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(cereal::make_nvp("hash", itr.hash()), cereal::make_nvp("prefix", itr.prefix()),
           cereal::make_nvp("depth", itr.depth()), cereal::make_nvp("entry", itr.data()));
        ar.finishNode();
    }
    ar.finishNode();
}

//======================================================================================//

template <typename Type>
template <typename Archive>
void
storage<Type, true>::serialize_me(std::true_type, Archive& ar, const unsigned int version,
                                  const result_array_t& graph_list)
{
    if(graph_list.size() == 0)
        return;

    // remove those const in case not marked const
    auto& _graph_list = const_cast<result_array_t&>(graph_list);

    Type& obj           = _graph_list.front().data();
    auto  labels        = obj.label_array();
    auto  descripts     = obj.descript_array();
    auto  units         = obj.unit_array();
    auto  display_units = obj.display_unit_array();
    ar(cereal::make_nvp("type", labels), cereal::make_nvp("description", descripts),
       cereal::make_nvp("unit_value", units),
       cereal::make_nvp("unit_repr", display_units));
    Type::extra_serialization(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(cereal::make_nvp("hash", itr.hash()), cereal::make_nvp("prefix", itr.prefix()),
           cereal::make_nvp("depth", itr.depth()), cereal::make_nvp("entry", itr.data()));
        ar.finishNode();
    }
    ar.finishNode();
}

//======================================================================================//

template <typename StorageType, typename Type, typename _HashMap, typename _GraphData>
typename StorageType::iterator
insert_heirarchy(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                 _HashMap& m_node_ids, _GraphData*& m_data, bool _has_head,
                 bool _is_master)
{
    using graph_t      = typename StorageType::graph_t;
    using graph_node_t = typename StorageType::graph_node_t;
    using iterator     = typename StorageType::iterator;

    // if first instance
    if(!_has_head || (_is_master && m_node_ids.size() == 0))
    {
        graph_node_t node(hash_id, obj, hash_depth);
        auto         itr                = m_data->append_child(node);
        m_node_ids[hash_depth][hash_id] = itr;
        return itr;
    }

    // lambda for updating settings
    auto _update = [&](iterator itr) {
        m_data->depth() = itr->depth();
        return (m_data->current() = itr);
    };

    if(m_node_ids[hash_depth].find(hash_id) != m_node_ids[hash_depth].end() &&
       m_node_ids[hash_depth].find(hash_id)->second->depth() == m_data->depth())
    {
        return _update(m_node_ids[hash_depth].find(hash_id)->second);
    }

    using sibling_itr = typename graph_t::sibling_iterator;
    graph_node_t node(hash_id, obj, m_data->depth());

    // lambda for inserting child
    auto _insert_child = [&]() {
        node.depth()                    = hash_depth;
        auto itr                        = m_data->append_child(node);
        m_node_ids[hash_depth][hash_id] = itr;
        return itr;
    };

    auto current = m_data->current();
    if(!m_data->graph().is_valid(current))
        _insert_child();

    if((hash_id) == current->id())
    {
        return current;
    }
    else if(m_data->graph().is_valid(current))
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

        // check child
        auto fchild = graph_t::child(current, 0);
        if(m_data->graph().is_valid(fchild))
        {
            for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
            {
                if((hash_id) == itr->id())
                    return _update(itr);
            }
        }
    }

    return _insert_child();
}

//======================================================================================//

}  // namespace impl

}  // namespace tim
