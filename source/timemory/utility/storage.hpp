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

/** \file utility/storage.hpp
 * \headerfile utility/storage.hpp "timemory/utility/storage.hpp"
 * Storage of the call-graph for each component. Each component has a thread-local
 * singleton that holds the call-graph. When a worker thread is deleted, it merges
 * itself back into the master thread storage. When the master thread is deleted,
 * it handles I/O (i.e. text file output, JSON output, stdout output). This class
 * needs some refactoring for clarity and probably some non-templated inheritance
 * for common features because this is probably the most convoluted piece of
 * development in the entire repository.
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/gperf.hpp"
#include "timemory/data/accumulators.hpp"
#include "timemory/mpl/apply.hpp"
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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace tim
{
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
class storage<ObjectType, true> : public base::storage
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
    using component_type = ObjectType;
    using this_type      = storage<ObjectType, true>;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;
    using node_tuple_t  = std::tuple<uint64_t, ObjectType, int64_t>;
    using result_array_t = std::vector<result_node>;
    using dmp_result_t   = std::vector<result_array_t>;
    using strvector_t    = std::vector<string_t>;
    using result_tuple_t =
        std::tuple<uint64_t, ObjectType, string_t, int64_t, uint64_t, strvector_t>;

    friend struct details::storage_deleter<this_type>;
    friend struct write_serialization<this_type>;
    friend class tim::manager;

public:
    // static functions
    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }
    static pointer noninit_instance() { return get_noninit_singleton().instance(); }
    static pointer noninit_master_instance()
    {
        return get_noninit_singleton().master_instance();
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
        ObjectType&  data() { return std::get<1>(*this); }
        string_t&    prefix() { return std::get<2>(*this); }
        int64_t&     depth() { return std::get<3>(*this); }
        uint64_t&    rolling_hash() { return std::get<4>(*this); }
        strvector_t& hierarchy() { return std::get<5>(*this); }

        const uint64_t&    hash() const { return std::get<0>(*this); }
        const ObjectType&  data() const { return std::get<1>(*this); }
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
        using data_value_type = typename ObjectType::value_type;
        using data_base_type  = typename ObjectType::base_type;
        using string_t        = std::string;

        uint64_t&   id() { return std::get<0>(*this); }
        ObjectType& obj() { return std::get<1>(*this); }
        int64_t&    depth() { return std::get<2>(*this); }

        const uint64_t&   id() const { return std::get<0>(*this); }
        const ObjectType& obj() const { return std::get<1>(*this); }
        const int64_t&    depth() const { return std::get<2>(*this); }

        string_t get_prefix() const { return master_instance()->get_prefix(*this); }

        graph_node()
        : base_type(0, ObjectType(), 0)
        {}

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {}

        graph_node(const uint64_t& _id, const ObjectType& _obj, int64_t _depth)
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

        size_t data_size() const { return sizeof(ObjectType) + 2 * sizeof(int64_t); }

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
    : base_type(singleton_t::is_master_thread(), instance_count()++, ObjectType::label())
    {
        if(settings::debug())
            printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);

        component::properties<ObjectType>::has_storage() = true;

        static std::atomic<int32_t> _skip_once;
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
    virtual void print() final
    {
        typename trait::external_output_handling<ObjectType>::type type;
        external_print(type);
    }

    virtual void cleanup() final { ObjectType::invoke_cleanup(); }

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
            ObjectType::thread_finalize_policy(this);

        if(m_is_master && m_global_init)
            ObjectType::global_finalize_policy(this);

        if(settings::debug())
            PRINT_HERE("[%s]> finalizing...", m_label.c_str());
    }

    void stack_clear()
    {
        std::unordered_set<ObjectType*> _stack = m_stack;
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
    virtual bool thread_init() final
    {
        static auto _lambda = [&]() {
            if(!m_is_master)
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

    void stack_push(ObjectType* obj) { m_stack.insert(obj); }
    void stack_pop(ObjectType* obj)
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
    iterator insert(uint64_t hash_id, const ObjectType& obj, uint64_t hash_depth)
    {
        // check this now to ensure everything is initialized
        if(m_node_ids.size() == 0 || m_graph_data_instance == nullptr)
            initialize();
        bool _has_head = _data().has_head();

        // if first instance
        if(!_has_head || (this == master_instance() && m_node_ids.size() == 0))
        {
            graph_node_t node(hash_id, obj, hash_depth);
            auto         itr                = _data().append_child(node);
            m_node_ids[hash_depth][hash_id] = itr;
            return itr;
        }

        // lambda for updating settings
        auto _update = [&](iterator itr) {
            m_graph_data_instance->depth() = itr->depth();
            return (m_graph_data_instance->current() = itr);
        };

        if(m_node_ids[hash_depth].find(hash_id) != m_node_ids[hash_depth].end() &&
           m_node_ids[hash_depth].find(hash_id)->second->depth() ==
               m_graph_data_instance->depth())
        {
            return _update(m_node_ids[hash_depth].find(hash_id)->second);
        }

        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node_t node(hash_id, obj, m_graph_data_instance->depth());

        // lambda for inserting child
        auto _insert_child = [&]() {
            node.depth()                    = hash_depth;
            auto itr                        = m_graph_data_instance->append_child(node);
            m_node_ids[hash_depth][hash_id] = itr;
            // if(m_node_ids[hash_depth].bucket_count() < m_node_ids[hash_depth].size())
            //    m_node_ids[hash_depth].rehash(m_node_ids[hash_depth].size() + 10);
            return itr;
        };

        auto current = m_graph_data_instance->current();
        if(!m_graph_data_instance->graph().is_valid(current))
            _insert_child();

        // auto nchildren = graph().number_of_children(current);
        // auto nsiblings = graph().number_of_siblings(current);

        if((hash_id) == current->id())
        {
            return current;
        }
        else if(m_graph_data_instance->graph().is_valid(current))
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
                if(m_graph_data_instance->graph().is_valid(fchild))
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
    iterator insert(uint64_t hash_id, const ObjectType& obj, uint64_t hash_depth)
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
    iterator insert(const ObjectType& obj, uint64_t hash_id)
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
    iterator insert(const ObjectType& obj, uint64_t hash_id)
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
            auto&& _tmp = ObjectType();
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
                (trait::is_available<_Up>::value &&
                 !(trait::external_output_handling<_Up>::value) &&
                 !(std::is_same<_Vp, void>::value));
        };

        using storage_t = this_type;

        template <typename _Archive, typename _Type = ObjectType,
                  typename std::enable_if<(is_enabled<_Type>::value), char>::type = 0>
        static void serialize(storage_t& _obj, _Archive& ar, const unsigned int version,
                              const result_array_t& result)
        {
            typename tim::trait::array_serialization<ObjectType>::type type;
            _obj.serialize_me(type, ar, version, result);
        }

        template <typename _Archive, typename _Type = ObjectType,
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
    // tim::trait::array_serialization<ObjectType>::type == TRUE
    template <typename Archive>
    void serialize_me(std::true_type, Archive&, const unsigned int,
                      const result_array_t&);

    // tim::trait::array_serialization<ObjectType>::type == FALSE
    template <typename Archive>
    void serialize_me(std::false_type, Archive&, const unsigned int,
                      const result_array_t&);

    // tim::trait::external_output_handling<ObjectType>::type == TRUE
    void external_print(std::true_type);

    // tim::trait::external_output_handling<ObjectType>::type == FALSE
    void external_print(std::false_type);

    graph_data_t&       _data();
    const graph_data_t& _data() const { return const_cast<this_type*>(this)->_data(); }

private:
    mutable graph_data_t*           m_graph_data_instance = nullptr;
    iterator_hash_map_t             m_node_ids;
    std::unordered_set<ObjectType*> m_stack;
};

//======================================================================================//
//
//              Storage class for types that DO NOT use storage
//
//======================================================================================//

template <typename ObjectType>
class storage<ObjectType, false> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    using base_type     = base::storage;
    using this_type     = storage<ObjectType, false>;
    using string_t      = std::string;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;

    friend class tim::manager;
    friend struct details::storage_deleter<this_type>;

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

public:
    //----------------------------------------------------------------------------------//
    //
    storage()
    : base_type(singleton_t::is_master_thread(), instance_count()++, ObjectType::label())
    {
        if(settings::debug())
            printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);
        get_shared_manager();
        component::properties<ObjectType>::has_storage() = false;
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

    virtual void cleanup() final { ObjectType::invoke_cleanup(); }

    virtual void stack_clear() final
    {
        std::unordered_set<ObjectType*> _stack = m_stack;
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
            ObjectType::thread_init_policy(this);
        }
        else
        {
            ObjectType::global_init_policy(this);
            ObjectType::thread_init_policy(this);
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
            ObjectType::thread_finalize_policy(this);
        }
        else
        {
            master_is_finalizing() = true;
            worker_is_finalizing() = true;
            ObjectType::thread_finalize_policy(this);
            ObjectType::global_finalize_policy(this);
        }
    }

public:
    bool          empty() const { return true; }
    inline size_t size() const { return 0; }
    inline size_t depth() const { return 0; }

    iterator pop() { return nullptr; }
    iterator insert(int64_t, const ObjectType&, const string_t&) { return nullptr; }

    template <typename _Archive>
    void serialize(_Archive&, const unsigned int)
    {}

    void stack_push(ObjectType* obj) { m_stack.insert(obj); }
    void stack_pop(ObjectType* obj)
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
    std::unordered_set<ObjectType*> m_stack;
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
class storage : public impl::storage<_Tp, implements_storage<_Tp>::value>
{
    static constexpr bool implements_storage_v = implements_storage<_Tp>::value;
    using this_type                            = storage<_Tp>;
    using base_type                            = impl::storage<_Tp, implements_storage_v>;
    using deleter_t                            = details::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct details::storage_deleter<this_type>;
    friend class manager;
};

//--------------------------------------------------------------------------------------//
/// args:
///     1) filename
///     2) reference an object
///
template <typename _Tp>
void
generic_serialization(const std::string&, const _Tp&);

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

template <typename StorageType>
struct tim::details::storage_deleter : public std::default_delete<StorageType>
{
    using Pointer     = std::unique_ptr<StorageType, storage_deleter<StorageType>>;
    using singleton_t = tim::singleton<StorageType, Pointer>;

    storage_deleter()  = default;
    ~storage_deleter() = default;

    void operator()(StorageType* ptr)
    {
        StorageType*    master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        tim::dmp::barrier();

        if(ptr && master && ptr != master)
        {
            ptr->StorageType::stack_clear();
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
                if(!_printed_master)
                {
                    master->StorageType::stack_clear();
                    master->StorageType::print();
                    master->StorageType::cleanup();
                    _printed_master = true;
                }
            }
        }

        if(this_tid == master_tid)
        {
            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
        }
        else
        {
            if(master && ptr != master)
            {
                singleton_t::remove(ptr);
            }
            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
        }
        if(_printed_master && !_deleted_master)
        {
            if(master)
                master->StorageType::free_shared_manager();
            delete master;
            _deleted_master = true;
        }
    }

    bool _printed_master = false;
    bool _deleted_master = false;
};

//======================================================================================//
