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

/** \file timemory/data/storage_true.hpp
 * \headerfile data/storage_true.hpp "timemory/data/storage_true.hpp"
 * Defines storage implementation when the data type is not void
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/dmp.hpp"
#include "timemory/data/base_storage.hpp"
#include "timemory/data/graph.hpp"
#include "timemory/data/graph_data.hpp"
#include "timemory/mpl/bits/operations.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/stream.hpp"
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
//======================================================================================//
//      plotting declaration
//
namespace plotting
{
//--------------------------------------------------------------------------------------//
//
template <typename... _Types, typename... _Args,
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type = 0>
void
plot(_Args&&...);

//--------------------------------------------------------------------------------------//
//
template <typename... _Types, typename... _Args,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type = 0>
void
plot(_Args&&...);

}  // namespace plotting

//======================================================================================//
//      implementation
//
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

    using strvector_t  = std::vector<string_t>;
    using uintvector_t = std::vector<uint64_t>;
    using EmptyT       = std::tuple<>;

protected:
    template <typename _Tp>
    struct write_serialization;

    struct storage_data
    {
        using type         = typename trait::statistics<Type>::type;
        using stats_policy = policy::record_statistics<Type, type>;
        using stats_type   = typename stats_policy::statistics_type;
        using node_type    = std::tuple<uint64_t, Type, int64_t, stats_type>;
        using result_type  = std::tuple<uint64_t, Type, string_t, int64_t, uint64_t,
                                       uintvector_t, stats_type>;
    };

    using storage_stats_t        = typename storage_data::stats_type;
    using storage_stats_policy_t = typename storage_data::stats_policy;
    using storage_node_t         = typename storage_data::node_type;
    using storage_result_t       = typename storage_data::result_type;

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
    using node_type      = storage_node_t;
    using stats_type     = storage_stats_t;
    using result_type    = storage_result_t;
    using result_array_t = std::vector<result_node>;
    using dmp_result_t   = std::vector<result_array_t>;

    friend struct impl::storage_deleter<this_type>;
    friend struct write_serialization<this_type>;
    friend struct operation::finalize::get<Type, true>;
    friend struct operation::finalize::mpi_get<Type, true>;
    friend struct operation::finalize::upc_get<Type, true>;
    friend struct operation::finalize::dmp_get<Type, true>;
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
    struct result_node : public result_type
    {
        using base_type = result_type;

        result_node() = default;
        result_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {}
        ~result_node()                  = default;
        result_node(const result_node&) = default;
        result_node(result_node&&)      = default;
        result_node& operator=(const result_node&) = default;
        result_node& operator=(result_node&&) = default;

        result_node(uint64_t _hash, const Type& _data, const string_t& _prefix,
                    int64_t _depth, uint64_t _rolling, const uintvector_t& _hierarchy,
                    const stats_type& _stats)
        : base_type(_hash, _data, _prefix, _depth, _rolling, _hierarchy, _stats)
        {}

        uint64_t&     hash() { return std::get<0>(*this); }
        Type&         data() { return std::get<1>(*this); }
        string_t&     prefix() { return std::get<2>(*this); }
        int64_t&      depth() { return std::get<3>(*this); }
        uint64_t&     rolling_hash() { return std::get<4>(*this); }
        uintvector_t& hierarchy() { return std::get<5>(*this); }
        stats_type&   stats() { return std::get<6>(*this); }

        const uint64_t&     hash() const { return std::get<0>(*this); }
        const Type&         data() const { return std::get<1>(*this); }
        const string_t&     prefix() const { return std::get<2>(*this); }
        const int64_t&      depth() const { return std::get<3>(*this); }
        const uint64_t&     rolling_hash() const { return std::get<4>(*this); }
        const uintvector_t& hierarchy() const { return std::get<5>(*this); }
        const stats_type&   stats() const { return std::get<6>(*this); }

        // this is for compatibility with a graph_node
        uint64_t&       id() { return std::get<0>(*this); }
        Type&           obj() { return std::get<1>(*this); }
        const uint64_t& id() const { return std::get<0>(*this); }
        const Type&     obj() const { return std::get<1>(*this); }
    };

    //----------------------------------------------------------------------------------//
    //
    //      Storage type in graph
    //
    //----------------------------------------------------------------------------------//
    struct graph_node : public node_type
    {
        using this_type       = graph_node;
        using base_type       = node_type;
        using data_value_type = typename Type::value_type;
        using data_base_type  = typename Type::base_type;
        using string_t        = std::string;

        uint64_t&   id() { return std::get<0>(*this); }
        Type&       obj() { return std::get<1>(*this); }
        int64_t&    depth() { return std::get<2>(*this); }
        stats_type& stats() { return std::get<3>(*this); }

        const uint64_t&   id() const { return std::get<0>(*this); }
        const Type&       obj() const { return std::get<1>(*this); }
        const int64_t&    depth() const { return std::get<2>(*this); }
        const stats_type& stats() const { return std::get<3>(*this); }

        string_t get_prefix() const { return master_instance()->get_prefix(*this); }

        graph_node()
        : base_type(0, Type(), 0, stats_type{})
        {}

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {}

        graph_node(const uint64_t& _id, const Type& _obj, int64_t _depth)
        : base_type(_id, _obj, _depth, stats_type{})
        {}

        ~graph_node() {}

        bool operator==(const graph_node& rhs) const
        {
            return (id() == rhs.id() && depth() == rhs.depth());
        }

        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        static Type get_dummy()
        {
            using object_base_t = typename Type::base_type;
            return object_base_t::dummy();
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
    : base_type(singleton_t::is_master_thread(), instance_count()++, Type::get_label())
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

    virtual void disable() final { trait::runtime_enabled<component_type>::set(false); }

    virtual void initialize()
    {
        if(m_initialized)
            return;
        if(settings::debug())
            printf("[%s]> initializing...\n", m_label.c_str());
        m_initialized = true;
    }

    virtual void finalize() final
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

    virtual void stack_clear() final
    {
        using Base                       = typename Type::base_type;
        std::unordered_set<Type*> _stack = m_stack;
        if(settings::stack_clearing())
            for(auto& itr : _stack)
            {
                static_cast<Base*>(itr)->stop();
                static_cast<Base*>(itr)->pop_node();
            }
        m_stack.clear();
    }

    //----------------------------------------------------------------------------------//
    //
    virtual bool global_init() final
    {
        static auto _lambda = [&]() {
            if(!m_is_master && master_instance())
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
            if(!m_is_master && master_instance())
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
            if(!m_is_master && master_instance())
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
    const graph_data_t& data() const
    {
        if(!is_finalizing())
        {
            static thread_local auto _init = const_cast<this_type*>(this)->data_init();
            consume_parameters(_init);
        }
        return _data();
    }

    const graph_t& graph() const
    {
        if(!is_finalizing())
        {
            static thread_local auto _init = const_cast<this_type*>(this)->data_init();
            consume_parameters(_init);
        }
        return _data().graph();
    }

    int64_t depth() const
    {
        if(!is_finalizing())
        {
            static thread_local auto _init = const_cast<this_type*>(this)->data_init();
            consume_parameters(_init);
        }
        return (is_finalizing()) ? 0 : _data().depth();
    }

    //----------------------------------------------------------------------------------//
    //
    graph_data_t& data()
    {
        if(!is_finalizing())
        {
            static thread_local auto _init = data_init();
            consume_parameters(_init);
        }
        return _data();
    }

    graph_t& graph()
    {
        if(!is_finalizing())
        {
            static thread_local auto _init = data_init();
            consume_parameters(_init);
        }
        return _data().graph();
    }

    iterator& current()
    {
        if(!is_finalizing())
        {
            static thread_local auto _init = data_init();
            consume_parameters(_init);
        }
        return _data().current();
    }

    //----------------------------------------------------------------------------------//
    //
    inline bool     empty() const { return (_data().graph().size() <= 1); }
    inline size_t   size() const { return _data().graph().size() - 1; }
    inline iterator pop()
    {
        auto itr = _data().pop_graph();
        // if data has popped all the way up to the zeroth (relative) depth
        // then worker threads should insert a new dummy at the current
        // master thread id and depth. Be aware, this changes 'm_current' inside
        // the data graph
        //
        if(_data().at_sea_level())
            _data().add_dummy();
        return itr;
    }

    result_array_t get();
    dmp_result_t   mpi_get();
    dmp_result_t   upc_get();
    dmp_result_t   dmp_get()
    {
        dmp_result_t _ret;
        operation::finalize::dmp_get<Type, true>(*this, _ret);
        return _ret;
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
    template <typename _Scope                                              = scope::tree,
              enable_if_t<(std::is_same<_Scope, scope::tree>::value), int> = 0>
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
    template <typename _Scope                                              = scope::tree,
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
    template <typename _Scope                                              = scope::tree,
              enable_if_t<(std::is_same<_Scope, scope::tree>::value), int> = 0>
    iterator insert(const Type& obj, uint64_t hash_id)
    {
        static bool              _global_init = global_init();
        static thread_local bool _thread_init = thread_init();
        static bool              _data_init   = data_init();
        consume_parameters(_global_init, _thread_init, _data_init);

        auto hash_depth = ((_data().depth() >= 0) ? (_data().depth() + 1) : 1);
        auto itr        = insert<_Scope>(hash_id * hash_depth, obj, hash_depth);
        add_hash_id(hash_id, hash_id * hash_depth);
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Scope                                              = scope::tree,
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
        using stats_policy_type = policy::record_statistics<Type>;

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
            auto& _stats = _nitr->second->stats();
            IF_CONSTEXPR(trait::record_statistics<Type>::value)
            {
                stats_policy_type::apply(_stats, _nitr->second->obj());
            }
        }
        else
        {
            // else, create a new entry
            auto&& _tmp = Type();
            _tmp += std::get<2>(_secondary);
            _tmp.laps = 1;
            graph_node_t _node(_hash, _tmp, _depth);
            _node.stats() += _tmp.get();
            auto& _stats = _node.stats();
            IF_CONSTEXPR(trait::record_statistics<Type>::value)
            {
                stats_policy_type::apply(_stats, _tmp);
            }
            m_node_ids[_depth][_hash] = _data().emplace_child(_itr, _node);
        }
    }

protected:
    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }
    string_t get_prefix(const uint64_t& _id);

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

//--------------------------------------------------------------------------------------//

template <typename Type>
std::string
storage<Type, true>::get_prefix(const uint64_t& id)
{
    auto _ret = get_hash_identifier(m_hash_ids, m_hash_aliases, id);
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master)
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(id);
        }
        else
        {
            return get_hash_identifier(id);
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

    if(m_graph_data_instance == nullptr)
    {
        auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);

        if(!m_is_master && master_instance())
        {
            static bool _data_init = master_instance()->data_init();
            auto&       m          = master_instance()->data();
            consume_parameters(_data_init);

            if(!lk.owns_lock())
                lk.lock();

            if(m.current())
            {
                auto         _current = m.current();
                auto         _id      = _current->id();
                auto         _depth   = _current->depth();
                graph_node_t node(_id, object_base_t::dummy(), _depth);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, _depth, &m);
                m_graph_data_instance->depth()     = _depth;
                m_graph_data_instance->sea_level() = _depth;
            }
            else
            {
                graph_node_t node(0, object_base_t::dummy(), 0);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, 0, nullptr);
                m_graph_data_instance->depth()     = 0;
                m_graph_data_instance->sea_level() = 0;
            }
        }
        else
        {
            if(!lk.owns_lock())
                lk.lock();

            std::string _prefix = "> [tot] total";
            add_hash_id(_prefix);
            graph_node_t node(0, object_base_t::dummy(), 0);
            if(!m_graph_data_instance)
                m_graph_data_instance = new graph_data_t(node, 0, nullptr);
            m_graph_data_instance->depth()     = 0;
            m_graph_data_instance->sea_level() = 0;
        }

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

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
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

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
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
                   Type::get_label().c_str());
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

    int64_t num_merged     = 0;
    auto    inverse_insert = itr->data().get_inverse_insert();

    for(auto entry : inverse_insert)
    {
        auto master_entry = data().find(entry.second);
        if(master_entry != data().end())
        {
            pre_order_iterator pitr(entry.second);

            if(itr->graph().is_valid(pitr) && pitr)
            {
                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> worker is merging %i records into %i records",
                               Type::get_label().c_str(), (int) itr->size(),
                               (int) this->size());

                pre_order_iterator pos = master_entry;

                if(*pos == *pitr)
                {
                    // auto prefix = get_prefix(pitr->id());
                    // PRINT_HERE("%s %s", "Merging!", prefix.c_str());

                    ++num_merged;
                    sibling_iterator other = pitr;
                    for(auto sitr = other.begin(); sitr != other.end(); ++sitr)
                    {
                        pre_order_iterator pchild = sitr;
                        if(pchild->obj().nlaps() == 0)
                            continue;
                        // auto prefix = get_prefix(pchild->id());
                        // PRINT_HERE("%s %s", "Appending child!", prefix.c_str());
                        // graph().prepend_child(pos, pchild);
                        graph().append_child(pos, pchild);
                    }
                }
                else
                {
                    // auto prefix = get_prefix(pitr->id());
                    // PRINT_HERE("%s %s", "Continuing past dummy!", prefix.c_str());
                }

                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> master has %i records", Type::get_label().c_str(),
                               (int) this->size());

                itr->graph().erase_children(entry.second);
                itr->graph().erase(entry.second);
            }
            else
            {
                // std::stringstream ss;
                // ss << std::boolalpha << "valid: " << (itr->graph().is_valid(pitr))
                //    << ", begin: " << (static_cast<bool>(pitr.begin()));
                // PRINT_HERE("Bookmark invalid: %s", ss.str().c_str());
            }
        }
        else
        {
            // PRINT_HERE("Missing bookmark on master: '%s' @ %lu",
            //            get_prefix(entry.second->id()).c_str(), entry.second->depth());
        }
    }

    int64_t merge_size = static_cast<int64_t>(inverse_insert.size());
    if(num_merged != merge_size)
    {
        int64_t           diff = merge_size - num_merged;
        std::stringstream ss;
        ss << "Testing error! Missing " << diff << " merge points. The worker thread "
           << "contained " << merge_size << " bookmarks but only merged " << num_merged
           << " nodes!";

        PRINT_HERE("%s", ss.str().c_str());

#if defined(TIMEMORY_TESTING)
        throw std::runtime_error(ss.str());
#endif
    }

    /*
    for(auto mitr = graph().begin(); mitr != graph().end(); ++mitr)
    {
        if(!itr->data().has_head())
            break;

        if(mitr && *mitr == *itr->data().head())
        {
            pre_order_iterator _nitr(itr->data().head());

            if(graph().is_valid(_nitr.begin()) && _nitr.begin())
            {
                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> worker is merging %i records into %i records",
                               Type::get_label().c_str(), (int) itr->size(),
                               (int) this->size());

                pre_order_iterator _pos   = mitr;
                sibling_iterator   _other = _nitr;
                for(auto sitr = _other.begin(); sitr != _other.end(); ++sitr)
                {
                    pre_order_iterator pitr = sitr;
                    graph().append_child(_pos, pitr);
                }
                _merged = true;

                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> master has %i records", Type::get_label().c_str(),
                               (int) this->size());
                break;
            }

        }
    }*/

    if(num_merged == 0)
    {
        if(settings::debug() || settings::verbose() > 2)
            PRINT_HERE("[%s]> worker is not merged!", Type::get_label().c_str());
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
    result_array_t _ret;
    operation::finalize::get<Type, true>(*this, _ret);
    return _ret;
}

//======================================================================================//

template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::mpi_get()
{
    dmp_result_t _ret;
    operation::finalize::mpi_get<Type, true>(*this, _ret);
    return _ret;
}

//======================================================================================//

template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::upc_get()
{
    dmp_result_t _ret;
    operation::finalize::upc_get<Type, true>(*this, _ret);
    return _ret;
}

//======================================================================================//

template <typename Type>
void
storage<Type, true>::internal_print()
{
    base::storage::stop_profiler();

    if(!m_initialized && !m_finalized)
        return;

    auto requires_json = trait::requires_json<Type>::value;
    auto label         = Type::get_label();

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        finalize();
    }
    else if(settings::auto_output())
    {
        merge();
        finalize();

        if(!trait::runtime_enabled<Type>::get())
            return;

        bool _json_forced = requires_json;
        bool _file_output = settings::file_output();
        bool _cout_output = settings::cout_output();
        bool _json_output = (settings::json_output() || _json_forced) && _file_output;
        bool _text_output = settings::text_output() && _file_output;
        bool _plot_output = settings::plot_output() && _json_output;

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
        // && !settings::mpi_output_per_rank());

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
                    std::cout << get_prefix(_hierarchy[i]);
                    if(i + 1 < _hierarchy.size())
                        std::cout << "/";
                }
                std::cout << std::endl;
            }
            printf("\n");
        }
#endif

        settings::indent_width<Type, 0>(Type::get_width());
        settings::indent_width<Type, 1>(4);
        settings::indent_width<Type, 2>(4);

        int64_t _max_depth = 0;
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
                _max_depth = std::max<int64_t>(_max_depth, itr_depth);
                // find global max
                settings::indent_width<Type, 0>(itr_prefix.length());
                settings::indent_width<Type, 1>(std::log10(itr_obj.nlaps()) + 1);
                settings::indent_width<Type, 2>(std::log10(itr_depth) + 1);
            }
        }

        // return type of get() function
        using get_return_type = decltype(std::declval<const Type>().get());
        using compute_type    = math::compute<get_return_type>;

        auto_lock_t slk(type_mutex<decltype(std::cout)>(), std::defer_lock);
        if(!slk.owns_lock())
            slk.lock();

        std::ofstream*       fout = nullptr;
        decltype(std::cout)* cout = nullptr;

        //--------------------------------------------------------------------------//
        // output to json file
        //
        if(_json_output)
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
                        // ensure write final block during destruction
                        // before the file is closed
                        auto oa = trait::output_archive<Type>::get(ofs);
                        oa->setNextName("timemory");
                        oa->startNode();
                        oa->setNextName("ranks");
                        oa->startNode();
                        oa->makeArray();
                        for(uint64_t i = 0; i < _dmp_results.size(); ++i)
                        {
                            oa->startNode();
                            (*oa)(cereal::make_nvp("rank", i));
                            (*oa)(cereal::make_nvp("concurrency", num_instances));
                            serial_write_t::serialize(*this, *oa, 1, _dmp_results.at(i));
                            oa->finishNode();
                        }
                        oa->finishNode();
                        oa->finishNode();
                    }
                    if(ofs)
                        ofs << std::endl;
                    ofs.close();
                }
            }

            if(_plot_output)
            {
                plotting::plot<Type>(Type::get_label(), settings::output_path(),
                                     settings::dart_output(), jname);
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

        auto stream_fmt   = Type::get_format_flags();
        auto stream_width = Type::get_width();
        auto stream_prec  = Type::get_precision();

        utility::stream _stream('|', '-', stream_fmt, stream_width, stream_prec);
        for(auto itr = _results.begin(); itr != _results.end(); ++itr)
        {
            auto& itr_obj    = itr->data();
            auto& itr_prefix = itr->prefix();
            auto& itr_depth  = itr->depth();
            auto  itr_laps   = itr_obj.nlaps();

            if(itr_depth < 0 || itr_depth > settings::max_depth())
                continue;

            // counts the number of non-exclusive values
            int64_t nexclusive = 0;
            // the sum of the exclusive values
            get_return_type exclusive_values{};

            // if we are not at the bottom of the call stack (i.e. completely
            // inclusive)
            if(itr_depth < _max_depth)
            {
                // get the next iteration
                auto eitr = itr;
                std::advance(eitr, 1);
                // counts the number of non-exclusive values
                nexclusive = 0;
                // the sum of the exclusive values
                exclusive_values = get_return_type{};
                // continue while not at end of graph until first sibling is
                // encountered
                if(eitr != _results.end())
                {
                    auto eitr_depth = eitr->depth();
                    while(eitr_depth != itr_depth)
                    {
                        auto& eitr_obj = eitr->data();

                        // if one level down, this is an exclusive value
                        if(eitr_depth == itr_depth + 1)
                        {
                            // if first exclusive value encountered: assign; else:
                            // combine
                            if(nexclusive == 0)
                                exclusive_values = eitr_obj.get();
                            else
                                compute_type::plus(exclusive_values, eitr_obj.get());
                            // increment. beyond 0 vs. 1, this value plays no role
                            ++nexclusive;
                        }
                        // increment iterator for next while check
                        ++eitr;
                        if(eitr == _results.end())
                            break;
                        eitr_depth = eitr->depth();
                    }
                }
            }

            auto itr_self  = compute_type::percent_diff(exclusive_values, itr_obj.get());
            auto itr_stats = itr->stats();

            bool _first = std::distance(_results.begin(), itr) == 0;
            if(_first)
                operation::print_header<Type>(itr_obj, _stream, itr_stats);

            operation::print<Type>(itr_obj, _stream, itr_prefix, itr_laps, itr_depth,
                                   itr_self, itr_stats);

            _stream.add_row();
        }

        if(cout != nullptr)
            *cout << _stream << std::flush;
        if(fout != nullptr)
            *fout << _stream << std::flush;

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

                auto&       itr_obj       = itr.data();
                auto&       itr_hierarchy = itr.hierarchy();
                strvector_t str_hierarchy{};
                for(const auto& hitr : itr_hierarchy)
                    str_hierarchy.push_back(get_prefix(hitr));
                operation::echo_measurement<Type>(itr_obj, str_hierarchy);
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

    ar(cereal::make_nvp("type", Type::get_label()),
       cereal::make_nvp("description", Type::get_description()),
       cereal::make_nvp("unit_value", Type::get_unit()),
       cereal::make_nvp("unit_repr", Type::get_display_unit()));
    Type::extra_serialization(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(cereal::make_nvp("hash", itr.hash()), cereal::make_nvp("prefix", itr.prefix()),
           cereal::make_nvp("depth", itr.depth()), cereal::make_nvp("entry", itr.data()),
           cereal::make_nvp("rolling_hash", itr.rolling_hash()),
           // cereal::make_nvp("heirarchy", itr.hierarchy()),
           cereal::make_nvp("stats", itr.stats()));
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
    auto  descripts     = obj.description_array();
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
           cereal::make_nvp("depth", itr.depth()), cereal::make_nvp("entry", itr.data()),
           cereal::make_nvp("rolling_hash", itr.rolling_hash()),
           // cereal::make_nvp("heirarchy", itr.hierarchy()),
           cereal::make_nvp("stats", itr.stats()));
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

    // check children first because in general, child match is ideal
    auto fchild = graph_t::child(current, 0);
    if(m_data->graph().is_valid(fchild))
    {
        for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
        {
            if((hash_id) == itr->id())
                return _update(itr);
        }
    }

    // occasionally, we end up here because of some of the threading stuff that
    // has to do with the head node. Protected against mis-matches in hierarchy
    // because the actual hash includes the depth so "example" at depth 2
    // has a different hash than "example" at depth 3.
    if((hash_id) == current->id())
        return current;

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

    return _insert_child();
}

//======================================================================================//

}  // namespace impl

}  // namespace tim
