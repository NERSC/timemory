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

/**
 * \file timemory/storage/declaration.hpp
 * \brief The declaration for the types for storage without definitions
 */

#pragma once

#include "timemory/storage/macros.hpp"
#include "timemory/storage/node.hpp"
#include "timemory/storage/types.hpp"
//
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/gperf.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/data/graph.hpp"
#include "timemory/data/graph_data.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/types.hpp"
//
#include "timemory/hash/declaration.hpp"

#include <atomic>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
storage_singleton<Tp>*
get_storage_singleton();
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
storage_singleton<Tp>*
get_storage_singleton()
{
    using singleton_type  = tim::storage_singleton<Tp>;
    using component_type  = typename Tp::component_type;
    static auto _instance = std::unique_ptr<singleton_type>(
        (trait::runtime_enabled<component_type>::get()) ? new singleton_type{} : nullptr);
    return _instance.get();
}
//
//--------------------------------------------------------------------------------------//
//
//                              base::storage
//
//--------------------------------------------------------------------------------------//
//
namespace base
{
//
//--------------------------------------------------------------------------------------//
//
class tim_dll storage
{
public:
    using string_t  = std::string;
    using this_type = storage;

public:
    storage(bool _is_master, int64_t _instance_id, const std::string& _label);
    virtual ~storage();

    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

    virtual void print() {}
    virtual void cleanup() {}
    virtual void stack_clear() {}
    virtual void disable() {}
    virtual void initialize() {}
    virtual void finalize() {}
    virtual bool global_init() { return false; }
    virtual bool thread_init() { return false; }
    virtual bool data_init() { return false; }

    template <typename Tp, typename Vp>
    static this_type* base_instance();

public:
    const graph_hash_map_ptr_t&   get_hash_ids() const { return m_hash_ids; }
    const graph_hash_alias_ptr_t& get_hash_aliases() const { return m_hash_aliases; }

    hash_result_type add_hash_id(const std::string& _prefix);
    void             add_hash_id(uint64_t _lhs, uint64_t _rhs);

    bool    is_initialized() const { return m_initialized; }
    int64_t instance_id() const { return m_instance_id; }
    void    free_shared_manager();

protected:
    void add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file);
    void add_text_output(const string_t& _label, const string_t& _file)
    {
        add_file_output("text", _label, _file);
    }
    void add_json_output(const string_t& _label, const string_t& _file)
    {
        add_file_output("json", _label, _file);
    }

    static std::atomic<int>& storage_once_flag();
    static void              stop_profiler();

protected:
    bool                     m_initialized  = false;
    bool                     m_finalized    = false;
    bool                     m_global_init  = false;
    bool                     m_thread_init  = false;
    bool                     m_data_init    = false;
    bool                     m_is_master    = false;
    bool                     m_node_init    = dmp::is_initialized();
    int32_t                  m_node_rank    = dmp::rank();
    int32_t                  m_node_size    = dmp::size();
    int64_t                  m_instance_id  = -1;
    int64_t                  m_thread_idx   = threading::get_id();
    string_t                 m_label        = "";
    graph_hash_map_ptr_t     m_hash_ids     = ::tim::get_hash_ids();
    graph_hash_alias_ptr_t   m_hash_aliases = ::tim::get_hash_aliases();
    std::shared_ptr<manager> m_manager;
};
//
//--------------------------------------------------------------------------------------//
//
inline storage::storage(bool _is_master, int64_t _instance_id, const std::string& _label)
: m_initialized(false)
, m_finalized(false)
, m_global_init(false)
, m_thread_init(false)
, m_data_init(false)
, m_is_master(_is_master)
, m_node_init(dmp::is_initialized())
, m_node_rank(dmp::rank())
, m_node_size(dmp::size())
, m_instance_id(_instance_id)
, m_thread_idx(threading::get_id())
, m_label(_label)
, m_hash_ids(::tim::get_hash_ids())
, m_hash_aliases(::tim::get_hash_aliases())
{
    if(m_is_master && m_instance_id > 0)
    {
        int _id = m_instance_id;
        PRINT_HERE("%s: %i (%s)",
                   "Error! base::storage is master but is not zero instance", _id,
                   m_label.c_str());
        if(m_instance_id > 10)
        {
            // at this point we have a recursive loop
            throw std::runtime_error("duplication!");
        }
    }

    if(!m_is_master && m_instance_id == 0)
    {
        int _id = m_instance_id;
        PRINT_HERE("%s: %i (%s)",
                   "Warning! base::storage is not master but is zero instance", _id,
                   m_label.c_str());
    }

    if(settings::debug())
        PRINT_HERE("%s: %i (%s)", "base::storage instance created", (int) m_instance_id,
                   m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
inline storage::~storage()
{
    if(settings::debug())
        PRINT_HERE("%s: %i (%s)", "base::storage instance deleted", (int) m_instance_id,
                   m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
inline std::atomic<int>&
storage::storage_once_flag()
{
    static std::atomic<int> _instance(0);
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
inline void
storage::stop_profiler()
{
    // disable gperf if profiling
#if defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_PROFILER) ||     \
    defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    try
    {
        if(storage_once_flag()++ == 0)
            gperf::profiler_stop();
    } catch(std::exception& e)
    {
        std::cerr << "Error calling gperf::profiler_stop(): " << e.what()
                  << ". Continuing..." << std::endl;
    }
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline void
storage::add_hash_id(uint64_t _lhs, uint64_t _rhs)
{
    ::tim::add_hash_id(m_hash_ids, m_hash_aliases, _lhs, _rhs);
}
//
//--------------------------------------------------------------------------------------//
//
inline hash_result_type
storage::add_hash_id(const std::string& _prefix)
{
    return ::tim::add_hash_id(m_hash_ids, _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
storage::add_file_output(const std::string& _category, const std::string& _label,
                         const std::string& _file)
{
    m_manager = manager::instance();
    if(m_manager)
        m_manager->add_file_output(_category, _label, _file);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
storage::free_shared_manager()
{
    if(m_manager)
        m_manager->remove_finalizer(m_label);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace base
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
template <typename StorageType, typename Type, typename HashMap, typename GraphData>
typename StorageType::iterator
insert_hierarchy(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                 HashMap& m_node_ids, GraphData*& m_data, bool _has_head, bool _is_master,
                 uint64_t);
//
//--------------------------------------------------------------------------------------//
//
//                              impl::storage<Tp, true>
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
class storage<Type, true> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //  forward decl of some internal types
    //
    using result_node = node::result<Type>;
    using graph_node  = node::graph<Type>;

    friend struct node::result<Type>;
    friend struct node::graph<Type>;

    using strvector_t  = std::vector<string_t>;
    using uintvector_t = std::vector<uint64_t>;
    using EmptyT       = std::tuple<>;

public:
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, true>;
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using node_type      = typename node::data<Type>::node_type;
    using stats_type     = typename node::data<Type>::stats_type;
    using result_type    = typename node::data<Type>::result_type;
    using result_array_t = std::vector<result_node>;
    using dmp_result_t   = std::vector<result_array_t>;
    using printer_t      = operation::finalize::print<Type, true>;
    using sample_array_t = std::vector<Type>;

    friend struct impl::storage_deleter<this_type>;
    friend struct operation::finalize::get<Type, true>;
    friend struct operation::finalize::mpi_get<Type, true>;
    friend struct operation::finalize::upc_get<Type, true>;
    friend struct operation::finalize::dmp_get<Type, true>;
    friend struct operation::finalize::print<Type, true>;
    friend struct operation::finalize::merge<Type, true>;
    friend class tim::manager;

public:
    // static functions
    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();

    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }
    static std::atomic<int64_t>& instance_count();

public:
    using graph_node_t   = graph_node;
    using graph_data_t   = graph_data<graph_node_t>;
    using graph_t        = typename graph_data_t::graph_t;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;
    template <typename Vp>
    using secondary_data_t = std::tuple<iterator, const std::string&, Vp>;
    template <typename KeyT, typename MappedT>
    using uomap_t                = std::unordered_map<KeyT, MappedT>;
    using iterator_hash_submap_t = uomap_t<int64_t, iterator>;
    using iterator_hash_map_t    = uomap_t<int64_t, iterator_hash_submap_t>;

public:
    storage();
    ~storage();

    storage(const this_type&) = delete;
    storage(this_type&&)      = delete;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

public:
    void get_shared_manager();

    virtual void print() final { internal_print(); }
    virtual void cleanup() final { Type::cleanup(); }
    virtual void disable() final { trait::runtime_enabled<component_type>::set(false); }
    virtual void initialize() final;
    virtual void finalize() final;
    virtual void stack_clear() final;
    virtual bool global_init() final;
    virtual bool thread_init() final;
    virtual bool data_init() final;

    const graph_data_t& data() const;
    const graph_t&      graph() const;
    int64_t             depth() const;
    graph_data_t&       data();
    graph_t&            graph();
    iterator&           current();

    inline bool    empty() const { return (_data().graph().size() <= 1); }
    inline size_t  size() const { return _data().graph().size() - 1; }
    iterator       pop();
    result_array_t get();
    dmp_result_t   mpi_get();
    dmp_result_t   upc_get();
    dmp_result_t   dmp_get();

    std::shared_ptr<printer_t> get_printer() const { return m_printer; }

    const iterator_hash_map_t get_node_ids() const { return m_node_ids; }

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    void insert_init();

    iterator insert(scope::config scope_data, const Type& obj, uint64_t hash_id);

    template <typename Vp>
    void append(const secondary_data_t<Vp>& _secondary);

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version);

    void add_sample(Type&& _obj) { m_samples.emplace_back(std::forward<Type>(_obj)); }

    auto&       get_samples() { return m_samples; }
    const auto& get_samples() const { return m_samples; }

protected:
    iterator insert_tree(uint64_t hash_id, const Type& obj, uint64_t hash_depth);
    iterator insert_timeline(uint64_t hash_id, const Type& obj, uint64_t hash_depth);
    iterator insert_flat(uint64_t hash_id, const Type& obj, uint64_t hash_depth);

    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }
    string_t get_prefix(const uint64_t& _id);

private:
    void check_consistency();

    template <typename Archive>
    void do_serialize(Archive& ar);

    void internal_print();

    graph_data_t&       _data();
    const graph_data_t& _data() const { return const_cast<this_type*>(this)->_data(); }

private:
    uint64_t                   m_timeline_counter    = 1;
    mutable graph_data_t*      m_graph_data_instance = nullptr;
    iterator_hash_map_t        m_node_ids;
    std::unordered_set<Type*>  m_stack;
    std::shared_ptr<printer_t> m_printer;
    sample_array_t             m_samples;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert(scope::config scope_data, const Type& obj, uint64_t hash_id)
{
    insert_init();
    auto hash_depth = scope_data.compute_depth(_data().depth());
    auto hash_value = scope_data.compute_hash(hash_id, hash_depth, m_timeline_counter);
    add_hash_id(hash_id, hash_value);

    // even when flat is combined with timeline, it still inserts at depth of 1
    // so this is easiest check
    if(scope_data.is_flat())
        return insert_flat(hash_value, obj, hash_depth);

    // in the case of tree + timeline, timeline will have appropriately modified the
    // depth and hash so it doesn't really matter which check happens first here
    // however, the query for is_timeline() is cheaper so we will check that
    // and fallback to inserting into tree without a check
    // if(scope_data.is_timeline())
    //    return insert_timeline(hash_value, obj, hash_depth);

    // default fall-through if neither flat nor timeline
    return insert_tree(hash_value, obj, hash_depth);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Vp>
void
storage<Type, true>::append(const secondary_data_t<Vp>& _secondary)
{
    insert_init();

    // get the iterator and check if valid
    auto&& _itr = std::get<0>(_secondary);
    if(!_data().graph().is_valid(_itr))
        return;

    // compute hash of prefix
    auto _hash_id = add_hash_id(std::get<1>(_secondary));
    // compute hash w.r.t. parent iterator (so identical kernels from different
    // call-graph parents do not locate same iterator)
    auto _hash = _hash_id ^ _itr->id();
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
        operation::add_statistics<Type>(_nitr->second->obj(), _stats);
    }
    else
    {
        // else, create a new entry
        auto&& _tmp = Type{};
        _tmp += std::get<2>(_secondary);
        _tmp.laps = 1;
        graph_node_t _node(_hash, _tmp, _depth, m_thread_idx);
        _node.stats() += _tmp.get();
        auto& _stats = _node.stats();
        operation::add_statistics<Type>(_tmp, _stats);
        m_node_ids[_depth][_hash] = _data().emplace_child(_itr, _node);
    }
}
//
//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_tree(uint64_t hash_id, const Type& obj, uint64_t hash_depth)
{
    bool _has_head = _data().has_head();
    return insert_hierarchy<this_type, Type>(hash_id, obj, hash_depth, m_node_ids,
                                             m_graph_data_instance, _has_head,
                                             m_is_master, m_thread_idx);
}

//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_timeline(uint64_t hash_id, const Type& obj,
                                     uint64_t hash_depth)
{
    auto         _current = _data().current();
    graph_node_t _node(hash_id, obj, hash_depth, m_thread_idx);
    return _data().emplace_child(_current, _node);
}

//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_flat(uint64_t hash_id, const Type& obj, uint64_t hash_depth)
{
    static thread_local auto _current = _data().head();
    static thread_local bool _first   = true;
    if(_first)
    {
        _first = false;
        if(_current.begin())
            _current = _current.begin();
        else
        {
            graph_node_t node(hash_id, obj, hash_depth, m_thread_idx);
            auto         itr                = _data().emplace_child(_current, node);
            m_node_ids[hash_depth][hash_id] = itr;
            _current                        = itr;
            return itr;
        }
    }

    auto _existing = m_node_ids[hash_depth].find(hash_id);
    if(_existing != m_node_ids[hash_depth].end())
        return m_node_ids[hash_depth].find(hash_id)->second;

    graph_node_t node(hash_id, obj, hash_depth, m_thread_idx);
    auto         itr                = _data().emplace_child(_current, node);
    m_node_ids[hash_depth][hash_id] = itr;
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
void
storage<Type, true>::serialize(Archive& ar, const unsigned int version)
{
    using bool_type = typename trait::array_serialization<Type>::type;

    auto   num_instances = instance_count().load();
    auto&& _results      = dmp_get();
    for(uint64_t i = 0; i < _results.size(); ++i)
    {
        if(_results.at(i).empty())
            continue;

        ar.startNode();

        ar(cereal::make_nvp("rank", i));
        ar(cereal::make_nvp("concurrency", num_instances));
        m_printer->print_metadata(bool_type{}, ar, _results.at(i).front().data());
        Type::extra_serialization(ar, 1);
        save(ar, _results.at(i));

        ar.finishNode();
    }
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
    auto _label = m_label;
    if(m_is_master)
        merge();
    ar(cereal::make_nvp(_label, *this));
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
//
//--------------------------------------------------------------------------------------//
//
//                      impl::storage<Type, false>
//                          impl::storage_false
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
class storage<Type, false> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, false>;
    using string_t       = std::string;
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using printer_t      = operation::finalize::print<Type, false>;

    friend class tim::manager;
    friend struct impl::storage_deleter<this_type>;
    friend struct operation::finalize::print<Type, false>;
    friend struct operation::finalize::merge<Type, false>;

    using result_node    = std::tuple<>;
    using graph_t        = std::tuple<>;
    using graph_node     = std::tuple<>;
    using dmp_result_t   = std::vector<std::tuple<>>;
    using result_array_t = std::vector<std::tuple<>>;
    using uintvector_t   = std::vector<uint64_t>;

public:
    using iterator       = void*;
    using const_iterator = const void*;

    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();

    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }
    static std::atomic<int64_t>& instance_count();

public:
    storage();
    ~storage();

    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

    virtual void print() final { finalize(); }
    virtual void cleanup() final { Type::cleanup(); }
    virtual void stack_clear() final;
    virtual void disable() final { trait::runtime_enabled<component_type>::set(false); }

    void initialize() final;
    void finalize() final;

    bool          empty() const { return true; }
    inline size_t size() const { return 0; }
    inline size_t depth() const { return 0; }

    iterator pop() { return nullptr; }
    iterator insert(int64_t, const Type&, const string_t&) { return nullptr; }

    template <typename Archive>
    void serialize(Archive&, const unsigned int)
    {}

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    std::shared_ptr<printer_t> get_printer() const { return m_printer; }

protected:
    void get_shared_manager();
    void merge();
    void merge(this_type* itr);

private:
    template <typename Archive>
    void do_serialize(Archive&)
    {}

private:
    std::unordered_set<Type*>  m_stack;
    std::shared_ptr<printer_t> m_printer;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp>
class storage : public impl::storage<Tp, implements_storage<Tp, Vp>::value>
{
public:
    static constexpr bool implements_storage_v = implements_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp, Vp>;
    using base_type                            = impl::storage<Tp, implements_storage_v>;
    using deleter_t                            = impl::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct impl::storage_deleter<this_type>;
    friend class manager;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
class storage<Tp, type_list<>>
: public storage<
      Tp, conditional_t<(trait::is_available<Tp>::value), typename Tp::value_type, void>>
{
public:
    using Vp =
        conditional_t<(trait::is_available<Tp>::value), typename Tp::value_type, void>;
    static constexpr bool implements_storage_v = implements_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp, Vp>;
    using base_type                            = impl::storage<Tp, implements_storage_v>;
    using deleter_t                            = impl::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct impl::storage_deleter<this_type>;
    friend class manager;
};
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
template <typename StorageType>
struct storage_deleter : public std::default_delete<StorageType>
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

        // tim::dmp::barrier();

        if(ptr && master && ptr != master)
        {
            ptr->StorageType::stack_clear();
            master->StorageType::merge(ptr);
        }
        else
        {
            // sometimes the worker threads get deleted after the master thread
            // but the singleton class will ensure it is merged so we are
            // safe to leak here
            if(ptr && !master && this_tid != master_tid)
            {
                ptr->StorageType::free_shared_manager();
                ptr = nullptr;
                return;
            }

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
            {
                // ptr->StorageType::disable();
                ptr->StorageType::free_shared_manager();
            }
            delete ptr;
        }
        else
        {
            if(master && ptr != master)
                singleton_t::remove(ptr);

            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
        }

        if(_printed_master && !_deleted_master)
        {
            if(master)
            {
                // master->StorageType::disable();
                master->StorageType::free_shared_manager();
            }
            delete master;
            _deleted_master = true;
        }
    }

    bool _printed_master = false;
    bool _deleted_master = false;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename StorageType, typename Type, typename HashMap, typename GraphData>
typename StorageType::iterator
insert_hierarchy(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                 HashMap& m_node_ids, GraphData*& m_data, bool _has_head, bool _is_master,
                 uint64_t tid)
{
    using graph_t       = typename StorageType::graph_t;
    using graph_node_t  = typename StorageType::graph_node_t;
    using iterator      = typename StorageType::iterator;
    using id_hash_map_t = typename HashMap::mapped_type;

    // if first instance
    if(!_has_head || (_is_master && m_node_ids.size() == 0))
    {
        graph_node_t node(hash_id, obj, hash_depth, tid);
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
    graph_node_t node(hash_id, obj, m_data->depth(), tid);

    // lambda for inserting child
    auto _insert_child = [&]() {
        node.depth() = hash_depth;
        auto itr     = m_data->append_child(node);
        auto ditr    = m_node_ids.find(hash_depth);
        if(ditr == m_node_ids.end())
            m_node_ids.insert({ hash_depth, id_hash_map_t{} });
        auto hitr = m_node_ids.at(hash_depth).find(hash_id);
        if(hitr == m_node_ids.at(hash_depth).end())
            m_node_ids.at(hash_depth).insert({ hash_id, iterator{} });
        m_node_ids.at(hash_depth).at(hash_id) = itr;
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
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp>
inline base::storage*
base::storage::base_instance()
{
    using storage_type = tim::storage<Tp, Vp>;
    if(trait::runtime_enabled<Tp>::get())
    {
        static thread_local auto _instance = []() {
            auto _tmp = storage_type::instance();
            // _tmp->initialize();
            return _tmp;
        }();
        return static_cast<base::storage*>(_instance);
    }
    return nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
