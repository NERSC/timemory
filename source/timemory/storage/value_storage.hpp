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

#pragma once

#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/cleanup.hpp"
#include "timemory/storage/base_storage.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tim
{
namespace operation
{
template <typename Tp>
struct call_stack;
}
//
namespace impl
{
//
template <typename Type>
class value_storage : public base::storage
{
private:
    using printer_t      = operation::finalize::print<Type, true>;
    using sample_array_t = std::vector<Type>;
    using call_stack_t   = operation::call_stack<Type>;

public:
    static constexpr bool has_data_v = true;

    template <typename KeyT, typename MappedT>
    using uomap_t = std::unordered_map<KeyT, MappedT>;

    using result_node            = node::result<Type>;
    using graph_node             = node::graph<Type>;
    using graph_type             = tim::graph<graph_node>;
    using graph_data_type        = graph_data<graph_node>;
    using iterator               = typename graph_type::iterator;
    using const_iterator         = typename graph_type::const_iterator;
    using base_type              = base::storage;
    using component_type         = Type;
    using this_type              = value_storage<Type>;
    using node_type              = typename node::data<Type>::node_type;
    using stats_type             = typename node::data<Type>::stats_type;
    using result_type            = typename node::data<Type>::result_type;
    using result_vector_type     = std::vector<result_node>;
    using dmp_result_vector_type = std::vector<result_vector_type>;
    using parent_type            = tim::storage<Type>;
    using auto_lock_t            = std::unique_lock<std::recursive_mutex>;

    template <typename Vp>
    using secondary_data_t       = std::tuple<iterator, const std::string&, Vp>;
    using iterator_hash_submap_t = uomap_t<int64_t, iterator>;
    using iterator_hash_map_t    = uomap_t<int64_t, iterator_hash_submap_t>;

    friend class tim::manager;
    friend struct node::result<Type>;
    friend struct node::graph<Type>;
    friend struct impl::storage_deleter<this_type>;
    friend struct operation::finalize::get<Type, has_data_v>;
    friend struct operation::finalize::mpi_get<Type, has_data_v>;
    friend struct operation::finalize::upc_get<Type, has_data_v>;
    friend struct operation::finalize::dmp_get<Type, has_data_v>;
    friend struct operation::finalize::print<Type, has_data_v>;
    friend struct operation::finalize::merge<Type, has_data_v>;

public:
    // static functions
    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static std::atomic<int64_t>& instance_count();

public:
    value_storage();
    ~value_storage() override;

    value_storage(const this_type&) = delete;
    value_storage(this_type&&)      = delete;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

public:
    void print() final { internal_print(); }
    void cleanup() final { operation::cleanup<Type>{}; }
    void disable() final { trait::runtime_enabled<Type>::set(false); }
    void initialize() final;
    void finalize() final;
    void stack_clear() final;
    bool global_init() final;
    bool thread_init() final;
    bool data_init() final;

    const graph_data_type& data() const { return *m_call_stack.data(); }
    const graph_type&      graph() const { return m_call_stack.data()->graph(); }
    int64_t                depth() const { return m_call_stack.depth(); }
    graph_data_type&       data() { return *m_call_stack.data(); }
    graph_type&            graph() { return m_call_stack.data()->graph(); }
    iterator&              current() { return m_call_stack.data()->current(); }

    void     reset() { m_call_stack.reset(); }
    bool     empty() const { return m_call_stack.empty(); }
    size_t   size() const { return m_call_stack.size(); }
    size_t   true_size() const { return m_call_stack.true_size(); }
    iterator pop() { return m_call_stack.pop(); }

    result_vector_type     get();
    dmp_result_vector_type mpi_get();
    dmp_result_vector_type upc_get();
    dmp_result_vector_type dmp_get();

    template <typename Tp>
    Tp& get(Tp&);
    template <typename Tp>
    Tp& mpi_get(Tp&);
    template <typename Tp>
    Tp& upc_get(Tp&);
    template <typename Tp>
    Tp& dmp_get(Tp&);

    auto        get_printer() const;
    void        stack_push(Type*);
    void        stack_pop(Type*);
    void        insert_init();
    void        add_sample(Type&&);
    auto&       get_samples();
    const auto& get_samples() const;

    iterator            insert(scope::config, const Type&, uint64_t);
    iterator_hash_map_t get_node_ids() const;

    // append an instance to the graph
    template <typename Vp>
    iterator append(const secondary_data_t<Vp>&);

    template <typename Archive>
    void serialize(Archive&, unsigned int);

protected:
    void     get_shared_manager();
    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }
    string_t get_prefix(uint64_t);

    parent_type&       get_upcast();
    const parent_type& get_upcast() const;

private:
    void internal_print();

    template <typename Archive>
    void do_serialize(Archive& ar);

private:
    call_stack_t               m_call_stack{};
    std::shared_ptr<printer_t> m_printer = {};
    sample_array_t             m_samples = {};
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
value_storage<Type>::master_is_finalizing()
{
    static bool _instance = false;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
value_storage<Type>::worker_is_finalizing()
{
    static thread_local bool _instance = master_is_finalizing();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
value_storage<Type>::is_finalizing()
{
    return worker_is_finalizing() || master_is_finalizing();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
void
value_storage<Type>::serialize(Archive& ar, const unsigned int version)
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
value_storage<Type>::do_serialize(Archive& ar)
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
auto
value_storage<Type>::get_printer() const
{
    return m_printer;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::iterator_hash_map_t
value_storage<Type>::get_node_ids() const
{
    return m_call_stack.get_node_ids();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::stack_push(Type* _obj)
{
    m_call_stack.stack_push(_obj);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::stack_pop(Type* _obj)
{
    m_call_stack.stack_pop(_obj);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename value_storage<Type>::iterator
value_storage<Type>::insert(scope::config scope_data, const Type& obj, uint64_t hash_id)
{
    return m_call_stack.insert(scope_data, obj, hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Vp>
typename value_storage<Type>::iterator
value_storage<Type>::append(const secondary_data_t<Vp>& _secondary)
{
    return m_call_stack.append(_secondary);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
value_storage<Type>::add_sample(Type&& _obj)
{
    m_samples.emplace_back(std::forward<Type>(_obj));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
auto&
value_storage<Type>::get_samples()
{
    return m_samples;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const auto&
value_storage<Type>::get_samples() const
{
    return m_samples;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
}  // namespace tim
