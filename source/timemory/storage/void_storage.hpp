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
#include <unordered_set>
#include <vector>

namespace tim
{
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
//                      impl::storage<Type, false>
//                          impl::storage_false
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
class void_storage : public base::storage
{
public:
    static constexpr bool has_data_v = false;

    using result_node            = std::tuple<>;
    using graph_node             = std::tuple<>;
    using graph_t                = std::tuple<>;
    using graph_type             = graph_t;
    using dmp_result_vector_type = std::vector<std::tuple<>>;
    using result_vector_type     = std::vector<std::tuple<>>;
    using base_type              = base::storage;
    using component_type         = Type;
    using this_type              = void_storage<Type>;
    using string_t               = std::string;
    using printer_t              = operation::finalize::print<Type, has_data_v>;
    using parent_type            = tim::storage<Type>;
    using auto_lock_t            = std::unique_lock<std::recursive_mutex>;

    using iterator       = void*;
    using const_iterator = const void*;

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
    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static std::atomic<int64_t>& instance_count();

public:
    void_storage();
    ~void_storage() override;

    void_storage(const this_type&) = delete;
    void_storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

public:
    void print() final { finalize(); }
    void cleanup() final { operation::cleanup<Type>{}; }
    void disable() final { trait::runtime_enabled<component_type>::set(false); }
    void initialize() final;
    void finalize() final;
    void stack_clear() final;

    int64_t  depth() const { return 0; }
    void     reset() {}
    bool     empty() const { return true; }
    size_t   size() const { return 0; }
    size_t   true_size() const { return 0; }
    iterator pop() { return nullptr; }

    iterator insert(int64_t, const Type&, const string_t&) { return nullptr; }

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    std::shared_ptr<printer_t> get_printer() const { return m_printer; }

    template <typename Archive>
    void serialize(Archive&, const unsigned int)
    {}

protected:
    void get_shared_manager();
    void merge();
    void merge(this_type* itr);

    parent_type&       get_upcast();
    const parent_type& get_upcast() const;

    parent_type*       get_parent();
    const parent_type* get_parent() const;

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
bool&
void_storage<Type>::master_is_finalizing()
{
    static bool _instance = false;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
void_storage<Type>::worker_is_finalizing()
{
    static thread_local bool _instance = master_is_finalizing();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
void_storage<Type>::is_finalizing()
{
    return worker_is_finalizing() || master_is_finalizing();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
}  // namespace tim
