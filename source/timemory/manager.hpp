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

/** \file manager.hpp
 * \headerfile manager.hpp "timemory/manager.hpp"
 * Static singleton handler that is not templated. In general, this is the
 * first object created and last object destroy. It should be utilized to
 * store type-independent data
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/apply.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"
#include "timemory/serializer.hpp"
#include "timemory/singleton.hpp"
#include "timemory/storage.hpp"
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//
namespace cupti
{
inline void
initialize()
{
#if defined(TIMEMORY_USE_CUPTI)
    unsigned int init_flags = 0;
    cuInit(init_flags);
#endif
}
}  // namespace cupti

//--------------------------------------------------------------------------------------//

template <typename... Types>
class component_tuple;

namespace details
{
struct manager_deleter;
}

//--------------------------------------------------------------------------------------//

tim_api class manager
{
public:
    using this_type     = manager;
    using pointer_t     = std::unique_ptr<this_type, details::manager_deleter>;
    using singleton_t   = singleton<this_type, pointer_t>;
    using size_type     = std::size_t;
    using string_t      = std::string;
    using comm_group_t  = std::tuple<MPI_Comm, int32_t>;
    using auto_lock_t   = std::unique_lock<mutex_t>;
    using pointer       = singleton_t::pointer;
    using smart_pointer = singleton_t::smart_pointer;
    using string_list_t = std::deque<string_t>;
    using void_counter  = counted_object<void>;

    //----------------------------------------------------------------------------------//
    //
    //  the node type
    //
    //----------------------------------------------------------------------------------//
    class graph_node : public std::tuple<int64_t, string_t, string_list_t>
    {
    public:
        using this_type = graph_node;
        using base_type = std::tuple<int64_t, string_t, string_list_t>;

        int64_t&       id() { return std::get<0>(*this); }
        string_t&      prefix() { return std::get<1>(*this); }
        string_list_t& data() { return std::get<2>(*this); }

        const int64_t&       id() const { return std::get<0>(*this); }
        const string_t&      prefix() const { return std::get<1>(*this); }
        const string_list_t& data() const { return std::get<2>(*this); }

        graph_node()
        : base_type(0, "", {})
        {
        }

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {
        }

        graph_node(const int64_t& _id, const string_t& _prefix, const string_list_t& _l)
        : base_type(_id, _prefix, _l)
        {
        }

        graph_node(const int64_t& _id, const string_t& _prefix, const string_t& _l)
        : base_type(_id, _prefix, string_list_t())
        {
            data().push_back(_l);
        }

        ~graph_node() {}
        // explicit graph_node(const this_type&) = default;
        // explicit graph_node(this_type&&)      = default;
        // graph_node& operator=(const this_type&) = default;
        // graph_node& operator=(this_type&&) = default;

        bool operator==(const graph_node& rhs) const { return (id() == rhs.id()); }
        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            for(const auto& itr : rhs.data())
                data().push_back(itr);
            return *this;
        }
    };

    //----------------------------------------------------------------------------------//

    using graph_t        = tim::graph<graph_node>;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;

    //----------------------------------------------------------------------------------//
    //
    //  graph instance + current node + head node
    //
    //----------------------------------------------------------------------------------//
    struct graph_data
    {
        using this_type  = graph_data;
        int64_t  m_depth = -1;
        graph_t  m_graph;
        iterator m_current;
        iterator m_head;

        graph_data()
        : m_depth(-1)
        {
        }

        ~graph_data() { m_graph.clear(); }

        graph_data(const this_type&) = default;
        graph_data& operator=(const this_type&) = delete;
        graph_data& operator=(this_type&&) = default;

        int64_t&  depth() { return m_depth; }
        graph_t&  graph() { return m_graph; }
        iterator& current() { return m_current; }
        iterator& head() { return m_head; }

        const graph_t& graph() const { return m_graph; }

        iterator       begin() { return m_graph.begin(); }
        iterator       end() { return m_graph.end(); }
        const_iterator begin() const { return m_graph.begin(); }
        const_iterator end() const { return m_graph.end(); }

        inline void reset()
        {
            m_graph.erase_children(m_head);
            m_depth   = 0;
            m_current = m_head;
        }

        inline iterator pop_graph()
        {
            if(m_depth > 0 && !m_graph.is_head(m_current))
            {
                --m_depth;
                m_current = graph_t::parent(m_current);
            }
            else if(m_depth == 0)
            {
                m_current = m_head;
            }
            return m_current;
        }

        inline iterator append_child(const graph_node& node)
        {
            ++m_depth;
            return (m_current = m_graph.append_child(m_current, node));
        }
    };

public:
    // Constructor and Destructors
    manager();
    ~manager();

public:
    // Public static functions
    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();
    static void    enable(bool val = true) { void_counter::enable(val); }
    static void    disable(bool val = true) { void_counter::enable(!val); }
    static bool    is_enabled() { return void_counter::enable(); }
    static void    max_depth(const int32_t& val) { void_counter::set_max_depth(val); }
    static int32_t max_depth() { return void_counter::max_depth(); }
    static int32_t total_instance_count() { return f_manager_instance_count().load(); }

    void merge(pointer);
    void print(bool ign_cutoff, bool endline);
    void insert(const int64_t& _hash_id, const string_t& _prefix, const string_t& _data);

    static void exit_hook()
    {
        auto*   ptr   = noninit_master_instance();
        int32_t count = 0;
        if(ptr)
        {
            ptr->print(false, false);
            count = ptr->instance_count();
            printf("\n\n############## tim::~manager destroyed [%i] ##############\n",
                   count);
            delete ptr;
        }
        tim::papi::shutdown();
        // tim::cupti::shutdown();
    }

    static void print(const tim::component_tuple<>&) {}

    template <typename Head, typename... Tail>
    static void print(const tim::component_tuple<Head, Tail...>&);

    template <typename ComponentTuple_t>
    static void print()
    {
        print(ComponentTuple_t());
    }

public:
    // Public member functions
    int32_t instance_count() const { return m_instance_count; }

public:
    //
    const graph_data& data() const { return m_data; }
    const graph_t&    graph() const { return m_data.graph(); }

    graph_data& data() { return m_data; }
    iterator&   current() { return m_data.current(); }
    graph_t&    graph() { return m_data.graph(); }

protected:
    // protected static functions
    static comm_group_t get_communicator_group();

protected:
    // protected functions
    string_t get_prefix() const;

protected:
    template <typename List>
    struct PopFront_t;

    template <typename Head, typename... Tail>
    struct PopFront_t<tim::component_tuple<Head, Tail...>>
    {
        using type = tim::component_tuple<Tail...>;
    };

    template <typename List>
    using PopFront = typename PopFront_t<List>::type;

private:
    // private static variables
    /// for temporary enabling/disabling
    // static bool f_enabled();
    /// number of timing manager instances
    static std::atomic<int32_t>& f_manager_instance_count();

private:
    // private variables
    /// instance id
    int32_t m_instance_count;
    /// mutex
    mutex_t m_mutex;
    /// data represented as string
    graph_data m_data;
    /// list of node ids
    std::unordered_map<int64_t, iterator> m_node_ids;

private:
    /// num-threads based on number of managers created
    static std::atomic<int32_t>& f_thread_counter()
    {
        static std::atomic<int32_t> _instance;
        return _instance;
    }
};

//======================================================================================//

namespace details
{
//--------------------------------------------------------------------------------------//

struct manager_deleter
{
    using Type        = tim::manager;
    using pointer_t   = std::unique_ptr<Type, manager_deleter>;
    using singleton_t = singleton<Type, pointer_t>;

    void operator()(Type* ptr)
    {
        Type*           master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        if(ptr && master && ptr != master)
        {
        }
        else
        {
            if(ptr)
            {
                // ptr->print();
            }
            else if(master)
            {
                // master->print();
            }
        }

        if(this_tid == master_tid)
        {
            // delete ptr;
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

//--------------------------------------------------------------------------------------//
inline manager::singleton_t&
manager_singleton()
{
    static manager::singleton_t _instance = manager::singleton_t::instance();
    return _instance;
}
//--------------------------------------------------------------------------------------//

}  // namespace details

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/impl/manager.icpp"
