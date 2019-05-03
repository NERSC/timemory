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
 * Static singleton handler of auto-timers
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

class manager;

namespace details
{
struct manager_deleter;
}

//--------------------------------------------------------------------------------------//

tim_api class manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    using this_type              = manager;
    using pointer_t              = std::unique_ptr<this_type, details::manager_deleter>;
    using singleton_t            = singleton<this_type, pointer_t>;
    using size_type              = std::size_t;
    using string_t               = std::string;
    using ostream_t              = std::ostream;
    using ofstream_t             = std::ofstream;
    using comm_group_t           = std::tuple<MPI_Comm, int32_t>;
    using mutex_t                = std::mutex;
    using mutex_map_t            = uomap<uint64_t, mutex_t>;
    using auto_lock_t            = std::unique_lock<mutex_t>;
    using pointer                = singleton_t::pointer;
    using smart_pointer          = singleton_t::smart_pointer;
    using daughter_list_t        = std::set<this_type*>;
    using get_num_threads_func_t = std::function<int64_t()>;
    using counter_t              = std::atomic<uint64_t>;
    using string_list_t          = std::deque<string_t>;

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

        ~graph_node() {}
        // explicit graph_node(const this_type&) = default;
        // explicit graph_node(this_type&&)      = default;
        // graph_node& operator=(const this_type&) = default;
        // graph_node& operator=(this_type&&) = default;

        bool operator==(const graph_node& rhs) const { return (id() == rhs.id()); }
        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            DEBUG_PRINT_HERE("");
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
            DEBUG_PRINT_HERE("default");
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
        const_iterator begin() const { return m_graph.cbegin(); }
        const_iterator end() const { return m_graph.cend(); }
        const_iterator cbegin() const { return m_graph.cbegin(); }
        const_iterator cend() const { return m_graph.cend(); }

        inline void reset()
        {
            m_graph.erase_children(m_head);
            m_depth   = 0;
            m_current = m_head;
        }

        inline iterator pop_graph()
        {
            DEBUG_PRINT_HERE("");
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
            DEBUG_PRINT_HERE("");
            ++m_depth;
            return (m_current = m_graph.append_child(m_current, node));
        }
    };

public:
    // Constructor and Destructors
    manager();
    virtual ~manager();

public:
    // Public static functions
    static pointer                  instance();
    static pointer                  master_instance();
    static pointer                  noninit_instance();
    static pointer                  noninit_master_instance();
    static void                     write_json(path_t _fname);
    static std::pair<int32_t, bool> write_json(ostream_t& os);
    static int  get_instance_count() { return f_manager_instance_count().load(); }
    static void set_get_num_threads_func(get_num_threads_func_t f)
    {
        f_get_num_threads() = std::bind(f);
    }

    static void    enable(bool val = true) { counted_object<void>::enable(val); }
    static bool    is_enabled() { return counted_object<void>::enable(); }
    static int32_t max_depth() { return counted_object<void>::max_depth(); }
    static void    max_depth(const int32_t& val) { set_max_depth(val); }
    static int32_t get_max_depth() { return max_depth(); }
    static void    set_max_depth(const int32_t& val)
    {
        counted_object<void>::set_max_depth(val);
    }

protected:
    static void                     write_json_no_mpi(path_t _fname);
    static void                     write_json_no_mpi(ostream_t& os);
    static void                     write_json_mpi(path_t _fname);
    static std::pair<int32_t, bool> write_json_mpi(ostream_t& os);

public:
    // Public member functions
    template <typename _Tp>
    _Tp& get(const string_t& key, const string_t& tag = "cxx", int32_t ncount = 0,
             int32_t nhash = 0);

    void set_output_stream(const path_t&);
    void set_output_stream(ostream_t& = std::cout);
    void write_report(path_t _fname, bool ign_cutoff = false);
    void write_report(ostream_t& os = std::cout, bool ign_cutoff = false,
                      bool endline = true);
    void write_serialization(const path_t& _fname) const { write_json(_fname); }
    void write_serialization(ostream_t& os = std::cout) const { write_json(os); }
    void report(bool ign_cutoff = false, bool endline = true) const;
    void report(ostream_t& os, bool ign_cutoff = false, bool endline = true) const;

    void                   merge();
    void                   merge(pointer);
    void                   clear();
    size_type              size() const;
    void                   print(bool ign_cutoff = false, bool endline = true);
    void                   sync_hierarchy();
    daughter_list_t&       daughters() { return m_daughters; }
    const daughter_list_t& daughters() const { return m_daughters; }
    void                   add(pointer ptr);
    void                   remove(pointer ptr);
    void                   set_merge(bool val) { m_merge.store(val); }
    bool                   is_reporting_to_file() const;
    ostream_t*             get_output_stream() const { return m_report; }
    uint64_t               laps() const { return compute_total_laps(); }
    uint64_t               total_laps() const;
    void                   update_total_timer_format();
    int32_t                instance_count() const { return m_instance_count; }
    void                   self_cost(bool) {}
    bool                   self_cost() const { return false; }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version);

    friend std::ostream& operator<<(std::ostream& os, const manager& man)
    {
        std::stringstream ss;
        man.report(ss, true, false);
        os << ss.str();
        return os;
    }

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
    inline uint64_t string_hash(const string_t&) const;
    string_t        get_prefix() const;
    uint64_t        compute_total_laps() const;
    void            insert_global_timer();
    void            compute_self();

protected:
    // protected static variables
    static get_num_threads_func_t& f_get_num_threads();

private:
    // private functions
    void        report(ostream_t*, bool = false, bool = true) const {}
    ofstream_t* get_ofstream(ostream_t* m_os) const;

private:
    // private static variables
    /// for temporary enabling/disabling
    // static bool f_enabled();
    /// number of timing manager instances
    static std::atomic_int& f_manager_instance_count();

private:
    // private variables
    /// merge checking
    std::atomic<bool> m_merge;
    /// self format
    bool m_self_format;
    /// instance id
    int32_t m_instance_count;
    /// total laps
    counter_t m_laps;
    /// mutex
    mutex_t m_mutex;
    /// daughter list
    daughter_list_t m_daughters;
    /// output stream for total timing report
    ostream_t* m_report;
    /// data represented as string
    graph_data m_data;
    /// list of node ids
    std::unordered_map<int64_t, iterator> m_node_ids;

public:
    //----------------------------------------------------------------------------------//
    //
    void insert(const int64_t& _hash_id, const string_t& _prefix, const string_t& _data)
    {
        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node node(_hash_id, _prefix, string_list_t({ _data }));

        auto _update = [&](iterator itr) {
            m_data.current() = itr;
            *m_data.current() += node;
        };

        // lambda for inserting child
        auto _insert_child = [&]() {
            auto itr = m_data.append_child(node);
            m_node_ids.insert(std::make_pair(_hash_id, itr));
        };

        if(m_node_ids.find(_hash_id) != m_node_ids.end())
        {
            _update(m_node_ids.find(_hash_id)->second);
        }

        // if first instance
        if(m_data.depth() < 0)
        {
            if(this == master_instance())
            {
                DEBUG_PRINT_HERE("insert first in master");
                m_data.depth()   = 0;
                m_data.head()    = m_data.graph().set_head(node);
                m_data.current() = m_data.head();
            }
            else
            {
                return;
            }
        }
        else
        {
            auto current = m_data.current();

            if(_hash_id == current->id())
            {
                return;
            }
            else if(m_data.graph().is_valid(current))
            {
                // check parent if not head
                if(!m_data.graph().is_head(current))
                {
                    auto parent = graph_t::parent(current);
                    for(sibling_itr itr = parent.begin(); itr != parent.end(); ++itr)
                    {
                        // check hash id's
                        if(_hash_id == itr->id())
                        {
                            _update(itr);
                        }
                    }
                }

                // check siblings
                for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
                {
                    // skip if current
                    if(itr == current)
                        continue;
                    // check hash id's
                    if(_hash_id == itr->id())
                    {
                        _update(itr);
                    }
                }

                // check children
                auto nchildren = graph_t::number_of_children(current);
                if(nchildren == 0)
                {
                    _insert_child();
                }
                else
                {
                    bool exists = false;
                    auto fchild = graph_t::child(current, 0);
                    for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                    {
                        if(_hash_id == itr->id())
                        {
                            exists = true;
                            _update(itr);
                            break;
                        }
                    }
                    if(!exists)
                        _insert_child();
                }
            }
        }
        return _insert_child();
    }
};

//--------------------------------------------------------------------------------------//
template <typename Archive>
inline void
manager::serialize(Archive& ar, const unsigned int /*version*/)
{
    auto _nthreads = (f_get_num_threads())();
    if(_nthreads == 1)
        _nthreads = f_manager_instance_count();
    bool _self_cost = this->self_cost();
    ar(serializer::make_nvp("concurrency", _nthreads));
    ar(serializer::make_nvp("self_cost", _self_cost));
}
//--------------------------------------------------------------------------------------//
inline uint64_t
manager::string_hash(const string_t& str) const
{
    return std::hash<string_t>()(str);
}
//--------------------------------------------------------------------------------------//
inline uint64_t
manager::total_laps() const
{
    return m_laps + compute_total_laps();
}
//--------------------------------------------------------------------------------------//
inline manager::ofstream_t*
manager::get_ofstream(ostream_t* m_os) const
{
    return (m_os != &std::cout && m_os != &std::cerr) ? static_cast<ofstream_t*>(m_os)
                                                      : nullptr;
}
//--------------------------------------------------------------------------------------//
inline void
manager::write_report(ostream_t& os, bool ign_cutoff, bool endline)
{
    report(os, ign_cutoff, endline);
}
//--------------------------------------------------------------------------------------//
inline void
manager::report(ostream_t& os, bool ign_cutoff, bool endline) const
{
    report(&os, ign_cutoff, endline);
}
//--------------------------------------------------------------------------------------//
inline bool
manager::is_reporting_to_file() const
{
    return (m_report != &std::cout) && (m_report != &std::cerr);
}
//--------------------------------------------------------------------------------------//
inline manager::size_type
manager::size() const
{
    // return timer_data.graph().size();
    return 0UL;
}
//--------------------------------------------------------------------------------------//

// tim::manager::pointer tim::manager::f_instance = tim::manager::instance();

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
            DEBUG_PRINT_HERE("manager_deleter");
        }
        else
        {
            DEBUG_PRINT_HERE("manager_deleter");
            if(ptr)
            {
                DEBUG_PRINT_HERE("manager_deleter");
                // ptr->print();
            }
            else if(master)
            {
                DEBUG_PRINT_HERE("manager_deleter");
                // master->print();
            }
        }

        DEBUG_PRINT_HERE("manager_deleter");
        if(this_tid == master_tid)
        {
            delete ptr;
        }
        else
        {
            if(master && ptr != master)
            {
                DEBUG_PRINT_HERE("manager_deleter");
                singleton_t::remove(ptr);
            }
            DEBUG_PRINT_HERE("manager_deleter");
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
