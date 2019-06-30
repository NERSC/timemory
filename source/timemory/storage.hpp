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
 * \headerfile storage.hpp "timemory/storage.hpp"
 * Storage of the call-graph for each component. Each component has a thread-local
 * singleton that hold the call-graph. When a worker thread is deleted, it merges
 * itself back into the master thread storage. When the master thread is deleted,
 * it handles I/O (i.e. text file output, JSON output, stdout output).
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
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace details
{
template <typename StorageType>
struct storage_deleter;

template <typename ObjectType>
class storage;

template <typename _Tp>
using storage_smart_pointer = std::unique_ptr<_Tp, details::storage_deleter<_Tp>>;
template <typename _Tp>
using storage_singleton_t = singleton<_Tp, storage_smart_pointer<_Tp>>;

}  // namespace details

template <typename _Tp>
details::storage_singleton_t<_Tp>&
get_storage_singleton()
{
    using _single_t            = details::storage_singleton_t<_Tp>;
    static _single_t _instance = _single_t::instance();
    return _instance;
}

template <typename _Tp>
details::storage_singleton_t<_Tp>&
get_noninit_storage_singleton()
{
    using _single_t            = details::storage_singleton_t<_Tp>;
    static _single_t _instance = _single_t::instance_ptr();
    return _instance;
}

//======================================================================================//
// static functions that return a string identifying the data type (used in Python plot)
//
template <typename _Tp>
struct type_id
{
    template <typename Type = _Tp, enable_if_t<(std::is_integral<Type>::value), int> = 0>
    static std::string value(const Type&)
    {
        return "int";
    }

    template <typename Type                                           = _Tp,
              enable_if_t<(std::is_floating_point<Type>::value), int> = 0>
    static std::string value(const Type&)
    {
        return "float";
    }

    template <typename SubType, enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::pair<SubType, SubType>&)
    {
        return "int_pair";
    }

    template <typename SubType,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::pair<SubType, SubType>&)
    {
        return "float_pair";
    }

    template <typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::array<SubType, SubTypeSize>&)
    {
        return "int_array";
    }

    template <typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::array<SubType, SubTypeSize>&)
    {
        return "float_array";
    }

    template <typename _Up, typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::pair<std::array<SubType, SubTypeSize>, _Up>&)
    {
        return "int_array_pair";
    }

    template <typename _Up, typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::pair<std::array<SubType, SubTypeSize>, _Up>&)
    {
        return "float_array_pair";
    }
};

//======================================================================================//

template <typename ObjectType>
class storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    using this_type     = storage<ObjectType>;
    using void_type     = storage<void>;
    using string_t      = std::string;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;
    using graph_node_tuple = std::tuple<int64_t, ObjectType, string_t, int64_t>;
    using count_type       = counted_object<ObjectType>;

    //----------------------------------------------------------------------------------//
    //
    //  the node type
    //
    //----------------------------------------------------------------------------------//
    class graph_node : public graph_node_tuple
    {
    public:
        using this_type      = graph_node;
        using base_type      = graph_node_tuple;
        using obj_value_type = typename ObjectType::value_type;
        using obj_base_type  = typename ObjectType::base_type;

        int64_t&    id() { return std::get<0>(*this); }
        ObjectType& obj() { return std::get<1>(*this); }
        string_t&   prefix() { return std::get<2>(*this); }
        int64_t&    depth() { return std::get<3>(*this); }

        const int64_t&    id() const { return std::get<0>(*this); }
        const ObjectType& obj() const { return std::get<1>(*this); }
        const string_t&   prefix() const { return std::get<2>(*this); }
        const int64_t&    depth() const { return std::get<3>(*this); }

        graph_node()
        : base_type(0, ObjectType(), "", 0)
        {
        }

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {
        }

        graph_node(const int64_t& _id, const ObjectType& _obj, int64_t _depth)
        : base_type(_id, _obj, "", _depth)
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
            static_cast<obj_base_type&>(_obj) += static_cast<const obj_base_type&>(_rhs);
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
    class graph_data
    {
    public:
        using this_type = graph_data;

    public:
        graph_data() = default;

        explicit graph_data(const graph_node& rhs)
        : m_has_head(true)
        , m_depth(0)
        {
            m_head    = m_graph.set_head(rhs);
            m_depth   = 0;
            m_current = m_head;
        }

        ~graph_data() { m_graph.clear(); }

        // allow move and copy construct
        explicit graph_data(const this_type&) = default;
        graph_data& operator=(this_type&&) = default;

        // delete copy-assignment
        graph_data& operator=(const this_type&) = delete;

        bool has_head() const { return m_has_head; }

        const int64_t& depth() const { return m_depth; }
        const graph_t& graph() const { return m_graph; }

        int64_t&  depth() { return m_depth; }
        graph_t&  graph() { return m_graph; }
        iterator& current() { return m_current; }
        iterator& head() { return m_head; }

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

        inline iterator append_child(graph_node& node)
        {
            ++m_depth;
            return (m_current = m_graph.append_child(m_current, node));
        }

    private:
        bool     m_has_head = false;
        int64_t  m_depth    = 0;
        graph_t  m_graph;
        iterator m_current = nullptr;
        iterator m_head    = nullptr;
    };

public:
    //----------------------------------------------------------------------------------//
    //
    //
    //
    //----------------------------------------------------------------------------------//

    storage()
    {
        instance_count()++;
        static std::atomic<short> _once;
        short                     _once_num = _once++;
        if(_once_num > 0 && !singleton_t::is_master(this))
        {
            m_data           = graph_data(*master_instance()->current());
            m_data.head()    = master_instance()->data().current();
            m_data.current() = master_instance()->data().current();
            m_data.depth()   = master_instance()->data().depth();
        }
        else
        {
            ObjectType::initialize_policy();
        }
    }

    ~storage()
    {
        if(!singleton_t::is_master(this))
            singleton_t::master_instance()->merge(this);
    }

    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = default;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = default;

    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }
    static pointer noninit_instance() { return get_noninit_singleton().instance(); }
    static pointer noninit_master_instance()
    {
        return get_noninit_singleton().master_instance();
    }

    void print();
    bool empty() const { return (m_node_ids.size() == 0); }

    const graph_data& data() const { return m_data; }
    const graph_t&    graph() const { return m_data.graph(); }
    const int64_t&    depth() const { return m_data.depth(); }

    graph_data& data() { return m_data; }
    iterator&   current() { return m_data.current(); }
    graph_t&    graph() { return m_data.graph(); }

public:
    //----------------------------------------------------------------------------------//
    //
    iterator pop() { return m_data.pop_graph(); }

    //----------------------------------------------------------------------------------//
    //
    iterator insert(const int64_t& hash_id, const ObjectType& obj, bool& exists)
    {
        // lambda for updating settings
        auto _update = [&](iterator itr) {
            exists         = true;
            m_data.depth() = itr->depth();
            // std::cout << "[master] storage<" << ObjectType::label() << "> = " << this
            //          << ", thread = " << std::this_thread::get_id() << "..."
            //          << " updating to depth " << itr->depth() << "..." << std::endl;
            return (m_data.current() = itr);
        };

        if(m_node_ids.find(hash_id) != m_node_ids.end() &&
           m_node_ids.find(hash_id)->second->depth() == m_data.depth())
        {
            _update(m_node_ids.find(hash_id)->second);
        }

        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node node(hash_id, obj, m_data.depth());

        // lambda for inserting child
        auto _insert_child = [&]() {
            exists       = false;
            node.depth() = m_data.depth() + 1;
            // std::cout << "[master] storage<" << ObjectType::label() << "> = " << this
            //          << ", thread = " << std::this_thread::get_id() << "..."
            //          << " inserting child at depth " << node.depth() << "..."
            //          << std::endl;
            auto itr = m_data.append_child(node);
            // m_node_ids.insert(std::make_pair(hash_id, itr));
            return itr;
        };

        // if first instance
        if(!m_data.has_head())
        {
            if(this == master_instance())
            {
                // std::cout << "[master] storage<" << ObjectType::label() << "> = " <<
                // this
                //          << ", thread = " << std::this_thread::get_id() << "..."
                //          << " creating new graph_data..." << std::endl;
                m_data         = graph_data(node);
                exists         = false;
                m_data.depth() = 0;
                m_node_ids.insert(std::make_pair(hash_id, m_data.current()));
                return m_data.current();
            }
            else
            {
                m_data         = graph_data(*master_instance()->current());
                m_data.depth() = master_instance()->data().depth();
                return _insert_child();
            }
        }
        else
        {
            auto current   = m_data.current();
            auto nchildren = graph_t::number_of_children(current);

            if(hash_id == current->id())
            {
                exists = true;
                return current;
            }
            else if(nchildren == 0 && graph().number_of_siblings(current) == 0)
            {
                return _insert_child();
            }
            else if(m_data.graph().is_valid(current))
            {
                // check siblings
                for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
                {
                    // skip if current
                    if(itr == current)
                        continue;
                    // check hash id's
                    if(hash_id == itr->id())
                    {
                        return _update(itr);
                    }
                }

                // check children
                if(nchildren == 0)
                {
                    return _insert_child();
                }
                else
                {
                    auto fchild = graph_t::child(current, 0);
                    for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                    {
                        if(hash_id == itr->id())
                        {
                            return _update(itr);
                        }
                    }
                }
            }
        }
        return _insert_child();
    }

    //----------------------------------------------------------------------------------//
    //
    iterator insert(int64_t hash_id, const ObjectType& obj, const string_t& prefix)
    {
        hash_id *= (m_data.depth() >= 0) ? (m_data.depth() + 1) : 1;
        bool exists = false;
        auto itr    = insert(hash_id, obj, exists);
        if(!exists)
            itr->prefix() = prefix;
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    void set_prefix(const string_t& _prefix) { m_data.current()->prefix() = _prefix; }

protected:
    friend struct details::storage_deleter<this_type>;

    void merge()
    {
        if(!singleton_t::is_master(this))
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
                itr->data().reset();
    }

    void merge(this_type* itr);

protected:
    graph_data                            m_data;
    std::unordered_map<int64_t, iterator> m_node_ids;

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
    template <typename Archive>
    void serialize(Archive&, const unsigned int);

    // tim::component::array_serialization<ObjectType>::type == TRUE
    template <typename Archive>
    void serialize(std::true_type, Archive&, const unsigned int);

    // tim::component::array_serialization<ObjectType>::type == FALSE
    template <typename Archive>
    void serialize(std::false_type, Archive&, const unsigned int);
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

template <typename _Tp>
void
serialize_storage(const std::string& fname, const _Tp& obj, int64_t concurrency = 1)
{
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::stringstream     ss;
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive          oa(ss, opts);
        oa.setNextName("rank");
        oa.startNode();
        auto rank = tim::mpi_rank();
        oa(cereal::make_nvp("rank_id", rank));
        oa(cereal::make_nvp("concurrency", concurrency));
        oa(cereal::make_nvp("data", obj));
        oa.finishNode();
    }
    std::ofstream ofs(fname.c_str());
    ofs << ss.str() << std::endl;
}

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
