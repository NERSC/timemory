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
 * Storage class for manager
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
//--------------------------------------------------------------------------------------//

template <typename Type>
struct storage_deleter
{
    using Pointer     = std::unique_ptr<Type, storage_deleter<Type>>;
    using singleton_t = singleton<Type, Pointer>;

    void operator()(Type* ptr)
    {
        Type*           master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();

        if(std::this_thread::get_id() == master_tid)
            delete ptr;
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

}  // namespace details

//======================================================================================//

template <typename ObjectType>
class graph_storage
{
public:
    using string_t = std::string;
    //----------------------------------------------------------------------------------//
    //
    //  the node type
    //
    //----------------------------------------------------------------------------------//
    struct graph_node : public std::tuple<intmax_t, ObjectType, string_t>
    {
        using this_type = graph_node;
        using base_type = std::tuple<intmax_t, ObjectType, string_t>;

        intmax_t&   id() { return std::get<0>(*this); }
        ObjectType& obj() { return std::get<1>(*this); }
        string_t&   prefix() { return std::get<2>(*this); }

        const intmax_t&   id() const { return std::get<0>(*this); }
        const ObjectType& obj() const { return std::get<1>(*this); }
        const string_t&   prefix() const { return std::get<2>(*this); }

        graph_node()
        : base_type(0, ObjectType(), "")
        {
        }

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {
        }

        graph_node(const intmax_t& _id, const ObjectType& _obj)
        : base_type(_id, _obj, "")
        {
        }

        ~graph_node() {}
        // explicit graph_node(const this_type&) = default;
        // explicit graph_node(this_type&&)      = default;
        // graph_node& operator=(const this_type&) = default;
        // graph_node& operator=(this_type&&) = default;

        bool operator==(const graph_node& rhs) const { return id() == rhs.id(); }
        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            obj() += rhs.obj();
            return *this;
        }
    };

    //----------------------------------------------------------------------------------//
    using this_type      = graph_storage<ObjectType>;
    using void_type      = graph_storage<void>;
    using graph_t        = tim::graph<graph_node>;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;
    using smart_pointer = std::unique_ptr<this_type, details::storage_deleter<this_type>>;
    using singleton_t   = singleton<this_type, smart_pointer>;
    using pointer       = typename singleton_t::pointer;
    using auto_lock_t   = typename singleton_t::auto_lock_t;

    //----------------------------------------------------------------------------------//
    //
    //  graph instance + current node + head node
    //
    //----------------------------------------------------------------------------------//
    struct graph_data
    {
        using this_type  = graph_data;
        intmax_t m_depth = -1;
        graph_t  m_graph;
        iterator m_current;
        iterator m_head;

        graph_data()
        : m_depth(-1)
        {
        }

        explicit graph_data(const graph_node& rhs)
        : m_depth(0)
        {
            m_head    = m_graph.set_head(rhs);
            m_current = m_head;
        }

        ~graph_data() { m_graph.clear(); }

        graph_data(const this_type&) = default;
        graph_data& operator=(const this_type&) = default;
        graph_data& operator=(this_type&&) = default;

        intmax_t& depth() { return m_depth; }
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
                m_current = graph_t::parent(m_current);
                --m_depth;
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
    //----------------------------------------------------------------------------------//
    //
    //
    //
    //----------------------------------------------------------------------------------//

    graph_storage()
    {
        static std::atomic<short> _once;
        short                     _once_num = _once++;
        if(_once_num > 0 && !singleton_t::is_master(this))
        {
            m_data           = graph_data(*master_instance()->current());
            m_data.head()    = master_instance()->data().current();
            m_data.current() = master_instance()->data().current();
            m_data.depth()   = master_instance()->data().depth();
        }
    }

    ~graph_storage()
    {
        if(!singleton_t::is_master(this))
            singleton_t::master_instance()->merge(this);
        else
        {
            print();
        }
    }
    explicit graph_storage(const this_type&) = delete;
    graph_storage(this_type&&)               = default;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = default;

    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }

    void print();

    const graph_data& data() const { return m_data; }
    const graph_t&    graph() const { return m_data.graph(); }

    graph_data& data() { return m_data; }
    iterator&   current() { return m_data.current(); }
    graph_t&    graph() { return m_data.graph(); }

public:
    //----------------------------------------------------------------------------------//
    //
    iterator pop() { return m_data.pop_graph(); }

    //----------------------------------------------------------------------------------//
    //
    iterator insert(const intmax_t& hash_id, const ObjectType& obj, bool& exists)
    {
        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node node(hash_id, obj);

        // lambda for inserting child
        auto _insert_child = [&]() {
            exists = false;
            return m_data.append_child(node);
        };

        // lambda for updating settings
        auto _update = [&](iterator itr) {
            exists = true;
            return (m_data.current() = itr);
        };

        // if first instance
        if(m_data.depth() < 0)
        {
            if(this == master_instance())
            {
                m_data = graph_data(node);
                exists = false;
                return m_data.current();
            }
            else
            {
                m_data = graph_data(*master_instance()->current());
                return _insert_child();
            }
        }
        else
        {
            auto current = m_data.current();

            if(hash_id == current->id())
            {
                exists = true;
                return current;
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
                        if(hash_id == itr->id())
                        {
                            return _update(itr);
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
                    if(hash_id == itr->id())
                    {
                        return _update(itr);
                    }
                }

                // check children
                auto nchildren = graph_t::number_of_children(current);
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
                    if(!exists)
                        return _insert_child();
                }
            }
        }
        return _insert_child();
    }

    void set_prefix(const string_t& _prefix) { m_data.current()->prefix() = _prefix; }

protected:
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

    void merge(this_type* itr)
    {
        if(itr == this)
            return;

        // create lock but don't immediately lock
        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

        // lock if not already owned
        if(!l.owns_lock())
            l.lock();

        auto _this_beg = graph().begin();
        auto _this_end = graph().end();

        bool _merged = false;
        for(auto _this_itr = _this_beg; _this_itr != _this_end; ++_this_itr)
        {
            if(_this_itr == itr->data().head())
            {
                auto _iter_beg = itr->graph().begin();
                auto _iter_end = itr->graph().end();
                graph().merge(_this_itr, _this_end, _iter_beg, _iter_end, false, true);
                _merged = true;
                break;
            }
        }

        if(_merged)
        {
            typedef decltype(_this_beg) predicate_type;
            auto _reduce = [](predicate_type lhs, predicate_type rhs) { *lhs += *rhs; };
            _this_beg    = graph().begin();
            _this_end    = graph().end();
            graph().reduce(_this_beg, _this_end, _this_beg, _this_end, _reduce);
        }
        else
        {
            auto_lock_t lerr(type_mutex<decltype(std::cerr)>());
            std::cerr << "Failure to merge graphs!" << std::endl;
            auto g = graph();
            graph().insert_subgraph_after(m_data.current(), itr->data().head());
            // itr->graph()
        }
    }

protected:
    graph_data m_data;

private:
    static singleton_t& get_singleton()
    {
        static singleton_t _instance = singleton_t::instance();
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#include "timemory/component_operations.hpp"
#include "timemory/environment.hpp"

//======================================================================================//

template <typename ObjectType>
void
tim::graph_storage<ObjectType>::print()
{
    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
    }
    else
    {
        merge();
        m_data.current() = m_data.head();
        intmax_t _width  = ObjectType::get_width();
        for(const auto& itr : m_data.graph())
        {
            intmax_t _len = itr.prefix().length();
            _width        = std::max(_len, _width);
        }

        std::stringstream _oss;
        for(const auto& itr : m_data.graph())
        {
            auto _obj    = itr.obj();
            auto _prefix = itr.prefix();
            auto _laps   = _obj.laps;
            component::print<ObjectType>(_obj, _oss, _prefix, _laps, _width, true);
        }

        auto          label = ObjectType::label();
        auto          fname = tim::env::compose_output_filename(label, ".txt");
        std::ofstream ofs(fname.c_str());
        if(ofs)
        {
            printf("[graph_storage<%s>::%s @ %i]> Outputting '%s'...\n",
                   ObjectType::label().c_str(), __FUNCTION__, __LINE__, fname.c_str());
            ofs << _oss.str();
            ofs.close();
        }
        else
        {
            auto_lock_t l(type_mutex<decltype(std::cout)>());
            std::cout << _oss.str();
        }

        // fname = tim::env::compose_output_filename(label, ".json");
    }
}
//======================================================================================//
