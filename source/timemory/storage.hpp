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
//
//
//
//======================================================================================//
template <typename _Key, typename _Mapped>
using uomap = std::unordered_map<_Key, _Mapped>;

class key_identifier_storage
{
public:
    using this_type      = key_identifier_storage;
    using string_t       = std::string;
    using singleton_t    = singleton<this_type, std::shared_ptr<this_type>>;
    using pointer        = typename singleton_t::pointer;
    using smart_pointer  = typename singleton_t::smart_pointer;
    using list_t         = typename singleton_t::list_t;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using key_t          = intmax_t;
    using value_t        = string_t;
    using data_type      = std::map<key_t, value_t>;
    using iterator       = typename data_type::iterator;
    using const_iterator = typename data_type::const_iterator;
    using size_type      = typename data_type::size_type;

public:
    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }

public:
    key_identifier_storage() = default;
    ~key_identifier_storage()
    {
        if(!singleton_t::is_master(this))
        {
            singleton_t::master_instance()->merge(this);
        }
        else
        {
        }
    }

    iterator       begin() { return m_data.begin(); }
    iterator       end() { return m_data.end(); }
    const_iterator begin() const { return m_data.begin(); }
    const_iterator end() const { return m_data.end(); }
    size_type      size() const { return m_data.size(); }

    data_type&       data() { return m_data; }
    const data_type& data() const { return m_data; }

    value_t& operator[](const key_t& key) { return m_data[key]; }
    bool     count(const key_t& key) const { return m_data.count(key); }
    bool     exists(const key_t& key) const { return m_data.find(key) != m_data.end(); }
    void     insert(const key_t& key, const value_t& val)
    {
        if(!exists(key))
            m_data[key] = val;
        else if(m_data[key].length() < val.length())
            m_data[key] = val;
        // singleton_t::auto_lock_t l(type_mutex<decltype(std::cout)>());
        // std::cout << "key = " << key << ", val = " << val << ", size = " << size()
        //          << std::endl;
    }
    value_t get(const key_t& key) const
    {
        if(exists(key))
        {
            return m_data.find(key)->second;
        }
        else
        {
            auto _get_prefix = []() {
                if(!mpi_is_initialized())
                    return string_t("> ");
                // prefix spacing
                static uint16_t width = 1;
                if(mpi_size() > 9)
                    width = std::max(width, (uint16_t)(log10(mpi_size()) + 1));
                std::stringstream ss;
                ss.fill('0');
                ss << "|" << std::setw(width) << mpi_rank() << "> ";
                return ss.str();
            };
            static string_t _prefix = _get_prefix();
            return _prefix;
        }
    }

    void merge(this_type* itr)
    {
        if(itr == this)
            return;

        auto_lock_t l(singleton_t::get_mutex());
        // printf("Merging %p into %p (master = %p) @ %s:%i...\n", (void*) itr, (void*)
        // this,
        //       (void*) singleton_t::master_instance_ptr(), __PRETTY_FUNCTION__,
        //       __LINE__);
        for(const auto& mitr : itr->data())
            insert(mitr.first, mitr.second);
    }

private:
    static singleton_t& get_singleton()
    {
        static singleton_t _instance = singleton_t::instance();
        return _instance;
    }

private:
    data_type m_data;
};

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
    struct graph_node : public std::tuple<intmax_t, intmax_t, ObjectType, string_t>
    {
        using this_type = graph_node;
        using base_type = std::tuple<intmax_t, intmax_t, ObjectType, string_t>;

        intmax_t&   id() { return std::get<0>(*this); }
        intmax_t&   laps() { return std::get<1>(*this); }
        ObjectType& obj() { return std::get<2>(*this); }
        string_t&   prefix() { return std::get<3>(*this); }

        const intmax_t&   id() const { return std::get<0>(*this); }
        const intmax_t&   laps() const { return std::get<1>(*this); }
        const ObjectType& obj() const { return std::get<2>(*this); }
        const string_t&   prefix() const { return std::get<3>(*this); }

        graph_node()
        : base_type(0, 0, ObjectType(), "")
        {
        }

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {
        }

        graph_node(const intmax_t& _id, const ObjectType& _obj)
        : base_type(_id, 0, _obj, "")
        {
        }

        ~graph_node()                         = default;
        explicit graph_node(const this_type&) = default;
        explicit graph_node(this_type&&)      = default;
        graph_node& operator=(const this_type&) = default;
        graph_node& operator=(this_type&&) = default;

        bool operator==(const graph_node& rhs) const { return id() == rhs.id(); }
        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            obj() += rhs.obj();
            return *this;
        }

        intmax_t operator++() { return ++laps(); }
        intmax_t operator++(int) { return laps()++; }
        intmax_t operator--() { return --laps(); }
        intmax_t operator--(int) { return laps()--; }
    };

    //----------------------------------------------------------------------------------//
    using key_id_pointer = std::shared_ptr<key_identifier_storage>;
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
        using this_type    = graph_data;
        intmax_t m_depth   = -1;
        graph_t  m_graph   = graph_t();
        iterator m_current = nullptr;
        iterator m_head    = nullptr;

        graph_data() {}
        explicit graph_data(const graph_node& rhs)
        : m_depth(0)
        , m_graph(rhs)
        , m_current(m_graph.begin())
        , m_head(m_graph.begin())
        {
        }
        ~graph_data()                = default;
        graph_data(const this_type&) = default;
        graph_data(this_type&&)      = default;
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

        inline void clear()
        {
            m_depth = -1;
            m_graph.clear();
            ;
            m_graph   = graph_t();
            m_current = nullptr;
            m_head    = nullptr;
        }

        inline void pop_graph()
        {
            --m_depth;
            m_current = graph_t::parent(m_current);
        }

        inline void insert(const graph_node& node)
        {
            ++m_depth;
            m_current = m_graph.append_child(m_current, node);
            m_current->operator++();
        }
    };

public:
    //----------------------------------------------------------------------------------//
    //
    //
    //
    //----------------------------------------------------------------------------------//

    graph_storage() {}
    ~graph_storage()
    {
        if(!singleton_t::is_master(this))
            singleton_t::master_instance()->merge(this);
        else
        {
            print();
        }
    }

    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }

    void print();

    iterator pop()
    {
        if(!m_data.graph().is_head(m_data.current()))
        {
            m_data.pop_graph();
        }
        return m_data.current();
    }

    iterator insert(const intmax_t& hash_id, const ObjectType& obj, bool& exists)
    {
        graph_node node(hash_id, obj);
        auto       _insert_child = [&]() { m_data.insert(node); };

        if(m_data.depth() < 0)
        {
            if(this == master_instance())
            {
                m_data = graph_data(node);
            }
            else
            {
                m_data = graph_data(*master_instance()->current());
                _insert_child();
            }
        }
        else
        {
            using sibling_itr = typename graph_t::sibling_iterator;
            auto nchildren    = graph_t::number_of_children(m_data.current());
            if(nchildren == 0)
            {
                _insert_child();
            }
            else
            {
                auto fchild = graph_t::child(m_data.current(), 0);
                for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                {
                    if(node == *itr)
                    {
                        m_data.current() = itr;
                        exists           = true;
                        break;
                    }
                }
                if(!exists)
                    _insert_child();
            }
        }
        return m_data.current();
    }

    void set_prefix(const string_t& _prefix) { m_data.current()->prefix() = _prefix; }

protected:
    explicit graph_storage(const this_type&) = default;
    explicit graph_storage(this_type&&)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&& rhs) = default;

    const graph_data& data() const { return m_data; }
    const graph_t&    graph() const { return m_data.graph(); }

    graph_data& data() { return m_data; }
    iterator&   current() { return m_data.current(); }
    iterator&   head() { return m_data.head(); }
    graph_t&    graph() { return m_data.graph(); }

    void merge()
    {
        if(!singleton_t::is_master(this))
            return;

        auto m_children = singleton_t::children();
        if(m_children.size() == 0)
            return;

        for(auto& itr : m_children)
            merge(itr);

        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
        if(!l.owns_lock())
            l.lock();

        for(auto& itr : m_children)
            if(itr != this)
                itr->data().clear();
    }

    void merge(this_type* itr)
    {
        if(itr == this)
            return;

        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
        if(!l.owns_lock())
            l.lock();

        // printf("Merging %p into %p (master = %p) @ %s:%i...\n", (void*) itr, (void*)
        // this,
        //       (void*) singleton_t::master_instance_ptr(), __PRETTY_FUNCTION__,
        //       __LINE__);

        auto _this_beg = graph().begin();
        auto _this_end = graph().end();

        for(auto _this_itr = _this_beg; _this_itr != _this_end; ++_this_itr)
        {
            if(_this_itr == itr->head())
            {
                auto _iter_beg = itr->graph().begin();
                auto _iter_end = itr->graph().end();
                graph().merge(_this_itr, _this_end, _iter_beg, _iter_end, false, true);
                break;
            }
        }

        typedef decltype(_this_beg) predicate_type;
        auto _reduce = [](predicate_type lhs, predicate_type rhs) { *lhs += *rhs; };
        _this_beg    = graph().begin();
        _this_end    = graph().end();
        graph().reduce(_this_beg, _this_end, _this_beg, _this_end, _reduce);
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
        intmax_t _width = ObjectType::get_width();
        for(const auto& itr : m_data.graph())
        {
            intmax_t _len = itr.prefix().length();
            _width        = std::max(_len, _width);
        }

        std::stringstream _oss;
        for(const auto& itr : m_data.graph())
        {
            const ObjectType& _obj    = itr.obj();
            auto              _prefix = itr.prefix();
            auto              _laps   = itr.laps();
            component::print<ObjectType>(_obj, _oss, _prefix, _laps, _width, true);
        }

        auto          label = ObjectType::label();
        auto          fname = tim::env::compose_output_filename(label, ".txt");
        std::ofstream ofs(fname.c_str());
        if(ofs)
        {
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
