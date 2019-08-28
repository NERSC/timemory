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
 * \headerfile storage.hpp "timemory/utility/storage.hpp"
 * Storage of the call-graph for each component. Each component has a thread-local
 * singleton that hold the call-graph. When a worker thread is deleted, it merges
 * itself back into the master thread storage. When the master thread is deleted,
 * it handles I/O (i.e. text file output, JSON output, stdout output).
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/mpi.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/utility/graph.hpp"
#include "timemory/utility/graph_data.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/type_id.hpp"
#include "timemory/utility/utility.hpp"

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
//--------------------------------------------------------------------------------------//
//
namespace cupti
{
struct result;
}  // cupti

//--------------------------------------------------------------------------------------//
//
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

//--------------------------------------------------------------------------------------//

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
    using count_type    = counted_object<ObjectType>;

    using graph_node_tuple = std::tuple<int64_t, ObjectType, std::string, int64_t>;

    class graph_node : public graph_node_tuple
    {
    public:
        using this_type       = graph_node;
        using base_type       = graph_node_tuple;
        using data_value_type = typename ObjectType::value_type;
        using data_base_type  = typename ObjectType::base_type;
        using string_t        = std::string;

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
            obj().activate_noop();
        }

        explicit graph_node(base_type&& _base)
        : base_type(std::forward<base_type>(_base))
        {
            obj().activate_noop();
        }

        graph_node(const int64_t& _id, const ObjectType& _obj, int64_t _depth)
        : base_type(_id, _obj, "", _depth)
        {
            obj().activate_noop();
        }

        graph_node(const int64_t& _id, const ObjectType& _obj, const string_t& _tag,
                   int64_t _depth)
        : base_type(_id, _obj, _tag, _depth)
        {
            obj().activate_noop();
        }

        ~graph_node() {}
        // explicit graph_node(const this_type&) = default;
        // explicit graph_node(this_type&&)      = default;
        // graph_node& operator=(const this_type&) = default;
        // graph_node& operator=(this_type&&) = default;

        bool operator==(const graph_node& rhs) const
        {
            return ((id() == rhs.id() && depth() == rhs.depth()) ||
                    (prefix() == rhs.prefix() && depth() == rhs.depth()));
        }

        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }

        graph_node& operator+=(const graph_node& rhs)
        {
            auto&       _obj = obj();
            const auto& _rhs = rhs.obj();
            static_cast<data_base_type&>(_obj) +=
                static_cast<const data_base_type&>(_rhs);
            return *this;
        }
    };

public:
    using graph_node_t   = graph_node;
    using graph_data_t   = graph_data<graph_node_t>;
    using graph_t        = typename graph_data_t::graph_t;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;

    static pointer instance() { return get_singleton().instance(); }
    static pointer master_instance() { return get_singleton().master_instance(); }
    static pointer noninit_instance() { return get_noninit_singleton().instance(); }
    static pointer noninit_master_instance()
    {
        return get_noninit_singleton().master_instance();
    }

public:
    //----------------------------------------------------------------------------------//
    //
    storage()
    {
        instance_count()++;
        static std::atomic<short> _once;
        short                     _once_num = _once++;
        if(_once_num > 0 && !singleton_t::is_master(this))
        {
            ObjectType::thread_init_policy();
        }
        else
        {
            ObjectType::global_init_policy();
            ObjectType::thread_init_policy();
        }
    }

    //----------------------------------------------------------------------------------//
    //
    ~storage()
    {
        if(!singleton_t::is_master(this))
            singleton_t::master_instance()->merge(this);
        delete __graph_data_instance;
        __graph_data_instance = nullptr;
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
    // there is always a head node that should not be counted
    //
    bool empty() const { return (_data().graph().size() <= 1); }

    //----------------------------------------------------------------------------------//
    // there is always a head node that should not be counted
    //
    inline size_t size() const { return _data().graph().size() - 1; }

    //----------------------------------------------------------------------------------//
    //
    iterator pop() { return _data().pop_graph(); }

    //----------------------------------------------------------------------------------//
    //
    iterator insert(const int64_t& hash_id, const ObjectType& obj, bool& exists)
    {
        // lambda for updating settings
        auto _update = [&](iterator itr) {
            exists          = true;
            _data().depth() = itr->depth();
            return (_data().current() = itr);
        };

        if(m_node_ids.find(hash_id) != m_node_ids.end() &&
           m_node_ids.find(hash_id)->second->depth() == _data().depth())
        {
            _update(m_node_ids.find(hash_id)->second);
        }

        using sibling_itr = typename graph_t::sibling_iterator;
        graph_node_t node(hash_id, obj, _data().depth());

        // lambda for inserting child
        auto _insert_child = [&]() {
            exists       = false;
            node.depth() = _data().depth() + 1;
            auto itr     = _data().append_child(node);
            return itr;
        };

        // if first instance
        if(!_data().has_head() || (this == master_instance() && m_node_ids.size() == 0))
        {
            if(this == master_instance())
            {
                _data()         = graph_data_t(node);
                exists          = false;
                _data().depth() = 0;
                m_node_ids.insert(std::make_pair(hash_id, _data().current()));
                return _data().current();
            }
            else
            {
                _data()           = graph_data_t(*master_instance()->current());
                _data().head()    = master_instance()->data().current();
                _data().current() = master_instance()->data().current();
                _data().depth()   = master_instance()->data().depth();
                return _insert_child();
            }
        }
        else
        {
            auto current   = _data().current();
            auto nchildren = graph_t::number_of_children(current);

            if(hash_id == current->id())
            {
                exists = true;
                return current;
            }
            else if(nchildren == 0 && graph().number_of_siblings(current) == 0)
                return _insert_child();
            else if(_data().graph().is_valid(current))
            {
                // check siblings
                for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
                {
                    // skip if current
                    if(itr == current)
                        continue;
                    // check hash id's
                    if(hash_id == itr->id())
                        return _update(itr);
                }

                // check children
                if(nchildren == 0)
                    return _insert_child();
                else
                {
                    auto fchild = graph_t::child(current, 0);
                    for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
                    {
                        if(hash_id == itr->id())
                            return _update(itr);
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
        hash_id *= (_data().depth() >= 0) ? (_data().depth() + 1) : 1;
        bool exists = false;
        auto itr    = insert(hash_id, obj, exists);
        if(!exists)
            itr->prefix() = prefix;
        return itr;
    }

    //----------------------------------------------------------------------------------//
    //
    void set_prefix(const string_t& _prefix) { _data().current()->prefix() = _prefix; }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        typename tim::trait::array_serialization<ObjectType>::type type;
        constexpr auto uses_external = trait::external_output_handling<ObjectType>::value;
        if(!uses_external)
            serialize<Archive>(type, ar, version);
    }

    //----------------------------------------------------------------------------------//
    //
    void print()
    {
        typename trait::external_output_handling<ObjectType>::type type;
        external_print(type);
    }

public:
    //----------------------------------------------------------------------------------//
    //
    const graph_data_t& data() const { return _data(); }
    const graph_t&      graph() const { return _data().graph(); }
    const int64_t&      depth() const { return _data().depth(); }

    //----------------------------------------------------------------------------------//
    //
    graph_data_t& data() { return _data(); }
    iterator&     current() { return _data().current(); }
    graph_t&      graph() { return _data().graph(); }

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
                itr->data().clear();
    }

    void merge(this_type* itr);

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

    // tim::trait::array_serialization<ObjectType>::type == TRUE
    template <typename Archive>
    void serialize(std::true_type, Archive&, const unsigned int);

    // tim::trait::array_serialization<ObjectType>::type == FALSE
    template <typename Archive>
    void serialize(std::false_type, Archive&, const unsigned int);

    // tim::trait::external_output_handling<ObjectType>::type == TRUE
    void external_print(std::true_type);

    // tim::trait::external_output_handling<ObjectType>::type == FALSE
    void external_print(std::false_type);

    graph_data_t& _data()
    {
        if(__graph_data_instance == nullptr && !singleton_t::is_master(this))
        {
            __graph_data_instance = new graph_data_t(*master_instance()->current());
            __graph_data_instance->head()    = master_instance()->current();
            __graph_data_instance->current() = master_instance()->current();
            __graph_data_instance->depth()   = master_instance()->depth();
        }
        else if(__graph_data_instance == nullptr)
        {
            graph_node_t node(0, ObjectType(), "> [tot] total", 0);
            __graph_data_instance          = new graph_data_t(node);
            __graph_data_instance->depth() = 0;
            m_node_ids.insert(std::make_pair(0, __graph_data_instance->current()));
        }
        return *__graph_data_instance;
    }

    const graph_data_t& _data() const { return const_cast<this_type*>(this)->_data(); }

    mutable graph_data_t*                 __graph_data_instance = nullptr;
    std::unordered_map<int64_t, iterator> m_node_ids;
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
        auto rank = tim::mpi::rank();
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
