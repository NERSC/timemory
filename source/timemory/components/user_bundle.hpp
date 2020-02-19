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
//

/** \file components/user_bundle.hpp
 * \headerfile components/user_bundle.hpp "timemory/components/user_bundle.hpp"
 * Defines the user_bundle component which can be used to inject components
 * at runtime. There are very useful for dynamically assembling collections
 * of tools at runtime
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>

//======================================================================================//

namespace tim
{
namespace component
{
#if defined(TIMEMORY_EXTERN_TEMPLATES) && !defined(TIMEMORY_BUILD_EXTERN_TEMPLATE)

extern template struct base<user_bundle<10101, api::native_tag>, void>;
extern template struct base<user_bundle<11011, api::native_tag>, void>;

#endif

//--------------------------------------------------------------------------------------//

template <size_t Idx, typename Tag>
struct user_bundle : public base<user_bundle<Idx, Tag>, void>
{
public:
    using mutex_t  = std::mutex;
    using lock_t   = std::unique_lock<mutex_t>;
    using string_t = std::string;

    using value_type   = void;
    using this_type    = user_bundle<Idx, Tag>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;

    using start_func_t  = std::function<void*(const string_t&, bool)>;
    using stop_func_t   = std::function<void(void*)>;
    using get_func_t    = std::function<void(void*, void*&, size_t)>;
    using delete_func_t = std::function<void(void*)>;

    static string_t   label() { return "user_bundle"; }
    static string_t   description() { return "user-defined bundle of tools"; }
    static value_type record() {}

    static void global_init(storage_type*);

    struct bundle_data
    {
        template <typename StartF, typename StopF, typename GetF, typename DelF>
        bundle_data(size_t _typeid, StartF&& _start, StopF&& _stop, GetF&& _get,
                    DelF&& _del)
        : m_typeid(_typeid)
        , m_start(std::move(_start))
        , m_stop(std::move(_stop))
        , m_get(std::move(_get))
        , m_del(std::move(_del))
        {}

        ~bundle_data()
        {
            if(m_data)
            {
                m_stop(m_data);
                m_del(m_data);
            }
        }

        void start(const string_t& _prefix, bool _flat)
        {
            if(m_data)
            {
                stop();
                cleanup();
            }
            m_data = m_start(_prefix, _flat);
        }

        void stop()
        {
            if(m_data)
                m_stop(m_data);
        }

        void cleanup()
        {
            if(m_data && !m_copy)
                m_del(m_data);
            m_data = nullptr;
        }

        void get(void*& ptr, size_t _hash) const
        {
            if(m_data)
                m_get(m_data, ptr, _hash);
        }

        void set_copy(bool val) { m_copy = val; }

        bool          m_copy   = false;
        size_t        m_typeid = 0;
        void*         m_data   = nullptr;
        start_func_t  m_start  = [](const string_t&, bool) { return nullptr; };
        stop_func_t   m_stop   = [](void*) {};
        get_func_t    m_get    = [](void*, void*&, size_t) {};
        delete_func_t m_del    = [](void*) {};
    };

    using bundle_data_vec_t = std::vector<bundle_data>;
    using typeid_set_t      = std::set<size_t>;

    static inline size_t get_hash(string_t&& key)
    {
        return ::tim::get_hash(std::forward<string_t>(key));
    }

    static size_t bundle_size() { return get_bundle_data().size(); }

protected:
    template <typename T>
    struct variadic_toolset;

    template <typename T>
    friend struct variadic_toolset;

    template <template <typename...> class Tuple, typename... T>
    struct variadic_toolset<Tuple<T...>>
    {
        template <typename U, typename... Tail,
                  enable_if_t<(sizeof...(Tail) == 0), int> = 0>
        static void add_typeids(typeid_set_t& _typeids)
        {
            _typeids.insert(get_hash(demangle<U>()));
        }

        template <typename U, typename... Tail,
                  enable_if_t<(sizeof...(Tail) > 0), int> = 0>
        static void add_typeids(typeid_set_t& _typeids)
        {
            add_typeids<U>(_typeids);
            add_typeids<Tail...>(_typeids);
        }

        static void add_typeids(typeid_set_t& _typeids) { add_typeids<T...>(_typeids); }

        template <typename... Args>
        static void configure(Args&&... args)
        {
            add_typeids<T...>(get_typeids());
            this_type::configure<T...>(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static void insert(this_type& obj, Args&&... args)
        {
            add_typeids<T...>(obj.m_typeids);
            obj.template insert<T...>(std::forward<Args>(args)...);
        }
    };

    template <template <typename...> class Tuple>
    struct variadic_toolset<Tuple<>>
    {
        static void add_typeids(typeid_set_t&) {}
        template <typename... Args>
        static void configure(Args&&...)
        {}
    };

public:
    //----------------------------------------------------------------------------------//
    //  Captures the statically-defined data so these can be changed without
    //  affecting this instance
    //
    user_bundle() = default;

    explicit user_bundle(const string_t& _prefix, bool _flat = false)
    : m_flat(_flat)
    , m_prefix(_prefix)
    {}

    user_bundle(const user_bundle& rhs)
    : m_flat(rhs.m_flat)
    , m_prefix(rhs.m_prefix)
    , m_typeids(rhs.m_typeids)
    , m_bundle(rhs.m_bundle)
    {
        for(auto& itr : m_bundle)
            itr.set_copy(true);
    }

    user_bundle(const string_t& _prefix, const bundle_data_vec_t& _bundle_vec,
                bool _flat = false)
    : m_flat(_flat)
    , m_prefix(_prefix)
    , m_bundle(_bundle_vec)
    {}

    ~user_bundle()
    {
        for(auto& itr : m_bundle)
            itr.cleanup();
    }

    user_bundle& operator=(const user_bundle& rhs)
    {
        if(this == &rhs)
            return *this;

        m_flat    = rhs.m_flat;
        m_prefix  = rhs.m_prefix;
        m_typeids = rhs.m_typeids;
        m_bundle  = rhs.m_bundle;
        for(auto& itr : m_bundle)
            itr.set_copy(true);
    }

    user_bundle(user_bundle&&) = default;
    user_bundle& operator=(user_bundle&&) = default;

public:
    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific component
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) == 0), int>                = 0,
              enable_if_t<(trait::is_available<Toolset>::value), int> = 0,
              enable_if_t<(Toolset::is_component), char>              = 0>
    static void configure(bool _flat, Args&&... args)
    {
        DEBUG_PRINT_HERE("%s", demangle<Toolset>().c_str());

        if(!trait::is_available<Toolset>::value)
            return;

        auto _typeid_hash = get_hash(demangle<Toolset>());
        if(get_typeids().count(_typeid_hash) > 0)
            return;

        internal_init<Toolset>();

        using Toolset_t = component_tuple<Toolset>;
        auto _start     = [=, &args...](const string_t& _prefix, bool _argflat) {
            Toolset_t* _result = new Toolset_t(_prefix, true, _flat || _argflat,
                                               std::forward<Args>(args)...);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->stop();
        };

        auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
            if(_hash == _typeid_hash && v_result && !ptr)
            {
                Toolset_t* _result = static_cast<Toolset_t*>(v_result);
                ptr                = static_cast<void*>(_result->template get<Toolset>());
            }
        };

        auto _del = [=](void* v_result) {
            if(v_result)
            {
                Toolset_t* _result = static_cast<Toolset_t*>(v_result);
                delete _result;
            }
        };

        lock_t lk(get_lock());
        get_typeids().insert(_typeid_hash);
        get_bundle_data().push_back({ _typeid_hash, std::move(_start), std::move(_stop),
                                      std::move(_get), std::move(_del) });
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) == 0), int>    = 0,
              enable_if_t<!(Toolset::is_component), char> = 0>
    static void configure(bool _flat, Args&&... args)
    {
        DEBUG_PRINT_HERE("%s", demangle<Toolset>().c_str());

        if(Toolset::size() == 0)
            return;

        auto _typeid_hash = get_hash(demangle<Toolset>());
        if(get_typeids().count(_typeid_hash) > 0)
            return;

        internal_init();

        auto _start = [=, &args...](const string_t& _prefix, bool _argflat) {
            constexpr bool is_component_type = Toolset::is_component_type;
            Toolset*       _result           = (is_component_type)
                                   ? new Toolset(_prefix, true, _flat || _argflat,
                                                 std::forward<Args>(args)...)
                                   : new Toolset(_prefix, _flat || _argflat,
                                                 settings::destructor_report(),
                                                 std::forward<Args>(args)...);
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            Toolset* _result = static_cast<Toolset*>(v_result);
            _result->stop();
        };

        auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
            if(v_result && !ptr)
            {
                Toolset* _result = static_cast<Toolset*>(v_result);
                _result->get(ptr, _hash);
            }
        };

        auto _del = [=](void* v_result) {
            if(v_result)
            {
                Toolset* _result = static_cast<Toolset*>(v_result);
                delete _result;
            }
        };

        lock_t lk(get_lock());
        variadic_toolset<Toolset>::add_typeids(get_typeids());
        get_bundle_data().push_back({ _typeid_hash, std::move(_start), std::move(_stop),
                                      std::move(_get), std::move(_del) });
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a variadic list of tools
    //
    template <typename Head, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) > 0), int> = 0>
    static void configure(bool flat, Args&&... args)
    {
        DEBUG_PRINT_HERE("%s", demangle<Head>().c_str());

        configure<Head>(flat, std::forward<Args>(args)...);
        configure<Tail...>(flat, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) == 0), int>                 = 0,
              enable_if_t<!(trait::is_available<Toolset>::value), int> = 0>
    static void configure(bool, Args&&...)
    {}

    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) > 0), int>                  = 0,
              enable_if_t<!(trait::is_available<Toolset>::value), int> = 0>
    static void configure(bool flat, Args&&... _args)
    {
        configure<Tail...>(flat, std::forward<Args>(_args)...);
    }

    template <typename Toolset, typename... Tail>
    static void configure()
    {
        configure<Toolset, Tail...>(settings::flat_profile());
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly clear the previous configurations
    //
    static void reset()
    {
        lock_t lk(get_lock());
        get_bundle_data().clear();
        get_typeids().clear();
    }

public:
    //----------------------------------------------------------------------------------//
    //  Member functions
    //
    void start()
    {
        base_type::set_started();
        for(auto& itr : m_bundle)
            itr.start(m_prefix, m_flat);
    }

    void stop()
    {
        for(auto& itr : m_bundle)
            itr.stop();
        base_type::set_stopped();
    }

    void clear()
    {
        if(base_type::is_running)
            stop();
        m_typeids.clear();
        m_bundle.clear();
    }

    template <typename T>
    T* get()
    {
        auto  _typeid_hash = get_hash(demangle<T>());
        void* void_ptr     = nullptr;
        for(auto& itr : m_bundle)
        {
            itr.get(void_ptr, _typeid_hash);
            if(void_ptr)
                return void_ptr;
        }
        return static_cast<T*>(void_ptr);
    }

    void get(void*& ptr, size_t _hash) const
    {
        for(const auto& itr : m_bundle)
        {
            itr.get(ptr, _hash);
            if(ptr)
                break;
        }
    }

    void* get() { return nullptr; }

    void set_prefix(const string_t& _prefix) { m_prefix = _prefix; }

    size_t size() const { return m_bundle.size(); }

public:
    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific component
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) == 0), int>                = 0,
              enable_if_t<(trait::is_available<Toolset>::value), int> = 0,
              enable_if_t<(Toolset::is_component), char>              = 0>
    void insert(bool _flat, Args&&... args)
    {
        DEBUG_PRINT_HERE("%s", demangle<Toolset>().c_str());

        auto _typeid_hash = get_hash(demangle<Toolset>());
        if(m_typeids.count(_typeid_hash) > 0)
            return;

        internal_init<Toolset>();

        using Toolset_t = component_tuple<Toolset>;
        auto _start     = [=, &args...](const string_t& _prefix, bool _argflat) {
            Toolset_t* _result = new Toolset_t(_prefix, true, _flat || _argflat,
                                               std::forward<Args>(args)...);
            _result->start();
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            Toolset_t* _result = static_cast<Toolset_t*>(v_result);
            _result->stop();
        };

        auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
            if(_hash == _typeid_hash && v_result && !ptr)
            {
                Toolset_t* _result = static_cast<Toolset_t*>(v_result);
                ptr                = static_cast<void*>(_result->template get<Toolset>());
            }
        };

        auto _del = [=](void* v_result) {
            if(v_result)
            {
                Toolset_t* _result = static_cast<Toolset_t*>(v_result);
                delete _result;
            }
        };

        m_bundle.emplace_back(bundle_data{ _typeid_hash, std::move(_start),
                                           std::move(_stop), std::move(_get),
                                           std::move(_del) });
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a specific set of tools
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) == 0), int>    = 0,
              enable_if_t<!(Toolset::is_component), char> = 0>
    void insert(bool _flat, Args&&... args)
    {
        DEBUG_PRINT_HERE("%s", demangle<Toolset>().c_str());

        if(Toolset::size() == 0)
            return;

        auto _typeid_hash = get_hash(demangle<Toolset>());
        if(m_typeids.count(_typeid_hash) > 0)
            return;

        internal_init();

        auto _start = [=, &args...](const string_t& _prefix, bool _argflat) {
            constexpr bool is_component_type = Toolset::is_component_type;
            Toolset*       _result           = (is_component_type)
                                   ? new Toolset(_prefix, true, _flat || _argflat,
                                                 std::forward<Args>(args)...)
                                   : new Toolset(_prefix, _flat || _argflat,
                                                 settings::destructor_report(),
                                                 std::forward<Args>(args)...);
            return (void*) _result;
        };

        auto _stop = [=](void* v_result) {
            Toolset* _result = static_cast<Toolset*>(v_result);
            _result->stop();
        };

        auto _get = [=](void* v_result, void*& ptr, size_t _hash) {
            if(v_result && !ptr)
            {
                Toolset* _result = static_cast<Toolset*>(v_result);
                _result->get(ptr, _hash);
            }
        };

        auto _del = [=](void* v_result) {
            if(v_result)
            {
                Toolset* _result = static_cast<Toolset*>(v_result);
                delete _result;
            }
        };

        lock_t lk(get_lock());
        variadic_toolset<Toolset>::add_typeids(m_typeids);
        m_bundle.push_back({ _typeid_hash, std::move(_start), std::move(_stop),
                             std::move(_get), std::move(_del) });
    }

    //----------------------------------------------------------------------------------//
    //  Configure the tool for a variadic list of tools
    //
    template <typename Head, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) > 0), int> = 0>
    void insert(bool flat, Args&&... args)
    {
        DEBUG_PRINT_HERE("%s", demangle<Head>().c_str());

        configure<Head>(flat, std::forward<Args>(args)...);
        configure<Tail...>(flat, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) == 0), int>                 = 0,
              enable_if_t<!(trait::is_available<Toolset>::value), int> = 0>
    void insert(bool, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //  If a tool is not avail
    //
    template <typename Toolset, typename... Tail, typename... Args,
              enable_if_t<(sizeof...(Tail) > 0), int>                  = 0,
              enable_if_t<!(trait::is_available<Toolset>::value), int> = 0>
    void insert(bool flat, Args&&... _args)
    {
        insert<Tail...>(flat, std::forward<Args>(_args)...);
    }

    template <typename Toolset, typename... Tail>
    void insert()
    {
        insert<Toolset, Tail...>(settings::flat_profile());
    }

    void set_flat_profile(bool val) { m_flat = val; }

protected:
    bool              m_flat    = false;
    string_t          m_prefix  = "";
    typeid_set_t      m_typeids = get_typeids();
    bundle_data_vec_t m_bundle  = get_bundle_data();

private:
    struct persistent_data
    {
        mutex_t           lock;
        bundle_data_vec_t data    = {};
        typeid_set_t      typeids = {};
    };

    //----------------------------------------------------------------------------------//
    //  Persistent data
    //
    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance{};
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  Bundle data
    //
    static bundle_data_vec_t& get_bundle_data() { return get_persistent_data().data; }

    //----------------------------------------------------------------------------------//
    //  The configuration strings
    //
    static typeid_set_t& get_typeids() { return get_persistent_data().typeids; }

    //----------------------------------------------------------------------------------//
    //  Get lock
    //
    static mutex_t& get_lock() { return get_persistent_data().lock; }

    //----------------------------------------------------------------------------------//
    //  Initialize the storage
    //
    static void internal_init()
    {
        static bool _inited = []() {
            auto ret = storage_type::instance();
            ret->initialize();
            return true;
        }();
        consume_parameters(_inited);
    }

    //
    template <typename Tool, enable_if_t<(Tool::is_component), char> = 0>
    static void internal_init()
    {
        internal_init();
        static bool _inited = []() {
            using tool_storage_type = typename Tool::storage_type;
            auto ret                = tool_storage_type::instance();
            ret->initialize();
            return true;
        }();
        consume_parameters(_inited);
    }
};

//--------------------------------------------------------------------------------------//

// generic user_bundle
template <size_t Idx, typename Tag>
inline void
user_bundle<Idx, Tag>::global_init(storage_type*)
{}

// user_tuple_bundle
template <>
inline void
user_bundle<10101, api::native_tag>::global_init(storage_type*)
{
    /*
    auto env_tool = tim::get_env<std::string>("USER_TUPLE_COMPONENTS", "");
    auto env_enum = tim::enumerate_components(tim::delimit(env_tool));
    tim::configure<user_bundle<10101, api::native_tag>>(env_enum);
    */
}

// user_list_bundle
template <>
inline void
user_bundle<11011, api::native_tag>::global_init(storage_type*)
{
    /*
    auto env_tool = tim::get_env<std::string>("USER_LIST_COMPONENTS", "");
    auto env_enum = tim::enumerate_components(tim::delimit(env_tool));
    tim::configure<user_bundle<11011, api::native_tag>>(env_enum);
    */
}

// user_ompt_bundle
template <>
inline void
user_bundle<10001, api::native_tag>::global_init(storage_type*)
{}

//--------------------------------------------------------------------------------------//

}  // namespace component

//--------------------------------------------------------------------------------------//
}  // namespace tim
