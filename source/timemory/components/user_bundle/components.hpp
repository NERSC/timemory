//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/user_bundle/components.hpp
 * \brief Implementation of the user_bundle component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/utility.hpp"

#include "timemory/components/user_bundle/backends.hpp"
#include "timemory/components/user_bundle/types.hpp"

#include "timemory/runtime/types.hpp"

#include <regex>

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace env
{
using user_bundle_variables_t =
    std::unordered_map<size_t, std::pair<std::string, std::vector<std::string>>>;
//
static inline user_bundle_variables_t&
get_user_bundle_variables()
{
    static user_bundle_variables_t _instance = {
        { component::global_bundle_idx, { "TIMEMORY_GLOBAL_COMPONENTS", {} } },
        { component::tuple_bundle_idx,
          { "TIMEMORY_TUPLE_COMPONENTS", { "TIMEMORY_GLOBAL_COMPONENTS" } } },
        { component::list_bundle_idx, { "TIMEMORY_LIST_COMPONENTS", {} } },
        { component::ompt_bundle_idx,
          { "TIMEMORY_OMPT_COMPONENTS",
            { "TIMEMORY_PROFILER_COMPONENTS", "TIMEMORY_GLOBAL_COMPONENTS",
              "TIMEMORY_COMPONENT_LIST_INIT" } } },
        { component::mpip_bundle_idx,
          { "TIMEMORY_MPIP_COMPONENTS",
            { "TIMEMORY_PROFILER_COMPONENTS", "TIMEMORY_GLOBAL_COMPONENTS",
              "TIMEMORY_COMPONENT_LIST_INIT" } } },
        { component::trace_bundle_idx,
          { "TIMEMORY_TRACE_COMPONENTS", { "TIMEMORY_GLOBAL_COMPONENTS" } } },
        { component::profiler_bundle_idx,
          { "TIMEMORY_PROFILER_COMPONENTS", { "TIMEMORY_GLOBAL_COMPONENTS" } } }
    };
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename VecT>
auto
get_bundle_components(const std::string& custom_env, const VecT& fallback_env)
{
    using string_t = std::string;
    const auto regex_constants =
        std::regex_constants::ECMAScript | std::regex_constants::icase;
    auto env_tool = get_env<string_t>(custom_env, "");
    if(env_tool.empty() &&
       !std::regex_match(env_tool, std::regex("none", regex_constants)))
    {
        for(const auto& itr : fallback_env)
        {
            env_tool = get_env<string_t>(itr);
            if(env_tool.length() > 0 ||
               std::regex_match(env_tool, std::regex("none", regex_constants)))
                break;
        }
    }
    auto env_enum = tim::enumerate_components(tim::delimit(env_tool));
    return env_enum;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Api,
          enable_if_t<(std::is_same<Api, api::native_tag>::value), int> = 0>
void
initialize_bundle()
{
    using user_bundle_type = component::user_bundle<Idx, Api>;
    auto itr               = env::get_user_bundle_variables().find(Idx);
    if(itr != env::get_user_bundle_variables().end())
    {
        auto env_enum = env::get_bundle_components(itr->second.first, itr->second.second);
        tim::configure<user_bundle_type>(env_enum);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Api,
          enable_if_t<!(std::is_same<Api, api::native_tag>::value), int> = 0>
void
initialize_bundle()
{}
}  // namespace env
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                                  USER BUNDLE
//
//--------------------------------------------------------------------------------------//
//
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

    using start_func_t  = std::function<void*(const string_t&, scope::config)>;
    using stop_func_t   = std::function<void(void*)>;
    using get_func_t    = std::function<void(void*, void*&, size_t)>;
    using delete_func_t = std::function<void(void*)>;

    static string_t   label() { return "user_bundle"; }
    static string_t   description() { return "user-defined bundle of tools"; }
    static value_type record() {}

    static void global_init(storage_type*);

    using opaque_array_t = std::vector<opaque>;
    using typeid_set_t   = std::set<size_t>;

    static size_t bundle_size() { return get_data().size(); }

public:
    //----------------------------------------------------------------------------------//
    //  Captures the statically-defined data so these can be changed without
    //  affecting this instance
    //
    user_bundle()
    : m_scope(scope::get_default())
    , m_prefix("")
    , m_typeids(get_typeids())
    , m_bundle(get_data())
    {}

    explicit user_bundle(const string_t& _prefix,
                         scope::config   _scope = scope::get_default())
    : m_scope(_scope)
    , m_prefix(_prefix)
    , m_typeids(get_typeids())
    , m_bundle(get_data())
    {}

    user_bundle(const user_bundle& rhs)
    : base_type(rhs)
    , m_scope(rhs.m_scope)
    , m_prefix(rhs.m_prefix)
    , m_typeids(rhs.m_typeids)
    , m_bundle(rhs.m_bundle)
    {
        for(auto& itr : m_bundle)
            itr.set_copy(true);
    }

    user_bundle(const string_t& _prefix, const opaque_array_t& _bundle_vec,
                const typeid_set_t& _typeids, scope::config _scope = scope::get_default())
    : m_scope(_scope)
    , m_prefix(_prefix)
    , m_typeids(_typeids)
    , m_bundle(_bundle_vec)
    {}

    ~user_bundle()
    {
        // gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());
        for(auto& itr : m_bundle)
            itr.cleanup();
    }

    user_bundle& operator=(const user_bundle& rhs)
    {
        if(this == &rhs)
            return *this;

        base_type::operator=(rhs);
        m_scope            = rhs.m_scope;
        m_prefix           = rhs.m_prefix;
        m_typeids          = rhs.m_typeids;
        m_bundle           = rhs.m_bundle;
        for(auto& itr : m_bundle)
            itr.set_copy(true);

        return *this;
    }

    user_bundle(user_bundle&&) = default;
    user_bundle& operator=(user_bundle&&) = default;

public:
    //  Configure the tool for a specific component
    static void configure(opaque&& obj, typeid_set_t&& _typeids)
    {
        if(obj)
        {
            lock_t lk(get_lock());
            size_t sum = 0;
            for(auto&& itr : _typeids)
            {
                if(itr > 0 && get_typeids().count(itr) > 0)
                {
                    if(settings::verbose() > 1)
                        PRINT_HERE("Skipping duplicate typeid: %lu", (unsigned long) itr);
                    return;
                }
                sum += itr;
                if(itr > 0)
                    get_typeids().insert(std::move(itr));
            }
            if(sum == 0)
            {
                PRINT_HERE("No typeids. Sum: %lu", (unsigned long) sum);
                return;
            }

            obj.init();
            get_data().emplace_back(std::forward<opaque>(obj));
        }
    }

    template <typename Type, typename... Types, typename... Args>
    static void configure(Args&&... args)
    {
        this_type::configure(factory::get_opaque<Type>(std::forward<Args>(args)...),
                             factory::get_typeids<Type>());

        TIMEMORY_FOLD_EXPRESSION(
            this_type::configure(factory::get_opaque<Types>(std::forward<Args>(args)...),
                                 factory::get_typeids<Types>()));
    }

    //----------------------------------------------------------------------------------//
    //  Explicitly clear the previous configurations
    //
    static void reset()
    {
        lock_t lk(get_lock());
        get_data().clear();
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
            itr.start(m_prefix, m_scope);
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

    void get() {}

    void set_prefix(const string_t& _prefix)
    {
        // skip unnecessary copies
        if(!m_bundle.empty())
            m_prefix = _prefix;
    }

    void set_scope(const scope::config& val)
    {
        // skip unnecessary copies
        if(!m_bundle.empty())
            m_scope = val;
    }

    size_t size() const { return m_bundle.size(); }

public:
    //  Configure the tool for a specific component
    void insert(opaque&& obj, typeid_set_t&& _typeids)
    {
        if(obj)
        {
            size_t sum = 0;
            for(auto&& itr : _typeids)
            {
                if(itr > 0 && m_typeids.count(itr) > 0)
                    return;
                sum += itr;
                m_typeids.insert(std::move(itr));
            }
            if(sum == 0)
                return;

            obj.init();
            m_bundle.emplace_back(std::forward<opaque>(obj));
        }
    }

    template <typename Type, typename... Types, typename... Args>
    void insert(Args&&... args)
    {
        this->insert(factory::get_opaque<Type>(std::forward<Args>(args)...),
                     factory::get_typeids<Type>());

        TIMEMORY_FOLD_EXPRESSION(
            this->insert(factory::get_opaque<Types>(std::forward<Args>(args)...),
                         factory::get_typeids<Types>()));
    }

protected:
    scope::config  m_scope   = scope::get_default();
    string_t       m_prefix  = "";
    typeid_set_t   m_typeids = get_typeids();
    opaque_array_t m_bundle  = get_data();

private:
    struct persistent_data
    {
        mutex_t        lock;
        opaque_array_t data    = {};
        typeid_set_t   typeids = {};
    };

    //----------------------------------------------------------------------------------//
    //  Persistent data
    //
    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance{};
        return _instance;
    }

public:
    //----------------------------------------------------------------------------------//
    //  Bundle data
    //
    static opaque_array_t& get_data() { return get_persistent_data().data; }

    //----------------------------------------------------------------------------------//
    //  The configuration strings
    //
    static typeid_set_t& get_typeids() { return get_persistent_data().typeids; }

    //----------------------------------------------------------------------------------//
    //  Get lock
    //
    static mutex_t& get_lock() { return get_persistent_data().lock; }
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Tag>
void
user_bundle<Idx, Tag>::global_init(storage_type*)
{
    env::initialize_bundle<Idx, Tag>();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
