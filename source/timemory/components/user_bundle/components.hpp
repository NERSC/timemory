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
#include "timemory/components/user_bundle/backends.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <functional>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace env
{
//
using user_bundle_spec_t = std::function<std::string()>;
//
using user_bundle_variables_t =
    std::unordered_map<size_t, std::vector<user_bundle_spec_t>>;
//
//--------------------------------------------------------------------------------------//
/// non-static so that projects can globally change this for their project/API
template <typename ApiT>
inline user_bundle_variables_t& get_user_bundle_variables(ApiT)
{
    static user_bundle_variables_t _instance{};
    return _instance;
}
//
#if defined(TIMEMORY_USER_BUNDLE_HEADER_MODE)
/// static so that projects cannot globally change this
static inline user_bundle_variables_t& get_user_bundle_variables(TIMEMORY_API);
static inline user_bundle_variables_t& get_user_bundle_variables(project::kokkosp);
//
inline std::vector<TIMEMORY_COMPONENT>
get_bundle_components(const std::vector<user_bundle_spec_t>& _priority);
#else
/// static so that projects cannot globally change this
user_bundle_variables_t& get_user_bundle_variables(TIMEMORY_API);
user_bundle_variables_t& get_user_bundle_variables(project::kokkosp);
//
std::vector<TIMEMORY_COMPONENT>
get_bundle_components(const std::vector<user_bundle_spec_t>& _priority);
#endif
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Api, typename AltApi = Api>
void
initialize_bundle(AltApi _api = AltApi{})
{
    using user_bundle_type = component::user_bundle<Idx, Api>;
    auto& variables        = env::get_user_bundle_variables(_api);
    auto  itr              = variables.find(Idx);
    if(itr != variables.end())
    {
        auto _enum = env::get_bundle_components(itr->second);
        tim::configure<user_bundle_type>(_enum);
    }
}
//
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
struct user_bundle
: public base<user_bundle<Idx, Tag>, void>
, concepts::runtime_configurable
{
public:
    static constexpr auto index = Idx;
    using tag_type              = Tag;
    using mutex_t               = std::mutex;
    using lock_t                = std::unique_lock<mutex_t>;
    using string_t              = std::string;

    using value_type   = void;
    using this_type    = user_bundle<Idx, Tag>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    using start_func_t  = std::function<void*(const string_t&, scope::config)>;
    using stop_func_t   = std::function<void(void*)>;
    using get_func_t    = std::function<void(void*, void*&, size_t)>;
    using delete_func_t = std::function<void(void*)>;

    static string_t label() { return "user_bundle"; }
    static string_t description()
    {
        return "Generic bundle of components designed for runtime configuration by a "
               "user via environment variables and/or direct insertion";
    }
    static value_type record() {}

    static void global_init() TIMEMORY_VISIBILITY("default");
    static void global_init(storage_type*) { global_init(); }

    using opaque_array_t = std::vector<opaque>;
    using typeid_vec_t   = std::vector<size_t>;
    using typeid_set_t   = std::set<size_t>;

    static size_t bundle_size() { return get_data().size(); }

public:
    //----------------------------------------------------------------------------------//
    //  Captures the statically-defined data so these can be changed without
    //  affecting this instance
    //
    user_bundle()
    : m_scope(scope::get_default())
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

    explicit user_bundle(const char* _prefix, scope::config _scope = scope::get_default())
    : m_scope(_scope)
    , m_prefix(_prefix)
    , m_typeids(get_typeids())
    , m_bundle(get_data())
    {}

    user_bundle(const char* _prefix, opaque_array_t _bundle_vec, typeid_vec_t _typeids,
                scope::config _scope = scope::get_default())
    : m_scope(_scope)
    , m_prefix(_prefix)
    , m_typeids(std::move(_typeids))
    , m_bundle(std::move(_bundle_vec))
    {}

    user_bundle(const char* _prefix, opaque_array_t _bundle_vec,
                const typeid_set_t& _typeids, scope::config _scope = scope::get_default())
    : m_scope(_scope)
    , m_prefix(_prefix)
    , m_bundle(std::move(_bundle_vec))
    {
        m_typeids.reserve(_typeids.size());
        for(const auto& itr : _typeids)
            m_typeids.emplace_back(itr);
    }

    ~user_bundle()
    {
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

    user_bundle(user_bundle&& rhs) noexcept
    : base_type(std::move(rhs))
    , m_scope(std::move(rhs.m_scope))
    , m_prefix(std::move(rhs.m_prefix))
    , m_typeids(std::move(rhs.m_typeids))
    , m_bundle(std::move(rhs.m_bundle))
    {
        rhs.m_bundle.clear();
    }

    user_bundle& operator=(user_bundle&& rhs) noexcept
    {
        if(this != &rhs)
        {
            base_type::operator=(std::move(rhs));
            m_scope            = std::move(rhs.m_scope);
            m_prefix           = std::move(rhs.m_prefix);
            m_typeids          = std::move(rhs.m_typeids);
            m_bundle           = std::move(rhs.m_bundle);
            rhs.m_bundle.clear();
        }
        return *this;
    }

public:
    //  Configure the tool for a specific component
    static void configure(opaque&& obj, std::set<size_t>&& _typeids)
    {
        if(obj)
        {
            lock_t lk(get_lock());
            size_t sum = 0;
            for(auto&& itr : _typeids)
            {
                if(itr > 0 && contains(itr, get_typeids()))
                {
                    if(settings::verbose() > 1 || settings::debug())
                        PRINT_HERE("Skipping duplicate typeid: %lu", (unsigned long) itr);
                    return;
                }
                sum += itr;
                if(itr > 0)
                    get_typeids().emplace_back(itr);
            }
            if(sum == 0)
            {
                PRINT_HERE("No typeids. Sum: %lu", (unsigned long) sum);
                return;
            }
            get_data().emplace_back(std::move(obj));
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
        if(settings::verbose() > 3 || settings::debug())
            PRINT_HERE("Resetting %s", demangle<this_type>().c_str());
        lock_t lk(get_lock());
        get_data().clear();
        get_typeids().clear();
    }

public:
    //----------------------------------------------------------------------------------//
    //  Member functions
    //
    void setup()
    {
        if(!m_setup)
        {
            m_setup = true;
            for(auto& itr : m_bundle)
                itr.setup(m_prefix, m_scope);
        }
    }

    void push()
    {
        setup();
        for(auto& itr : m_bundle)
            itr.push(m_prefix, m_scope);
    }

    void sample()
    {
        setup();
        for(auto& itr : m_bundle)
            itr.sample();
    }

    void start()
    {
        setup();
        for(auto& itr : m_bundle)
            itr.start();
    }

    void stop()
    {
        for(auto& itr : m_bundle)
            itr.stop();
    }

    void pop()
    {
        for(auto& itr : m_bundle)
            itr.pop();
    }

    void clear()
    {
        if(base_type::get_is_running())
            stop();
        m_typeids.clear();
        m_bundle.clear();
        m_setup = false;
    }

    template <typename T>
    T* get()
    {
        auto  _typeid_hash = typeid_hash<T>();
        void* void_ptr     = nullptr;
        for(auto& itr : m_bundle)
        {
            itr.get(void_ptr, _typeid_hash);
            if(void_ptr)
                return static_cast<T*>(void_ptr);
        }
        return static_cast<T*>(void_ptr);
    }

    void get(void*& ptr, size_t _hash) const
    {
        if(ptr == nullptr)
        {
            for(const auto& itr : m_bundle)
            {
                itr.get(ptr, _hash);
                if(ptr)
                    break;
            }
        }
    }

    void get() {}

    void set_prefix(const char* _prefix)
    {
        m_prefix = _prefix;
        m_setup  = false;
    }

    void set_scope(const scope::config& val)
    {
        m_scope = val;
        m_setup = false;
    }

    TIMEMORY_NODISCARD size_t size() const { return m_bundle.size(); }

public:
    //  Configure the tool for a specific component
    void insert(opaque&& obj, typeid_set_t&& _typeids)
    {
        if(obj)
        {
            size_t sum = 0;
            for(auto&& itr : _typeids)
            {
                if(itr > 0 && contains(itr, m_typeids))
                {
                    if(settings::verbose() > 1 || settings::debug())
                        PRINT_HERE("Skipping duplicate typeid: %lu", (unsigned long) itr);
                    return;
                }
                sum += itr;
                if(itr > 0)
                    m_typeids.emplace_back(itr);
            }
            if(sum == 0)
            {
                PRINT_HERE("No typeids. Sum: %lu", (unsigned long) sum);
                return;
            }
            m_bundle.emplace_back(std::move(obj));
        }
    }

    template <typename Type, typename... Types, typename... Args>
    void insert(Args... args)
    {
        this->insert(factory::get_opaque<Type>(args...), factory::get_typeids<Type>());
        TIMEMORY_FOLD_EXPRESSION(this->insert(factory::get_opaque<Types>(args...),
                                              factory::get_typeids<Types>()));
    }

protected:
    bool           m_setup   = false;
    scope::config  m_scope   = {};
    const char*    m_prefix  = nullptr;
    typeid_vec_t   m_typeids = get_typeids();
    opaque_array_t m_bundle  = get_data();

protected:
    static bool contains(size_t _val, const typeid_vec_t& _targ)
    {
        return std::any_of(_targ.begin(), _targ.end(),
                           [&_val](auto itr) { return (itr == _val); });
    }

private:
    struct persistent_data
    {
        mutex_t        m_lock;
        opaque_array_t m_data    = {};
        typeid_vec_t   m_typeids = {};
    };

    //----------------------------------------------------------------------------------//
    //  Persistent data
    //
    static persistent_data& get_persistent_data() TIMEMORY_VISIBILITY("default");

public:
    //----------------------------------------------------------------------------------//
    //  Bundle data
    //
    static opaque_array_t& get_data() { return get_persistent_data().m_data; }

    //----------------------------------------------------------------------------------//
    //  The configuration strings
    //
    static typeid_vec_t& get_typeids() { return get_persistent_data().m_typeids; }

    //----------------------------------------------------------------------------------//
    //  Get lock
    //
    static mutex_t& get_lock() { return get_persistent_data().m_lock; }
};
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Tag>
void
user_bundle<Idx, Tag>::global_init()
{
    if(settings::verbose() > 2 || settings::debug())
        PRINT_HERE("Global initialization of %s", demangle<this_type>().c_str());
    env::initialize_bundle<Idx, Tag>();
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename Tag>
typename user_bundle<Idx, Tag>::persistent_data&
user_bundle<Idx, Tag>::get_persistent_data()
{
    static persistent_data _instance{};
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//

#if defined(TIMEMORY_USER_BUNDLE_HEADER_MODE)
#    include "timemory/components/user_bundle/components.cpp"
#endif
