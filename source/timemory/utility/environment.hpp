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

/** \file utility/environment.hpp
 * \headerfile utility/environment.hpp "timemory/utility/environment.hpp"
 * Functions for processing the environment
 *
 */

#pragma once

#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/utility.hpp"

//--------------------------------------------------------------------------------------//

namespace tim
{
//======================================================================================//
//
//  Environment
//
//======================================================================================//

class env_settings
{
public:
    using mutex_t        = std::recursive_mutex;
    using string_t       = std::string;
    using env_map_t      = std::map<string_t, string_t>;
    using env_uomap_t    = std::map<string_t, string_t>;
    using env_pair_t     = std::pair<string_t, string_t>;
    using iterator       = typename env_map_t::iterator;
    using const_iterator = typename env_map_t::const_iterator;

public:
    static env_settings* instance();

private:
    // if passed a global, insert m_env into it's array
    env_settings(env_settings* _global = nullptr, int _id = 0)
    : m_id(_id)
    , m_env(new env_uomap_t)
    {
        if(_global && _id != 0)
        {
            auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
            lock_flag().store(true);
            if(!lk.owns_lock())
                lk.lock();
            _global->m_env_other.emplace_back(m_env);
            lock_flag().store(false);
        }
    }

    // only the zero id deletes
    ~env_settings()
    {
        if(m_id == 0)
        {
            delete m_env;
            for(auto& itr : m_env_other)
                delete itr;
        }
    }

public:
    template <typename _Tp>
    void insert(const std::string& env_id, _Tp val)
    {
#if !defined(TIMEMORY_DISABLE_STORE_ENVIRONMENT)
        std::stringstream ss;
        ss << std::boolalpha << val;

        auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
        if(lock_flag().load() && !lk.owns_lock())
            lk.lock();

        if(m_env->find(env_id) == m_env->end() || m_env->find(env_id)->second != ss.str())
            (*m_env)[env_id] = ss.str();
#endif
    }

    env_map_t get() const
    {
        auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
        lock_flag().store(true);
        if(!lk.owns_lock())
            lk.lock();

        auto      _tmp = *m_env;
        env_map_t _ret;
        for(const auto& itr : _tmp)
            _ret[itr.first] = itr.second;

        lock_flag().store(false);
        return _ret;
    }

    iterator get(const string_t& _entry) { return m_env->find(_entry); }

    const_iterator get(const string_t& _entry) const { return m_env->find(_entry); }

    iterator       begin() { return m_env->begin(); }
    iterator       end() { return m_env->end(); }
    const_iterator begin() const { return m_env->begin(); }
    const_iterator end() const { return m_env->end(); }

    friend std::ostream& operator<<(std::ostream& os, const env_settings& env)
    {
        if(env.m_id == 0)
            const_cast<env_settings&>(env).collapse();

        auto _data = env.get();

        std::stringstream filler;
        filler.fill('-');
        filler << '#';
        filler << std::setw(88) << "";
        filler << '#';

        std::stringstream ss;
        ss << filler.str() << '\n';
        ss << "# Environment settings:\n";
        for(const auto& itr : _data)
        {
            ss << "# " << std::setw(35) << std::right << itr.first << "\t = \t"
               << std::left << itr.second << '\n';
        }
        ss << filler.str();
        os << ss.str() << '\n';
        return os;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        if(!m_env)
            m_env = new env_uomap_t();

        collapse();

        auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
        lock_flag().store(true);

        if(!lk.owns_lock())
            lk.lock();

        ar(cereal::make_nvp("environment", *m_env));

        lock_flag().store(false);
    }

private:
    void collapse()
    {
        if(m_id != 0)
            return;

        auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();

        lock_flag().store(true);

        // insert all the other maps into this map
        for(size_t i = 0; i < m_env_other.size(); ++i)
        {
            // get the map instance
            auto itr = m_env_other[i];
            if(itr)
            {
                // loop over entries
                for(const auto& mitr : *m_env_other[i])
                {
                    auto key = mitr.first;
                    if(m_env->find(key) != m_env->end())
                    {
                        // if the key already exists and if the entries differ,
                        // create a new unique entry for the index
                        if(m_env->find(key)->second != mitr.second)
                        {
                            key += string_t("[@") + std::to_string(i + 1) + string_t("]");
                            (*m_env)[key] = mitr.second;
                        }
                    }
                    else
                    {
                        // if a new entry, insert without unique tag to avoid duplication
                        // when thread count is very high
                        (*m_env)[key] = mitr.second;
                    }
                }
            }
        }
        lock_flag().store(false);
    }

private:
    static mutex_t& mutex()
    {
        static mutex_t m_mutex;
        return m_mutex;
    }

    static std::atomic_bool& lock_flag()
    {
        static std::atomic_bool _instance(false);
        return _instance;
    }

private:
    int                       m_id  = 0;
    env_uomap_t*              m_env = new env_uomap_t();
    std::vector<env_uomap_t*> m_env_other;
};

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_EXTERN_INIT)
inline env_settings*
env_settings::instance()
{
    static std::atomic<int>           _count;
    static env_settings*              _instance = new env_settings();
    static thread_local int           _id       = _count++;
    static thread_local env_settings* _local =
        (_id == 0) ? _instance : new env_settings(_instance, _id);
    return _local;
}
#endif

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp
get_env(const std::string& env_id, _Tp _default = _Tp())
{
    if(env_id.empty())
        return _default;

    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp                var = _Tp();
        iss >> var;
        env_settings::instance()->insert<_Tp>(env_id, var);
        return var;
    }
    // record default value
    env_settings::instance()->insert<_Tp>(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
// specialization for string since the above will have issues if string
// includes spaces
template <>
inline std::string
get_env(const std::string& env_id, std::string _default)
{
    if(env_id.empty())
        return _default;

    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::stringstream ss;
        ss << env_var;
        env_settings::instance()->insert(env_id, ss.str());
        return ss.str();
    }
    // record default value
    env_settings::instance()->insert(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  overload for boolean
//
template <>
inline bool
get_env(const std::string& env_id, bool _default)
{
    if(env_id.empty())
        return _default;

    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string var = std::string(env_var);
        bool        val = true;
        if(var.find_first_not_of("0123456789") == std::string::npos)
            val = (bool) atoi(var.c_str());
        else
        {
            for(auto& itr : var)
                itr = tolower(itr);
            if(var == "off" || var == "false")
                val = false;
        }
        env_settings::instance()->insert<bool>(env_id, val);
        return val;
    }
    // record default value
    env_settings::instance()->insert<bool>(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp
load_env(const std::string& env_id, _Tp _default = _Tp())
{
    if(env_id.empty())
        return _default;

    auto _env_settings = env_settings::instance();
    auto itr           = _env_settings->get(env_id);
    if(itr != _env_settings->end())
    {
        std::istringstream iss(itr->second);
        _Tp                var = _Tp();
        iss >> var;
        return var;
    }

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
// specialization for string since the above will have issues if string
// includes spaces
template <>
inline std::string
load_env(const std::string& env_id, std::string _default)
{
    if(env_id.empty())
        return _default;

    auto _env_settings = env_settings::instance();
    auto itr           = _env_settings->get(env_id);
    if(itr != _env_settings->end())
        return itr->second;

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  overload for boolean
//
template <>
inline bool
load_env(const std::string& env_id, bool _default)
{
    if(env_id.empty())
        return _default;

    auto _env_settings = env_settings::instance();
    auto itr           = _env_settings->get(env_id);
    if(itr != _env_settings->end())
        return (itr->second == "true") ? true : false;

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//

inline void
print_env(std::ostream& os = std::cout)
{
    os << (*env_settings::instance());
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
