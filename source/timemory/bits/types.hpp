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

/** \file bits/types.hpp
 * \headerfile bits/types.hpp "timemory/bits/types.hpp"
 * Provides some additional info for timemory/components/types.hpp
 *
 */

#pragma once

#include "timemory/bits/settings.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
struct properties
{
    static constexpr TIMEMORY_COMPONENT value = TIMEMORY_COMPONENTS_END;
    static bool&                        has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace component

//======================================================================================//
// generate a master instance and a nullptr on the first pass
// generate a worker instance on subsequent and return master and worker
//
template <typename _Tp, typename _Ptr = std::shared_ptr<_Tp>,
          typename _Pair = std::pair<_Ptr, _Ptr>>
_Pair&
get_shared_ptr_pair()
{
    static auto              _master = std::make_shared<_Tp>();
    static std::atomic<int>  _counter;
    static thread_local auto _worker   = _Ptr((_counter++ == 0) ? nullptr : new _Tp());
    static thread_local auto _instance = _Pair{ _master, _worker };
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Ptr = std::shared_ptr<_Tp>,
          typename _Pair = std::pair<_Ptr, _Ptr>>
_Ptr
get_shared_ptr_pair_instance()
{
    static thread_local auto& _pinst = get_shared_ptr_pair<_Tp>();
    static thread_local auto& _inst  = _pinst.second.get() ? _pinst.second : _pinst.first;
    return _inst;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Ptr = std::shared_ptr<_Tp>,
          typename _Pair = std::pair<_Ptr, _Ptr>>
_Ptr
get_shared_ptr_pair_master_instance()
{
    static auto& _pinst = get_shared_ptr_pair<_Tp>();
    static auto  _inst  = _pinst.first;
    return _inst;
}

//--------------------------------------------------------------------------------------//
//
//  hash storage and source_location
//
//--------------------------------------------------------------------------------------//

using hash_result_type          = std::hash<std::string>::result_type;
using graph_hash_map_t          = std::unordered_map<hash_result_type, std::string>;
using graph_hash_alias_t        = std::unordered_map<hash_result_type, hash_result_type>;
using graph_hash_map_ptr_t      = std::shared_ptr<graph_hash_map_t>;
using graph_hash_map_ptr_pair_t = std::pair<graph_hash_map_ptr_t, graph_hash_map_ptr_t>;
using graph_hash_alias_ptr_t    = std::shared_ptr<graph_hash_alias_t>;

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_EXTERN_INIT)

extern graph_hash_map_ptr_t
get_hash_ids();

extern graph_hash_alias_ptr_t
get_hash_aliases();

#else

//--------------------------------------------------------------------------------------//

inline graph_hash_map_ptr_t
get_hash_ids()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_map_t>();
    return _inst;
}

//--------------------------------------------------------------------------------------//

inline graph_hash_alias_ptr_t
get_hash_aliases()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_alias_t>();
    return _inst;
}

#endif

//--------------------------------------------------------------------------------------//

inline hash_result_type
add_hash_id(graph_hash_map_ptr_t& _hash_map, const std::string& prefix)
{
    hash_result_type _hash_id = std::hash<std::string>()(prefix.c_str());
    if(_hash_map && _hash_map->find(_hash_id) == _hash_map->end())
    {
        if(settings::debug())
            printf("[%s@'%s':%i]> adding hash id: %s = %llu...\n", __FUNCTION__, __FILE__,
                   __LINE__, prefix.c_str(), (long long unsigned) _hash_id);

        (*_hash_map)[_hash_id] = prefix;
        if(_hash_map->bucket_count() < _hash_map->size())
            _hash_map->rehash(_hash_map->size() + 10);
    }
    return _hash_id;
}

//--------------------------------------------------------------------------------------//

inline hash_result_type
add_hash_id(const std::string& prefix)
{
    static thread_local auto _hash_map = get_hash_ids();
    return add_hash_id(_hash_map, prefix);
}

//--------------------------------------------------------------------------------------//

inline void
add_hash_id(graph_hash_map_ptr_t _hash_map, graph_hash_alias_ptr_t _hash_alias,
            hash_result_type _hash_id, hash_result_type _alias_hash_id)
{
    if(_hash_alias->find(_alias_hash_id) == _hash_alias->end() &&
       _hash_map->find(_hash_id) != _hash_map->end())
    {
        (*_hash_alias)[_alias_hash_id] = _hash_id;
        if(_hash_alias->bucket_count() < _hash_alias->size())
            _hash_alias->rehash(_hash_alias->size() + 10);
    }
}

//--------------------------------------------------------------------------------------//

inline void
add_hash_id(hash_result_type _hash_id, hash_result_type _alias_hash_id)
{
    add_hash_id(get_hash_ids(), get_hash_aliases(), _hash_id, _alias_hash_id);
}

//--------------------------------------------------------------------------------------//

inline std::string
get_hash_identifier(graph_hash_map_ptr_t _hash_map, graph_hash_alias_ptr_t _hash_alias,
                    hash_result_type _hash_id)
{
    auto _map_itr   = _hash_map->find(_hash_id);
    auto _alias_itr = _hash_alias->find(_hash_id);

    if(_map_itr != _hash_map->end())
        return _map_itr->second;
    else if(_alias_itr != _hash_alias->end())
    {
        _map_itr = _hash_map->find(_alias_itr->second);
        if(_map_itr != _hash_map->end())
            return _map_itr->second;
    }

    if(settings::verbose() > 0 || settings::debug())
    {
        std::stringstream ss;
        ss << "Error! node with hash " << _hash_id
           << " did not have an associated prefix!\n";
        ss << "Hash map:\n";
        auto _w = 30;
        for(const auto& itr : *_hash_map)
            ss << "    " << std::setw(_w) << itr.first << " : " << (itr.second) << "\n";
        if(_hash_alias->size() > 0)
        {
            ss << "Alias hash map:\n";
            for(const auto& itr : *_hash_alias)
                ss << "    " << std::setw(_w) << itr.first << " : " << itr.second << "\n";
        }
        fprintf(stderr, "%s\n", ss.str().c_str());
    }
    return std::string("unknown-hash=") + std::to_string(_hash_id);
}

//--------------------------------------------------------------------------------------//

inline std::string
get_hash_identifier(hash_result_type _hash_id)
{
    return get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id);
}

//======================================================================================//

class source_location
{
public:
    using join_type   = apply<std::string>;
    using result_type = std::tuple<std::string, size_t>;

public:
    //
    //  Public types
    //

    //==================================================================================//
    //
    enum class mode : short
    {
        blank = 0,
        basic = 1,
        full  = 2
    };

    //==================================================================================//
    //
    struct captured
    {
    public:
        const std::string&      get_id() const { return std::get<0>(m_result); }
        const hash_result_type& get_hash() const { return std::get<1>(m_result); }
        const result_type&      get() const { return m_result; }

        explicit captured(const result_type& _result)
        : m_result(_result)
        {}

        captured()  = default;
        ~captured() = default;

        captured(const captured&) = default;
        captured(captured&&)      = default;

        captured& operator=(const captured&) = default;
        captured& operator=(captured&&) = default;

    protected:
        friend class source_location;
        result_type m_result = result_type("", 0);

        template <typename... _Args>
        captured& set(const source_location& obj, _Args&&... _args)
        {
            switch(obj.m_mode)
            {
                case mode::blank:
                {
                    auto&& _tmp = join_type::join("", std::forward<_Args>(_args)...);
                    m_result    = result_type(_tmp, add_hash_id(_tmp));
                    break;
                }
                case mode::basic:
                case mode::full:
                {
                    auto&& _suffix = join_type::join("", std::forward<_Args>(_args)...);
                    auto   _tmp    = join_type::join("/", obj.m_prefix.c_str(), _suffix);
                    m_result       = result_type(_tmp, add_hash_id(_tmp));
                    break;
                }
            }
            return *this;
        }
    };

public:
    //
    //  Static functions
    //

    //==================================================================================//
    //
    template <typename... _Args>
    static captured get_captured_inline(const mode& _mode, const char* _func, int _line,
                                        const char* _fname, _Args&&... _args)
    {
        source_location _loc(_mode, _func, _line, _fname, std::forward<_Args>(_args)...);
        return _loc.get_captured(std::forward<_Args>(_args)...);
    }

public:
    //
    //  Constructors, destructors, etc.
    //

    //----------------------------------------------------------------------------------//
    //
    template <typename... _Args>
    source_location(const mode& _mode, const char* _func, int _line, const char* _fname,
                    _Args&&...)
    : m_mode(_mode)
    {
        switch(m_mode)
        {
            case mode::blank: break;
            case mode::basic: compute_data(_func); break;
            case mode::full: compute_data(_func, _line, _fname); break;
        }
    }

    //----------------------------------------------------------------------------------//
    //  if a const char* is passed, assume it is constant
    //
    source_location(const mode& _mode, const char* _func, int _line, const char* _fname,
                    const char* _arg)
    : m_mode(_mode)
    {
        switch(m_mode)
        {
            case mode::blank:
            {
                // label and hash
                auto&& _label = std::string(_arg);
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
            case mode::basic:
            {
                compute_data(_func);
                auto&& _label = (_arg) ? _join(_arg) : std::string(m_prefix);
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
            case mode::full:
            {
                compute_data(_func, _line, _fname);
                // label and hash
                auto&& _label = (_arg) ? _join(_arg) : std::string(m_prefix);
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    source_location()                       = delete;
    ~source_location()                      = default;
    source_location(const source_location&) = delete;
    source_location(source_location&&)      = default;
    source_location& operator=(const source_location&) = delete;
    source_location& operator=(source_location&&) = default;

protected:
    //----------------------------------------------------------------------------------//
    //
    void compute_data(const char* _func) { m_prefix = _func; }

    //----------------------------------------------------------------------------------//
    //
    void compute_data(const char* _func, int _line, const char* _fname)
    {
#if defined(_WINDOWS)
        static const char delim = '\\';
#else
        static const char delim = '/';
#endif
        std::string _filename(_fname);
        if(_filename.find(delim) != std::string::npos)
            _filename = _filename.substr(_filename.find_last_of(delim) + 1).c_str();
        m_prefix = join_type::join("", _func, "@", _filename, ":", _line);
    }

public:
    //----------------------------------------------------------------------------------//
    //
    template <typename... _Args>
    const captured& get_captured(_Args&&... _args)
    {
        return (settings::enabled())
                   ? m_captured.set(*this, std::forward<_Args>(_args)...)
                   : m_captured;
    }

    //----------------------------------------------------------------------------------//
    //
    const captured& get_captured(const char*) { return m_captured; }

private:
    mode        m_mode;
    std::string m_prefix = "";
    captured    m_captured;

private:
    std::string _join(const char* _arg)
    {
        return (strcmp(_arg, "") == 0) ? m_prefix
                                       : join_type::join("/", m_prefix.c_str(), _arg);
    }
};

}  // namespace tim

#if !defined(TIMEMORY_SOURCE_LOCATION)

#    define _AUTO_LOCATION_COMBINE(X, Y) X##Y
#    define _AUTO_LOCATION(Y) _AUTO_LOCATION_COMBINE(timemory_source_location_, Y)

#    define TIMEMORY_SOURCE_LOCATION(MODE, ...)                                          \
        ::tim::source_location(MODE, __FUNCTION__, __LINE__, __FILE__, __VA_ARGS__)

#    define TIMEMORY_CAPTURE_MODE(MODE_TYPE) ::tim::source_location::mode::MODE_TYPE

#    define TIMEMORY_CAPTURE_ARGS(...) _AUTO_LOCATION(__LINE__).get_captured(__VA_ARGS__)

#    define TIMEMORY_INLINE_SOURCE_LOCATION(MODE, ...)                                   \
        ::tim::source_location::get_captured_inline(                                     \
            TIMEMORY_CAPTURE_MODE(MODE), __FUNCTION__, __LINE__, __FILE__, __VA_ARGS__)

#    define _TIM_STATIC_SRC_LOCATION(MODE, ...)                                          \
        static thread_local auto _AUTO_LOCATION(__LINE__) =                              \
            TIMEMORY_SOURCE_LOCATION(TIMEMORY_CAPTURE_MODE(MODE), __VA_ARGS__)

#endif
