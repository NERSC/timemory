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

/** \file utility.hpp
 * \headerfile utility.hpp "timemory/utility.hpp"
 * General utility functions
 *
 */

#pragma once

// C library
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdlib.h>
// I/O
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
// general
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
// container
#include <deque>
#include <map>
#include <set>
#include <tuple>
#include <vector>
// threading
#include <atomic>
#include <mutex>
#include <thread>

#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/singleton.hpp"

#if defined(_UNIX)
#    include <errno.h>
#    include <stdio.h>
#    include <string.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#elif defined(_WINDOWS)
#    include <direct.h>
#endif

#if !defined(DEFAULT_UMASK)
#    define DEFAULT_UMASK 0777
#endif

//--------------------------------------------------------------------------------------//

// stringify some macro -- uses TIMEMORY_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#    define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#    define TIMEMORY_STRINGIZE2(X) #    X
#endif

// stringify the __LINE__ macro
#if !defined(TIMEMORY_LINE_STRING)
#    define TIMEMORY_LINE_STRING TIMEMORY_STRINGIZE(__LINE__)
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline bool
isfinite(const _Tp& arg)
{
#if defined(_WINDOWS)
    // Windows seems to be missing std::isfinite
    return (arg == arg && arg != std::numeric_limits<_Tp>::infinity() &&
            arg != -std::numeric_limits<_Tp>::infinity())
               ? true
               : false;
#else
    return std::isfinite(arg);
#endif
}

//--------------------------------------------------------------------------------------//

typedef std::string               string_t;
typedef std::deque<string_t>      str_list_t;
typedef std::mutex                mutex_t;
typedef std::unique_lock<mutex_t> auto_lock_t;

//======================================================================================//
//
//  General functions
//
//======================================================================================//

template <typename _Tp>
mutex_t&
type_mutex(const uintmax_t& _n = 0)
{
    static mutex_t* _mutex = new mutex_t();
    if(_n == 0)
        return *_mutex;

    static std::vector<mutex_t*> _mutexes;
    if(_n > _mutexes.size())
        _mutexes.resize(_n, nullptr);
    if(!_mutexes[_n])
        _mutexes[_n] = new mutex_t();
    return *(_mutexes[_n - 1]);
}

//--------------------------------------------------------------------------------------//

inline std::string
dirname(std::string _fname)
{
#if defined(_UNIX)
    char* _cfname = realpath(_fname.c_str(), NULL);
    _fname        = std::string(_cfname);
    free(_cfname);

    while(_fname.find("\\\\") != std::string::npos)
        _fname.replace(_fname.find("\\\\"), 2, "/");
    while(_fname.find("\\") != std::string::npos)
        _fname.replace(_fname.find("\\"), 1, "/");

    return _fname.substr(0, _fname.find_last_of("/"));
#elif defined(_WINDOWS)
    while(_fname.find("/") != std::string::npos)
        _fname.replace(_fname.find("/"), 1, "\\\\");

    _fname = _fname.substr(0, _fname.find_last_of("\\"));
    return (_fname.at(_fname.length() - 1) == '\\')
               ? _fname.substr(0, _fname.length() - 1)
               : _fname;
#endif
}

//--------------------------------------------------------------------------------------//

inline int
makedir(std::string _dir, int umask = DEFAULT_UMASK)
{
#if defined(_UNIX)
    while(_dir.find("\\\\") != std::string::npos)
        _dir.replace(_dir.find("\\\\"), 2, "/");
    while(_dir.find("\\") != std::string::npos)
        _dir.replace(_dir.find("\\"), 1, "/");

    if(mkdir(_dir.c_str(), umask) != 0)
    {
        std::stringstream _sdir;
        _sdir << "mkdir -p " << _dir;
        return system(_sdir.str().c_str());
    }
#elif defined(_WINDOWS)
    consume_parameters<int>(umask);
    while(_dir.find("/") != std::string::npos)
        _dir.replace(_dir.find("/"), 1, "\\\\");

    if(_mkdir(_dir.c_str()) != 0)
    {
        std::stringstream _sdir;
        _sdir << "dir " << _dir;
        return system(_sdir.str().c_str());
    }
#endif
    return 0;
}

//--------------------------------------------------------------------------------------//

inline int32_t
get_max_threads()
{
    int32_t _fallback = std::thread::hardware_concurrency();
#ifdef ENV_NUM_THREADS_PARAM
    return get_env<int32_t>(TIMEMORY_STRINGIZE(ENV_NUM_THREADS_PARAM), _fallback);
#else
    return _fallback;
#endif
}

//--------------------------------------------------------------------------------------//

namespace internal
{
//--------------------------------------------------------------------------------------//
inline std::string
dummy_str_return(std::string str)
{
    return str;
}

//--------------------------------------------------------------------------------------//
}  // namespace internal

//======================================================================================//
//
//  Environment
//
//======================================================================================//

class env_settings
{
public:
    typedef std::mutex                        mutex_t;
    typedef std::string                       string_t;
    typedef std::multimap<string_t, string_t> env_map_t;
    typedef std::pair<string_t, string_t>     env_pair_t;

public:
    static env_settings* GetInstance()
    {
        static env_settings* _instance = new env_settings();
        return _instance;
    }

public:
    template <typename _Tp>
    void insert(const std::string& env_id, _Tp val)
    {
        std::stringstream ss;
        ss << std::boolalpha << val;
        std::unique_lock<std::mutex> lock(m_mutex);
        if(m_env.find(env_id) != m_env.end())
        {
            for(const auto& itr : m_env)
                if(itr.first == env_id && itr.second == ss.str())
                {
                    return;
                }
        }
        m_env.insert(env_pair_t(env_id, ss.str()));
    }

    const env_map_t& get() const { return m_env; }
    mutex_t&         mutex() const { return m_mutex; }

    friend std::ostream& operator<<(std::ostream& os, const env_settings& env)
    {
        std::stringstream filler;
        filler.fill('#');
        filler << std::setw(90) << "";
        std::stringstream ss;
        ss << filler.str() << "\n# Environment settings:\n";
        env.mutex().lock();
        for(const auto& itr : env.get())
        {
            ss << "# " << std::setw(35) << std::right << itr.first << "\t = \t"
               << std::left << itr.second << "\n";
        }
        env.mutex().unlock();
        ss << filler.str();
        os << ss.str() << std::endl;
        return os;
    }

private:
    env_map_t       m_env;
    mutable mutex_t m_mutex;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp
get_env(const std::string& env_id, _Tp _default = _Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
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
get_env(const std::string& env_id, std::string _default)
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::stringstream ss;
        ss << env_var;
        return ss.str();
    }
    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  delimit line : e.g. delimit_line("a B\t c", " \t") --> { "a", "B", "c"}
inline str_list_t
delimit(const std::string& _str, const std::string& _delims,
        std::string (*strop)(std::string) = internal::dummy_str_return)
{
    str_list_t _list;
    while(true)
    {
        size_t _end = 0;
        size_t _beg = _str.find_first_not_of(_delims, _end);
        if(_beg == std::string::npos)
            break;
        _end = _str.find_first_of(_delims, _beg);
        if(_beg < _end)
            _list.push_back(strop(_str.substr(_beg, _end - _beg)));
    }
    return _list;
}

//======================================================================================//
//
//  path
//
//======================================================================================//

class path_t : public std::string
{
public:
    typedef std::string             string_t;
    typedef string_t::size_type     size_type;
    typedef std::basic_string<char> stl_string;

public:
    path_t(const std::string& _path)
    : string_t(osrepr(_path))
    {
    }
    path_t(char* _path)
    : string_t(osrepr(string_t(_path)))
    {
    }
    path_t(const path_t& rhs)
    : string_t(osrepr(rhs))
    {
    }
    path_t(const char* _path)
    : string_t(osrepr(string_t(const_cast<char*>(_path))))
    {
    }

    path_t& operator=(const string_t& rhs)
    {
        string_t::operator=(osrepr(rhs));
        return *this;
    }

    path_t& operator=(const path_t& rhs)
    {
        if(this != &rhs)
            string_t::operator=(osrepr(rhs));
        return *this;
    }

    path_t& insert(size_type __pos, const stl_string& __s)
    {
        string_t::operator=(osrepr(string_t::insert(__pos, __s)));
        return *this;
    }

    path_t& insert(size_type __pos, const path_t& __s)
    {
        string_t::operator=(osrepr(string_t::insert(__pos, __s)));
        return *this;
    }

    string_t os() const
    {
#if defined(_WINDOWS)
        return "\\\\";
#elif defined(_UNIX)
        return "/";
#endif
    }

    string_t inverse() const
    {
#if defined(_WINDOWS)
        return "/";
#elif defined(_UNIX)
        return "\\\\";
#endif
    }

    // OS-dependent representation
    string_t osrepr(string_t _path)
    {
#if defined(_WINDOWS)
        while(_path.find("/") != std::string::npos)
            _path.replace(_path.find("/"), 1, "\\\\");
#elif defined(_UNIX)
        while(_path.find("\\\\") != std::string::npos)
            _path.replace(_path.find("\\\\"), 2, "/");
        while(_path.find("\\") != std::string::npos)
            _path.replace(_path.find("\\"), 1, "/");
#endif
        return _path;
    }
};

// use this function to get rid of "unused parameter" warnings
template <typename _Tp, typename... _Args>
void
consume_parameters(_Tp, _Args...)
{
}

//======================================================================================//
//
//
//
//======================================================================================//

template <typename CountedType>
class counted_object
{
public:
    typedef counted_object<CountedType> this_type;
    typedef counted_object<void>        void_type;

public:
    // return number of existing objects:
    static intmax_t           live() { return count().load(); }
    static constexpr intmax_t zero() { return static_cast<intmax_t>(0); }
    static intmax_t           max_depth() { return fmax_depth; }

    static void enable(const bool& val) { fenabled = val; }
    static void set_max_depth(const intmax_t& val) { fmax_depth = val; }
    static bool is_enabled() { return fenabled; }
    static bool is_master() { return thread_number() == 0; }

    template <typename _Tp = CountedType,
              typename std::enable_if<std::is_same<_Tp, void>::value>::type* = nullptr>
    static bool enable()
    {
        return fenabled && fmax_depth > count().load();
    }
    // the void type is consider the global setting
    template <typename _Tp = CountedType,
              typename std::enable_if<!std::is_same<_Tp, void>::value>::type* = nullptr>
    static bool enable()
    {
        return void_type::is_enabled() && void_type::max_depth() > count().load() &&
               fenabled && fmax_depth > count().load();
    }

protected:
    // default constructor
    counted_object()
    : m_instance(count()++)
    {
    }
    ~counted_object() { --count(); }
    counted_object(const this_type&)
    : m_instance(count()++)
    {
    }
    counted_object(this_type&&) = default;
    this_type& operator         =(const this_type&) { return *this; }
    this_type& operator=(this_type&& rhs) = default;

protected:
    intmax_t m_instance;

private:
    // number of existing objects
    static intmax_t&             thread_number();
    static std::atomic_intmax_t& master_count();
    static std::atomic_intmax_t& count();
    static intmax_t              fmax_depth;
    static bool                  fenabled;
};

//--------------------------------------------------------------------------------------//

template <typename CountedType>
intmax_t&
counted_object<CountedType>::thread_number()
{
    static std::atomic<intmax_t> _all_instance;
    static thread_local intmax_t _instance = _all_instance++;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
std::atomic_intmax_t&
counted_object<CountedType>::master_count()
{
    static std::atomic_intmax_t _instance(0);
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
std::atomic_intmax_t&
counted_object<CountedType>::count()
{
    if(thread_number() == 0)
        return master_count();
    static thread_local std::atomic_intmax_t _instance(master_count().load());
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
intmax_t counted_object<CountedType>::fmax_depth = std::numeric_limits<intmax_t>::max();

//--------------------------------------------------------------------------------------//

template <typename CountedType>
bool counted_object<CountedType>::fenabled = true;

//======================================================================================//
//
//
//
//======================================================================================//

template <typename HashedType>
class hashed_object
{
public:
    typedef hashed_object<HashedType> this_type;

public:
    // return running hash of existing objects
    static intmax_t           live() { return hash(); }
    static constexpr intmax_t zero() { return static_cast<intmax_t>(0); }

protected:
    // default constructor
    template <typename _Tp,
              typename std::enable_if<!std::is_integral<_Tp>::value>::type* = nullptr>
    explicit hashed_object(const _Tp& _val)
    : m_hash(std::hash<_Tp>(_val))
    {
        hash() += m_hash;
    }

    template <typename _Tp,
              typename std::enable_if<std::is_integral<_Tp>::value>::type* = nullptr>
    explicit hashed_object(const _Tp& _val)
    : m_hash(_val)
    {
        hash() += m_hash;
    }

    ~hashed_object() { hash() -= m_hash; }
    hashed_object(const this_type&)     = default;
    explicit hashed_object(this_type&&) = default;
    static intmax_t& hash();

private:
    // number of existing objects
    static intmax_t& thread_number();
    static intmax_t& master_hash();
    intmax_t         m_hash;
};

//--------------------------------------------------------------------------------------//

template <typename HashedType>
intmax_t&
hashed_object<HashedType>::thread_number()
{
    static std::atomic<intmax_t> _all_instance;
    static thread_local intmax_t _instance = _all_instance++;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename HashedType>
intmax_t&
hashed_object<HashedType>::master_hash()
{
    static intmax_t _instance = 0;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename HashedType>
intmax_t&
hashed_object<HashedType>::hash()
{
    if(thread_number() == 0)
        return master_hash();
    static thread_local intmax_t _instance = master_hash();
    return _instance;
}

//======================================================================================//
//
//
//
//======================================================================================//

template <typename ObjectType>
class graph_object
{
public:
    struct graph_node
    {
        intmax_t    id;
        ObjectType* ptr;

        graph_node(intmax_t _id, ObjectType* _ptr)
        : id(_id)
        , ptr(_ptr)
        {
        }

        bool operator==(const graph_node& rhs) const { return id == rhs.id; }
        bool operator!=(const graph_node& rhs) const { return !(*this == rhs); }
    };

    using this_type      = graph_object<ObjectType>;
    using void_type      = graph_object<void>;
    using graph_t        = tim::graph<graph_node>;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;
    using pointer_t      = std::unique_ptr<this_type>;
    using singleton_t    = singleton<this_type>;

    struct graph_data
    {
        graph_t  m_graph;
        iterator m_current = nullptr;
        iterator m_head    = nullptr;

        graph_t&  graph() { return m_graph; }
        iterator& current() { return m_current; }
        iterator& head() { return m_head; }

        iterator       begin() { return m_graph.begin(); }
        iterator       end() { return m_graph.end(); }
        const_iterator begin() const { return m_graph.cbegin(); }
        const_iterator end() const { return m_graph.cend(); }
        const_iterator cbegin() const { return m_graph.cbegin(); }
        const_iterator cend() const { return m_graph.cend(); }

        void pop_graph() { m_current = graph_t::parent(m_current); }
    };

public:
    // return number of existing objects:
    // default constructor
    graph_object()
    {
        if(!instance().get())
        {
            instance().reset(this);
            if(!master_instance())
            {
                master_instance() = this;
            }
        }
    }
    ~graph_object()
    {
        // if(instance())
        //    instance()->current() = graph_t::parent(instance()->current());
    }

    graph_object(const this_type&) = default;
    graph_object(this_type&&)      = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&& rhs) = default;

    static graph_object*& master_instance()
    {
        static graph_object* _instance = nullptr;
        return _instance;
    }

    static pointer_t& instance()
    {
        static thread_local pointer_t _instance = pointer_t(nullptr);
        return _instance;
    }

    void insert(intmax_t hash_id, ObjectType* obj)
    {
        graph_node node(hash_id, obj);
        if(!m_data.head())
        {
            if(this == master_instance())
            {
                m_data.head()    = m_data.graph().set_head(node);
                m_data.current() = m_data.head();
            }
            else
            {
                m_data.head()    = m_data.graph().set_head(*master_instance()->current());
                m_data.current() = m_data.head();
            }
        }
        else
        {
            using sibling_itr = typename graph_t::sibling_iterator;
            for(sibling_itr itr = m_data.begin(); itr != m_data.end(); ++itr)
            {
                if(node == *itr)
                    return;
            }
            m_data.current() = instance()->graph().append_child(current(), node);
        }
    }

protected:
    graph_data m_data;

    iterator& current() { return m_data.current(); }
    iterator& head() { return m_data.head(); }
    graph_t&  graph() { return m_data.graph(); }

private:
    // number of existing objects
};

//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
