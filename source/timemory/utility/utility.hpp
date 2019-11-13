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
 * \headerfile utility.hpp "timemory/utility/utility.hpp"
 * General utility functions
 *
 */

#pragma once

#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"

// C library
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
// I/O
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
// general
#include <functional>
#include <limits>
#include <utility>
// container
#include <map>
#include <vector>
// threading
#include <atomic>
#include <mutex>
#include <thread>

#if defined(_UNIX)
#    include <cxxabi.h>
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
#if !defined(TIMEMORY_TIM_LINESTR)
#    define TIMEMORY_TIM_LINESTR TIMEMORY_STRINGIZE(__LINE__)
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//
// use this function to get rid of "unused parameter" warnings
template <typename... _Args>
void
consume_parameters(_Args&&...)
{}

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

using string_t    = std::string;
using str_list_t  = std::vector<string_t>;
using mutex_t     = std::recursive_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;

//======================================================================================//
//
//  General functions
//
//======================================================================================//

template <typename _Tp>
mutex_t&
type_mutex(const uint64_t& _n = 0)
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
demangle(const char* _cstr)
{
#if defined(_TIMEMORY_ENABLE_DEMANGLE)
    // demangling a string when delimiting
    int   _ret    = 0;
    char* _demang = abi::__cxa_demangle(_cstr, 0, 0, &_ret);
    if(_demang && _ret == 0)
        return std::string(const_cast<const char*>(_demang));
    else
        return _cstr;
#else
    return _cstr;
#endif
}

//--------------------------------------------------------------------------------------//

inline std::string
demangle(const std::string& _str)
{
    return demangle(_str.c_str());
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline std::string
demangle()
{
    return demangle(typeid(_Tp).name());
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
        _fname.replace(_fname.find("/"), 1, "\\");

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

    if(_dir.length() == 0)
        return 0;

    if(mkdir(_dir.c_str(), umask) != 0)
    {
        std::stringstream _sdir;
        _sdir << "mkdir -p " << _dir;
        return system(_sdir.str().c_str());
    }
#elif defined(_WINDOWS)
    consume_parameters(umask);
    while(_dir.find("/") != std::string::npos)
        _dir.replace(_dir.find("/"), 1, "\\");

    if(_dir.length() == 0)
        return 0;

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
//  delimit a string into a set
//
template <typename _Container = std::vector<std::string>,
          typename _Predicate = std::function<string_t(string_t)>>
inline _Container
delimit(const string_t& line, const string_t& delimiters = ",; ",
        _Predicate&& predicate = [](string_t s) -> string_t { return s; })
{
    auto _get_first_not_of = [&delimiters](const string_t& _string, const size_t& _beg) {
        return _string.find_first_not_of(delimiters, _beg);
    };

    auto _get_first_of = [&delimiters](const string_t& _string, const size_t& _beg) {
        return _string.find_first_of(delimiters, _beg);
    };

    _Container _result;
    size_t     _beginp = 0;  // position that is the beginning of the new string
    size_t     _delimp = 0;  // position of the delimiter in the string
    while(_beginp < line.length() && _delimp < line.length())
    {
        // find the first character (starting at _end) that is not a delimiter
        _beginp = _get_first_not_of(line, _delimp);
        // if no a character after or at _end that is not a delimiter is not found
        // then we are done
        if(_beginp == string_t::npos)
        {
            break;
        }
        // starting at the position of the new string, find the next delimiter
        _delimp = _get_first_of(line, _beginp);
        // if(d2 == string_t::npos) { d2 = string_t::npos; }
        string_t _tmp = "";
        try
        {
            // starting at the position of the new string, get the characters
            // between this position and the next delimiter
            _tmp = line.substr(_beginp, _delimp - _beginp);
        } catch(std::exception& e)
        {
            // print the exception but don't fail, unless maybe it should?
            std::stringstream ss;
            ss << e.what();
            fprintf(stderr, "%s\n", ss.str().c_str());
        }
        // don't add empty strings
        if(!_tmp.empty())
        {
            _result.insert(_result.end(), predicate(_tmp));
        }
    }
    return _result;
}

//======================================================================================//
//
//  Environment
//
//======================================================================================//

class env_settings
{
public:
    using mutex_t     = std::recursive_mutex;
    using string_t    = std::string;
    using env_map_t   = std::map<string_t, string_t>;
    using env_uomap_t = std::map<string_t, string_t>;
    using env_pair_t  = std::pair<string_t, string_t>;

public:
    static env_settings* instance();

public:
    template <typename _Tp>
    void insert(const std::string& env_id, _Tp val)
    {
#if !defined(TIMEMORY_DISABLE_STORE_ENVIRONMENT)
        std::stringstream ss;
        ss << std::boolalpha << val;

        auto_lock_t lk(env_settings::mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        if(m_env.find(env_id) == m_env.end() || m_env.find(env_id)->second != ss.str())
            m_env[env_id] = ss.str();
#endif
    }

    env_map_t get() const
    {
        auto      _tmp = m_env;
        env_map_t _ret;
        for(const auto& itr : _tmp)
            _ret[itr.first] = itr.second;
        return _ret;
    }

    static mutex_t& mutex()
    {
        static mutex_t m_mutex;
        return m_mutex;
    }

    friend std::ostream& operator<<(std::ostream& os, const env_settings& env)
    {
        std::stringstream filler;
        filler.fill('#');
        filler << std::setw(90) << "";
        std::stringstream ss;
        ss << filler.str() << "\n# Environment settings:\n";

        auto_lock_t lk(env_settings::mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        auto _data = env.get();
        lk.unlock();

        for(const auto& itr : _data)
        {
            ss << "# " << std::setw(35) << std::right << itr.first << "\t = \t"
               << std::left << itr.second << "\n";
        }
        ss << filler.str();
        os << ss.str() << std::endl;
        return os;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto_lock_t lk(env_settings::mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();
        ar(serializer::make_nvp("environment", m_env));
    }

private:
    env_uomap_t m_env;
};

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_EXTERN_INIT)
inline env_settings*
env_settings::instance()
{
    static env_settings* _instance = new env_settings();
    return _instance;
}
#endif

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

inline void
print_env(std::ostream& os = std::cout)
{
    os << (*env_settings::instance());
}

//--------------------------------------------------------------------------------------//
//  delimit line : e.g. delimit_line("a B\t c", " \t") --> { "a", "B", "c"}
inline str_list_t
delimit(std::string _str, const std::string& _delims)
{
    str_list_t _list;
    while(_str.length() > 0)
    {
        size_t _end = 0;
        size_t _beg = _str.find_first_not_of(_delims, _end);
        if(_beg == std::string::npos)
            break;
        _end = _str.find_first_of(_delims, _beg);
        if(_beg < _end)
        {
            _list.push_back(_str.substr(_beg, _end - _beg));
            _str.erase(_beg, _end - _beg);
        }
    }
    return _list;
}

//--------------------------------------------------------------------------------------//
//  delimit line : e.g. delimit_line("a B\t c", " \t") --> { "a", "B", "c"}
template <typename _Func>
inline str_list_t
delimit(std::string _str, const std::string& _delims,
        const _Func& strop = [](const std::string& s) { return s; })
{
    str_list_t _list;
    while(_str.length() > 0)
    {
        size_t _end = 0;
        size_t _beg = _str.find_first_not_of(_delims, _end);
        if(_beg == std::string::npos)
            break;
        _end = _str.find_first_of(_delims, _beg);
        if(_beg < _end)
        {
            _list.push_back(strop(_str.substr(_beg, _end - _beg)));
            _str.erase(_beg, _end - _beg);
        }
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
    using string_t   = std::string;
    using size_type  = string_t::size_type;
    using stl_string = std::basic_string<char>;

public:
    path_t(const std::string& _path)
    : string_t(osrepr(_path))
    {}
    path_t(char* _path)
    : string_t(osrepr(string_t(_path)))
    {}
    path_t(const path_t& rhs)
    : string_t(osrepr(rhs))
    {}
    path_t(const char* _path)
    : string_t(osrepr(string_t(const_cast<char*>(_path))))
    {}

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
        return "\\";
#elif defined(_UNIX)
        return "/";
#endif
    }

    string_t inverse() const
    {
#if defined(_WINDOWS)
        return "/";
#elif defined(_UNIX)
        return "\\";
#endif
    }

    // OS-dependent representation
    string_t osrepr(string_t _path)
    {
#if defined(_WINDOWS)
        while(_path.find("/") != std::string::npos)
            _path.replace(_path.find("/"), 1, "\\");
#elif defined(_UNIX)
        while(_path.find("\\\\") != std::string::npos)
            _path.replace(_path.find("\\\\"), 2, "/");
        while(_path.find("\\") != std::string::npos)
            _path.replace(_path.find("\\"), 1, "/");
#endif
        return _path;
    }
};

//======================================================================================//
//
//  Counting the number of objects of a given type
//
//======================================================================================//

template <typename CountedType>
class static_counted_object
{
public:
    typedef static_counted_object<CountedType> this_type;

public:
    // return number of existing objects:
    static int64_t live() { return count().load(); }
    static bool    is_master() { return thread_number() == 0; }

protected:
    // default constructor
    static_counted_object()
    : m_thread(thread_number())
    , m_count(count()++)
    {}
    ~static_counted_object()
    {
        if(m_decrement)
        {
            --count();
        }
    }
    static_counted_object(const this_type& rhs)
    : m_decrement(false)
    , m_thread(rhs.m_thread)
    , m_count(rhs.m_count)
    {}
    this_type& operator=(const this_type& rhs)
    {
        if(this != &rhs)
        {
            m_decrement = false;
            m_thread    = rhs.m_thread;
            m_count     = rhs.m_count;
        }
        return *this;
    }
    static_counted_object(this_type&& rhs)
    : m_decrement(false)
    , m_thread(std::move(rhs.m_thread))
    , m_count(std::move(rhs.m_count))
    {}

protected:
    bool    m_decrement = true;
    int64_t m_thread;
    int64_t m_count;

private:
    // number of existing objects
    static std::atomic<int64_t>& count();
    static int64_t&              thread_number();
};

//--------------------------------------------------------------------------------------//

template <typename CountedType>
int64_t&
static_counted_object<CountedType>::thread_number()
{
    static std::atomic<int64_t> _all_instance;
    static thread_local int64_t _instance = _all_instance++;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
std::atomic<int64_t>&
static_counted_object<CountedType>::count()
{
    static thread_local std::atomic<int64_t> _instance;
    return _instance;
}

//======================================================================================//
//
//  Counting the number of objects of a given type
//
//======================================================================================//

template <typename CountedType>
class counted_object
{
public:
    using this_type = counted_object<CountedType>;
    using void_type = counted_object<void>;

public:
    // return number of existing objects:
    static int64_t           live() { return count().load(); }
    static constexpr int64_t zero() { return static_cast<int64_t>(0); }
    static int64_t           max_depth() { return fmax_depth; }

    static void enable(const bool& val) { fenabled = val; }
    static void set_max_depth(const int64_t& val) { fmax_depth = val; }
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

    void activate_noop()
    {
        count()--;
        is_noop = true;
    }

protected:
    // default constructor
    counted_object()
    : m_count(count()++)
    {}
    ~counted_object()
    {
        if(!is_noop)
            --count();
    }
    explicit counted_object(const this_type&)
    : m_count(count()++)
    {}
    explicit counted_object(this_type&&) = default;
    this_type& operator=(const this_type&) = default;
    this_type& operator=(this_type&& rhs) = default;

protected:
    int64_t m_count;

private:
    bool is_noop = false;

private:
    // number of existing objects
    static int64_t&              thread_number();
    static std::atomic<int64_t>& master_count();
    static std::atomic<int64_t>& count();
    static int64_t               fmax_depth;
    static bool                  fenabled;
};

//--------------------------------------------------------------------------------------//

template <typename CountedType>
int64_t&
counted_object<CountedType>::thread_number()
{
    static std::atomic<int64_t> _all_instance;
    static thread_local int64_t _instance = _all_instance++;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
std::atomic<int64_t>&
counted_object<CountedType>::master_count()
{
    static std::atomic<int64_t> _instance(0);
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
std::atomic<int64_t>&
counted_object<CountedType>::count()
{
    if(thread_number() == 0)
        return master_count();
    static thread_local std::atomic<int64_t> _instance(master_count().load());
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename CountedType>
int64_t counted_object<CountedType>::fmax_depth = std::numeric_limits<int64_t>::max();

//--------------------------------------------------------------------------------------//

template <typename CountedType>
bool counted_object<CountedType>::fenabled = true;

//======================================================================================//
//
//  Running hash for object and children of a given type
//
//======================================================================================//

template <typename HashedType>
class hashed_object
{
public:
    using this_type = hashed_object<HashedType>;

public:
    // return running hash of existing objects
    static int64_t           live() { return hash(); }
    static constexpr int64_t zero() { return static_cast<int64_t>(0); }

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
    explicit hashed_object(const this_type&) = default;
    explicit hashed_object(this_type&&)      = default;
    static int64_t& hash();

protected:
    int64_t m_hash;

private:
    // number of existing objects
    static int64_t& thread_number();
    static int64_t& master_hash();
};

//--------------------------------------------------------------------------------------//

template <typename HashedType>
int64_t&
hashed_object<HashedType>::thread_number()
{
    static std::atomic<int64_t> _all_instance;
    static thread_local int64_t _instance = _all_instance++;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename HashedType>
int64_t&
hashed_object<HashedType>::master_hash()
{
    static int64_t _instance = 0;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename HashedType>
int64_t&
hashed_object<HashedType>::hash()
{
    if(thread_number() == 0)
        return master_hash();
    static thread_local int64_t _instance = master_hash();
    return _instance;
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
