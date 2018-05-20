// MIT License
//
// Copyright (c) 2018, The Regents of the University of California, 
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

/** \file utility.hpp
 * \headerfile utility.hpp "timemory/utility.hpp"
 * General utility functions
 *
 */

#ifndef TIMEMORY_UTIL_INTERNAL_HPP
#define TIMEMORY_UTIL_INTERNAL_HPP

// C library
#include <stdint.h>
#include <stdlib.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
// I/O
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
// general
#include <exception>
#include <stdexcept>
#include <functional>
#include <limits>
// container
#include <deque>
#include <set>
#include <vector>
// threading
#include <atomic>
#include <mutex>
#include <thread>

#include "timemory/macros.hpp"

#if defined(_UNIX)
#   include <stdio.h>
#   include <string.h>
#   include <errno.h>
#   include <sys/stat.h>
#   include <sys/types.h>
#elif defined(_WINDOWS)
#   include <direct.h>
#endif

#if !defined(DEFAULT_UMASK)
#   define DEFAULT_UMASK 0777
#endif

#if defined(_OPENMP)
#   include <omp.h>
#endif

//----------------------------------------------------------------------------//

// stringify some macro -- uses TIMEMORY_STRINGIFY2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#   define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#   define TIMEMORY_STRINGIZE2(X) #X
#endif

// stringify the __LINE__ macro
#if !defined(TIMEMORY_LINE_STRING)
#   define TIMEMORY_LINE_STRING TIMEMORY_STRINGIZE(__LINE__)
#endif

//----------------------------------------------------------------------------//

namespace tim
{

//----------------------------------------------------------------------------//

template <typename _Tp>
inline bool isfinite(const _Tp& arg)
{
    #if defined(_WINDOWS)
    // Windows seems to be missing std::isfinite
    return (arg == arg &&
            arg !=  std::numeric_limits<_Tp>::infinity() &&
            arg != -std::numeric_limits<_Tp>::infinity()) ? true : false;
    #else
    return std::isfinite(arg);
    #endif
}

//----------------------------------------------------------------------------//

typedef std::string                 string_t;
typedef std::deque<string_t>        str_list_t;
typedef std::mutex                  mutex_t;
typedef std::unique_lock<mutex_t>   auto_lock_t;

//----------------------------------------------------------------------------//

template <typename _Tp>
mutex_t& type_mutex(const uintmax_t& _n = 0)
{
    static mutex_t* _mutex = new mutex_t();
    if(_n == 0)
        return *_mutex;

    static std::vector<mutex_t*> _mutexes;
    if(_n > _mutexes.size())
        _mutexes.resize(_n, nullptr);
    if(!_mutexes[_n])
        _mutexes[_n] = new mutex_t();
    return *(_mutexes[_n-1]);
}

//----------------------------------------------------------------------------//

namespace internal
{
inline std::string dummy_str_return(std::string str) { return str; }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
_Tp get_env(const std::string& env_id, _Tp _default = _Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp var = _Tp();
        iss >> var;
        return var;
    }
    // return default if not specified in environment
    return _default;
}

//----------------------------------------------------------------------------//
// specialization for string since the above will have issues if string
// includes spaces
template <> inline
std::string get_env(const std::string& env_id, std::string _default)
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

//----------------------------------------------------------------------------//
//  delimit line : e.g. delimit_line("a B\t c", " \t") --> { "a", "B", "c"}
inline str_list_t
delimit(const std::string& _str, const std::string& _delims,
        std::string (*strop)(std::string) = internal::dummy_str_return)
{
    str_list_t _list;
    size_t _beg = 0;
    size_t _end = 0;
    while(true)
    {
        _beg = _str.find_first_not_of(_delims, _end);
        if(_beg == std::string::npos)
            break;
        _end = _str.find_first_of(_delims, _beg);
        if(_beg < _end)
            _list.push_back(strop(_str.substr(_beg, _end - _beg)));
    }
    return _list;
}

//----------------------------------------------------------------------------//

class path_t : public std::string
{
public:
    typedef std::string         string_t;
    typedef string_t::size_type size_type;

public:
    path_t(const std::string& _path)    : string_t(osrepr(_path)) { }
    path_t(char* _path)                 : string_t(osrepr(string_t(_path))) { }
    path_t(const path_t& rhs)           : string_t(osrepr(rhs)) { }
    path_t(const char* _path)
    : string_t(osrepr(string_t(const_cast<char*>(_path)))) { }

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

    path_t& insert(size_type __pos, const string_t& __s)
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
		//auto _orig = _path;
#if defined(_WINDOWS)
		while (_path.find("/") != std::string::npos)
			_path.replace(_path.find("/"), 1, "\\\\");
#elif defined(_UNIX)
		while (_path.find("\\\\") != std::string::npos)
			_path.replace(_path.find("\\\\"), 2, "/");
		while (_path.find("\\") != std::string::npos)
			_path.replace(_path.find("\\"), 1, "/");
#endif
		//std::cout << "path_t::osrepr - converted \"" << _orig << "\" to \""
		//          << _path << "\"..." << std::endl;
		return _path;
	}
};

//----------------------------------------------------------------------------//
// use this function to get rid of "unused parameter" warnings
template <typename _Tp, typename... _Args>
void consume_parameters(_Tp, _Args...)
{ }

//----------------------------------------------------------------------------//

inline std::string dirname(std::string _fname)
{
#if defined(_UNIX)
    char* _cfname = realpath(_fname.c_str(), NULL);
    _fname = std::string(_cfname);
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
    return (_fname.at(_fname.length()-1) == '\\')
            ? _fname.substr(0, _fname.length()-1)
            : _fname;
#endif
}

//----------------------------------------------------------------------------//

inline int makedir(std::string _dir, int umask = DEFAULT_UMASK)
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

//----------------------------------------------------------------------------//

inline int32_t get_max_threads()
{
#if defined(_OPENMP)
    int32_t _fallback = omp_get_max_threads();
#else
    int32_t _fallback = 1;
#endif

#ifdef ENV_NUM_THREADS_PARAM
    return get_env<int32_t>( TIMEMORY_STRINGIZE( ENV_NUM_THREADS_PARAM ),
                             _fallback );
#else
    return _fallback;
#endif
}

//----------------------------------------------------------------------------//

} // namespace tim

//----------------------------------------------------------------------------//

#endif
