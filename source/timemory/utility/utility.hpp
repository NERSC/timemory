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

/** \file utility/utility.hpp
 * \headerfile utility/utility.hpp "timemory/utility/utility.hpp"
 * General utility functions
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#if defined(_WINDOWS)
// Without this include on windows launch_process is needed by makedir but is
// missing at link time. With this include on linux there is a problem with
// delimit not being defined before it is used in popen.hpp. I think including
// popen.hpp at the end of this header would be portable and not need the ifdef,
// but that would be too weird, so we may have to live with the ifdef. (At
// least until makedir is implemented using std::filesystem.)
#    include "timemory/utility/popen.hpp"
#endif

// C library
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// I/O
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
// general
#include <functional>
#include <limits>
#include <regex>
#include <typeindex>
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
#    include <execinfo.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#elif defined(_WINDOWS)
#    include <direct.h>
using pid_t = int;
#endif

#if !defined(DEFAULT_UMASK)
#    define DEFAULT_UMASK 0777
#endif

//--------------------------------------------------------------------------------------//

// stringify some macro -- uses TIMEMORY_STRINGIZE2 which does the actual
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
// alias here for common string type
// there is also a string_view_t alias in macros/language.hpp which is std::string in
// c++14 and std::string_view in c++17 and newer
using string_t = std::string;

// thread synchronization aliases. Recursive mutex is used for convenience since the
// performance penalty vs. a regular mutex is not really an issue since there are not
// many locks in general.
using mutex_t     = std::recursive_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;

//--------------------------------------------------------------------------------------//
// definition in popen.hpp
bool
launch_process(const char* cmd, const std::string& extra = "");

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline bool
isfinite(const Tp& arg)
{
#if defined(_WINDOWS)
    // Windows seems to be missing std::isfinite
    return (arg == arg && arg != std::numeric_limits<Tp>::infinity() &&
            arg != -std::numeric_limits<Tp>::infinity())
               ? true
               : false;
#else
    return std::isfinite(arg);
#endif
}

//======================================================================================//
//
//  General functions
//
//======================================================================================//

template <typename Tp, typename ApiT = TIMEMORY_API>
auto&
type_mutex(const uint64_t& _n = 0)
{
    static std::vector<mutex_t*> _mutexes{};
    if(_n < _mutexes.size())
        return *(_mutexes.at(_n));

    static mutex_t _internal{};
    auto_lock_t    _internal_lk{ _internal };

    // check in case another already resized
    if(_n < _mutexes.size())
        return *(_mutexes.at(_n));

    // append new mutexes
    auto i0 = _mutexes.size();
    for(auto i = i0; i < (_n + 1); ++i)
        _mutexes.push_back(new mutex_t{});
    return *(_mutexes.at(_n));
}

//--------------------------------------------------------------------------------------//

inline std::string
demangle(const char* _cstr)
{
#if defined(TIMEMORY_ENABLE_DEMANGLE)
    // demangling a string when delimiting
    int   _ret    = 0;
    char* _demang = abi::__cxa_demangle(_cstr, nullptr, nullptr, &_ret);
    if(_demang && _ret == 0)
        return std::string(const_cast<const char*>(_demang));
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

template <typename Tp>
inline auto
demangle()
{
    // a type demangle will always be the same
    static auto _value = demangle(typeid(Tp).name());
    return _value;
}

//--------------------------------------------------------------------------------------//

namespace internal
{
template <typename Tp, typename Up = Tp>
inline auto
try_demangle(int) -> decltype(demangle<Tp>(), std::string())
{
    return demangle<Tp>();
}
//
template <typename Tp, typename Up = Tp>
inline auto
try_demangle(long)
{
    return "";
}
}  // namespace internal

template <typename Tp>
inline auto
try_demangle()
{
    return internal::try_demangle<Tp>(0);
}

//--------------------------------------------------------------------------------------//

namespace internal
{
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(int) -> decltype(demangle<Tp>(), size_t{})
{
    return std::type_index(typeid(Tp)).hash_code();
}
//
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(long)
{
    return 0;
}
}  // namespace internal

template <typename Tp>
inline auto
typeid_hash()
{
    return internal::typeid_hash<Tp>(0);
}

//--------------------------------------------------------------------------------------//

template <typename T>
inline T
from_string(const std::string& str)
{
    std::stringstream ss;
    ss << str;
    T val{};
    ss >> val;
    return val;
}

//--------------------------------------------------------------------------------------//

template <typename T>
inline T
from_string(const char* cstr)
{
    std::stringstream ss;
    ss << cstr;
    T val{};
    ss >> val;
    return val;
}

//--------------------------------------------------------------------------------------//

inline std::string
dirname(std::string _fname)
{
#if defined(_UNIX)
    char* _cfname = realpath(_fname.c_str(), nullptr);
    _fname        = std::string(_cfname);
    free(_cfname);

    while(_fname.find("\\\\") != std::string::npos)
        _fname.replace(_fname.find("\\\\"), 2, "/");
    while(_fname.find('\\') != std::string::npos)
        _fname.replace(_fname.find('\\'), 1, "/");

    return _fname.substr(0, _fname.find_last_of('/'));
#elif defined(_WINDOWS)
    while(_fname.find('/') != std::string::npos)
        _fname.replace(_fname.find('/'), 1, "\\");

    _fname = _fname.substr(0, _fname.find_last_of('\\'));
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
    while(_dir.find('\\') != std::string::npos)
        _dir.replace(_dir.find('\\'), 1, "/");

    if(_dir.length() == 0)
        return 0;

    int ret = mkdir(_dir.c_str(), umask);
    if(ret != 0)
    {
        int err = errno;
        if(err != EEXIST)
        {
            std::cerr << "mkdir(" << _dir.c_str() << ", " << umask
                      << ") returned: " << ret << std::endl;
            std::stringstream _sdir;
            _sdir << "/bin/mkdir -p " << _dir;
            return (launch_process(_sdir.str().c_str())) ? 0 : 1;
        }
    }
#elif defined(_WINDOWS)
    consume_parameters(umask);
    while(_dir.find('/') != std::string::npos)
        _dir.replace(_dir.find('/'), 1, "\\");

    if(_dir.length() == 0)
        return 0;

    int ret = _mkdir(_dir.c_str());
    if(ret != 0)
    {
        int err = errno;
        if(err != EEXIST)
        {
            std::cerr << "_mkdir(" << _dir.c_str() << ") returned: " << ret << std::endl;
            std::stringstream _sdir;
            _sdir << "mkdir " << _dir;
            return (launch_process(_sdir.str().c_str())) ? 0 : 1;
        }
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

inline bool
get_bool(const std::string& strbool, bool _default = false) noexcept
{
    // empty string returns default
    if(strbool.empty())
        return _default;

    // check if numeric
    if(strbool.find_first_not_of("0123456789") == std::string::npos)
    {
        if(strbool.length() > 1 || strbool[0] != '0')
            return true;
        return false;
    }

    // convert to lowercase
    auto _val = std::string{ strbool };
    for(auto& itr : _val)
        itr = tolower(itr);

    // check for matches to acceptable forms of false
    for(const auto& itr : { "off", "false", "no", "n", "f" })
    {
        if(_val == itr)
            return false;
    }

    // check for matches to acceptable forms of true
    for(const auto& itr : { "on", "true", "yes", "y", "t" })
    {
        if(_val == itr)
            return false;
    }

    return _default;
}

//--------------------------------------------------------------------------------------//
//
#if defined(_UNIX)
//
static inline auto
demangle_backtrace(const char* cstr)
{
    auto _trim = [](std::string& _sub, size_t& _len) {
        size_t _pos = 0;
        while((_pos = _sub.find_first_of(' ')) == 0)
        {
            _sub = _sub.erase(_pos, 1);
            --_len;
        }
        while((_pos = _sub.find_last_of(' ')) == _sub.length() - 1)
        {
            _sub = _sub.substr(0, _sub.length() - 1);
            --_len;
        }
        return _sub;
    };

    auto str = demangle(std::string(cstr));
    auto beg = str.find("(");
    if(beg == std::string::npos)
    {
        beg = str.find("_Z");
        if(beg != std::string::npos)
            beg -= 1;
    }
    auto end = str.find("+", beg);
    if(beg != std::string::npos && end != std::string::npos)
    {
        auto len = end - (beg + 1);
        auto sub = str.substr(beg + 1, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg + 1, len, dem);
    }
    else if(beg != std::string::npos)
    {
        auto len = str.length() - (beg + 1);
        auto sub = str.substr(beg + 1, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg + 1, len, dem);
    }
    else if(end != std::string::npos)
    {
        auto len = end;
        auto sub = str.substr(beg, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg, len, dem);
    }
    return str;
}
//
static inline auto
demangle_backtrace(const std::string& str)
{
    return demangle_backtrace(str.c_str());
}
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE auto
get_backtrace()
{
    static_assert((Depth - Offset) >= 1, "Error Depth - Offset should be >= 1");

    using type = const char*;
    // destination
    std::array<type, Depth> btrace;
    btrace.fill(nullptr);

    // plus one for this stack-frame
    std::array<void*, Depth + Offset> buffer;
    // size of returned buffer
    auto sz = backtrace(buffer.data(), Depth + Offset);
    // size of relevant data
    auto n = sz - Offset;

    // skip ahead (Offset + 1) stack frames
    char** bsym = backtrace_symbols(buffer.data() + Offset, n);

    // report errors
    if(bsym == nullptr)
        perror("backtrace_symbols");
    else
    {
        for(decltype(n) i = 0; i < n; ++i)
            btrace[i] = bsym[i];
    }
    return btrace;
}
//
template <size_t Depth, size_t Offset, typename Func>
TIMEMORY_NOINLINE auto
get_backtrace(Func&& func = [](const char* inp) { return std::string(inp); })
{
    static_assert((Depth - Offset) >= 1, "Error Depth - Offset should be >= 1");

    using type = std::result_of_t<Func(const char*)>;
    // destination
    std::array<type, Depth> btrace;
    btrace.fill((std::is_pointer<type>::value) ? nullptr : type{});

    // plus one for this stack-frame
    std::array<void*, Depth + Offset> buffer;
    // size of returned buffer
    auto sz = backtrace(buffer.data(), Depth + Offset);
    // size of relevant data
    auto n = sz - Offset;

    // skip ahead (Offset + 1) stack frames
    char** bsym = backtrace_symbols(buffer.data() + Offset, n);

    // report errors
    if(bsym == nullptr)
        perror("backtrace_symbols");
    else
    {
        for(decltype(n) i = 0; i < n; ++i)
            btrace[i] = func(bsym[i]);
        free(bsym);
    }
    return btrace;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE auto
get_demangled_backtrace()
{
    auto demangle_bt = [](const char* cstr) { return demangle_backtrace(cstr); };
    return get_backtrace<Depth, Offset>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE void
print_backtrace(std::ostream& os = std::cerr)
{
    auto              bt = tim::get_backtrace<Depth, Offset>();
    std::stringstream ss;
    for(const auto& itr : bt)
    {
        ss << "\nBacktrace:\n";
        if(itr.length() > 0)
            ss << itr << "\n";
    }
    ss << "\n";
    os << std::flush;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 3>
TIMEMORY_NOINLINE void
print_demangled_backtrace(std::ostream& os = std::cerr)
{
    auto              bt = tim::get_demangled_backtrace<Depth, Offset>();
    std::stringstream ss;
    for(const auto& itr : bt)
    {
        ss << "\nBacktrace:\n";
        if(itr.length() > 0)
            ss << itr << "\n";
    }
    ss << "\n";
    os << std::flush;
}
//
#else
//
// define these dummy functions since they are used in operation::decode
//
static inline auto
demangle_backtrace(const char* cstr)
{
    return std::string(cstr);
}
//
static inline auto
demangle_backtrace(const std::string& str)
{
    return str;
}
//
#endif
//
//--------------------------------------------------------------------------------------//
//  delimit a string into a set
//
template <typename ContainerT = std::vector<std::string>,
          typename PredicateT = std::function<std::string(const std::string&)>>
inline ContainerT
delimit(const std::string& line, const std::string& delimiters = "\"',;: ",
        PredicateT&& predicate = [](const std::string& s) -> std::string { return s; })
{
    ContainerT _result{};
    size_t     _beginp = 0;  // position that is the beginning of the new string
    size_t     _delimp = 0;  // position of the delimiter in the string
    while(_beginp < line.length() && _delimp < line.length())
    {
        // find the first character (starting at _delimp) that is not a delimiter
        _beginp = line.find_first_not_of(delimiters, _delimp);
        // if no a character after or at _end that is not a delimiter is not found
        // then we are done
        if(_beginp == std::string::npos)
            break;
        // starting at the position of the new string, find the next delimiter
        _delimp = line.find_first_of(delimiters, _beginp);
        std::string _tmp{};
        try
        {
            // starting at the position of the new string, get the characters
            // between this position and the next delimiter
            _tmp = line.substr(_beginp, _delimp - _beginp);
        } catch(std::exception& e)
        {
            // print the exception but don't fail, unless maybe it should?
            fprintf(stderr, "%s\n", e.what());
        }
        // don't add empty strings
        if(!_tmp.empty())
        {
            _result.insert(_result.end(), predicate(_tmp));
        }
    }
    return _result;
}

//
//--------------------------------------------------------------------------------------//
///  \brief apply a string transformation to substring inbetween a common delimiter.
///  e.g.
//
template <typename PredicateT = std::function<std::string(const std::string&)>>
inline std::string
str_transform(const std::string& input, const std::string& _begin,
              const std::string& _end, PredicateT&& predicate)
{
    size_t      _beg_pos = 0;  // position that is the beginning of the new string
    size_t      _end_pos = 0;  // position of the delimiter in the string
    std::string _result  = input;
    while(_beg_pos < _result.length() && _end_pos < _result.length())
    {
        // find the first sequence of characters after the end-position
        _beg_pos = _result.find(_begin, _end_pos);

        // if sequence wasn't found, we are done
        if(_beg_pos == std::string::npos)
            break;

        // starting after the position of the first delimiter, find the end sequence
        _end_pos = _result.find(_end, _beg_pos + 1);

        // break if not found
        if(_end_pos == std::string::npos)
            break;

        // length of the substr being operated on
        auto _len = _end_pos - _beg_pos;

        // get the substring between the two delimiters (including first delimiter)
        auto _sub = _result.substr(_beg_pos, _len);

        // apply the transform
        auto _transformed = predicate(_sub);

        // only replace if necessary
        if(_sub != _transformed)
        {
            _result = _result.replace(_beg_pos, _len, _transformed);
            // move end to the end of transformed string
            _end_pos = _beg_pos + _transformed.length();
        }
    }
    return _result;
}

//--------------------------------------------------------------------------------------//
//
inline std::vector<std::string>
read_command_line(pid_t _pid)
{
    std::vector<std::string> _cmdline;
#if defined(_LINUX)
    std::stringstream fcmdline;
    fcmdline << "/proc/" << _pid << "/cmdline";
    std::ifstream ifs(fcmdline.str().c_str());
    if(ifs)
    {
        char        cstr;
        std::string sarg;
        while(!ifs.eof())
        {
            ifs >> cstr;
            if(!ifs.eof())
            {
                if(cstr != '\0')
                {
                    sarg += cstr;
                }
                else
                {
                    _cmdline.push_back(sarg);
                    sarg = "";
                }
            }
        }
        ifs.close();
    }

#else
    consume_parameters(_pid);
#endif
    return _cmdline;
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

    static string_t os()
    {
#if defined(_WINDOWS)
        return "\\";
#elif defined(_UNIX)
        return "/";
#endif
    }

    static string_t inverse()
    {
#if defined(_WINDOWS)
        return "/";
#elif defined(_UNIX)
        return "\\";
#endif
    }

    // OS-dependent representation
    static string_t osrepr(string_t _path)
    {
#if defined(_WINDOWS)
        while(_path.find('/') != std::string::npos)
            _path.replace(_path.find('/'), 1, "\\");
#elif defined(_UNIX)
        while(_path.find("\\\\") != std::string::npos)
            _path.replace(_path.find("\\\\"), 2, "/");
        while(_path.find('\\') != std::string::npos)
            _path.replace(_path.find('\\'), 1, "/");
#endif
        return _path;
    }
};

//--------------------------------------------------------------------------------------//

template <typename T>
TIMEMORY_INLINE size_t
                get_hash(T&& obj)
{
    return std::hash<decay_t<T>>()(std::forward<T>(obj));
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INLINE size_t
                get_hash(const string_view_t& str)
{
    return std::hash<string_view_t>{}(str);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INLINE size_t
                get_hash(const char* cstr)
{
    return std::hash<string_view_t>{}(cstr);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_HOT_INLINE size_t
                    get_hash(char* cstr)
{
    return std::hash<string_view_t>{}(cstr);
}

//--------------------------------------------------------------------------------------//

template <typename T>
struct hasher
{
    inline size_t operator()(T&& val) const { return get_hash(std::forward<T>(val)); }
    inline size_t operator()(const T& val) const { return get_hash(val); }
};

//--------------------------------------------------------------------------------------//
/*
#if defined(_UNIX) &&                                                                    \
    (defined(TIMEMORY_UTILITY_SOURCE) || defined(TIMEMORY_USE_UTILITY_EXTERN))
//
extern template auto
get_backtrace<2, 1>();
extern template auto
get_backtrace<3, 1>();
extern template auto
get_backtrace<4, 1>();
extern template auto
get_backtrace<8, 1>();
extern template auto
get_backtrace<16, 1>();
extern template auto
get_backtrace<32, 1>();
//
extern template auto
get_demangled_backtrace<3, 2>();
extern template auto
get_demangled_backtrace<4, 2>();
extern template auto
get_demangled_backtrace<8, 2>();
extern template auto
get_demangled_backtrace<16, 2>();
extern template auto
get_demangled_backtrace<32, 2>();
//
extern template auto
get_backtrace<3, 2>();
extern template auto
get_backtrace<4, 2>();
extern template auto
get_backtrace<8, 2>();
extern template auto
get_backtrace<16, 2>();
extern template auto
get_backtrace<32, 2>();
//
extern template auto
get_demangled_backtrace<4, 3>();
extern template auto
get_demangled_backtrace<8, 3>();
extern template auto
get_demangled_backtrace<16, 3>();
extern template auto
get_demangled_backtrace<32, 3>();
//
#endif
*/
//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
