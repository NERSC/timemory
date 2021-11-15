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

#ifndef TIMEMORY_UTILITY_UTILITY_CPP_
#define TIMEMORY_UTILITY_UTILITY_CPP_ 1

#include "timemory/macros/os.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/macros.hpp"

#if !defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/utility.hpp"
#endif

namespace tim
{
//
std::string
dirname(std::string _fname)
{
#if defined(TIMEMORY_UNIX)
    char* _cfname = realpath(_fname.c_str(), nullptr);
    _fname        = std::string(_cfname);
    free(_cfname);

    while(_fname.find("\\\\") != std::string::npos)
        _fname.replace(_fname.find("\\\\"), 2, "/");
    while(_fname.find('\\') != std::string::npos)
        _fname.replace(_fname.find('\\'), 1, "/");

    return _fname.substr(0, _fname.find_last_of('/'));
#elif defined(TIMEMORY_WINDOWS)
    while(_fname.find('/') != std::string::npos)
        _fname.replace(_fname.find('/'), 1, "\\");

    _fname = _fname.substr(0, _fname.find_last_of('\\'));
    return (_fname.at(_fname.length() - 1) == '\\')
               ? _fname.substr(0, _fname.length() - 1)
               : _fname;
#endif
}

//--------------------------------------------------------------------------------------//

int
makedir(std::string _dir, int umask)
{
    return filepath::makedir(std::move(_dir), umask);
}

//--------------------------------------------------------------------------------------//

bool
get_bool(const std::string& strbool, bool _default) noexcept
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
            return true;
    }

    return _default;
}

//--------------------------------------------------------------------------------------//

std::vector<std::string>
read_command_line(pid_t _pid)
{
    std::vector<std::string> _cmdline;
#if defined(TIMEMORY_LINUX)
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
//
namespace utility
{
//
path::path(const std::string& _path)
: std::string(osrepr(_path))
{}

path::path(char* _path)
: std::string(osrepr(std::string(_path)))
{}

path::path(const path& rhs)
: std::string(osrepr(rhs))
{}

path::path(const char* _path)
: std::string(osrepr(std::string(const_cast<char*>(_path))))
{}

path&
path::operator=(const std::string& rhs)
{
    std::string::operator=(osrepr(rhs));
    return *this;
}

path&
path::operator=(const path& rhs)
{
    if(this != &rhs)
        std::string::operator=(osrepr(rhs));
    return *this;
}

path&
path::insert(size_type __pos, const std::string& __s)
{
    std::string::operator=(osrepr(std::string::insert(__pos, __s)));
    return *this;
}

path&
path::insert(size_type __pos, const path& __s)
{
    std::string::operator=(osrepr(std::string::insert(__pos, __s)));
    return *this;
}

std::string
path::os()
{
#if defined(TIMEMORY_WINDOWS)
    return "\\";
#elif defined(TIMEMORY_UNIX)
    return "/";
#endif
}

std::string
path::inverse()
{
#if defined(TIMEMORY_WINDOWS)
    return "/";
#elif defined(TIMEMORY_UNIX)
    return "\\";
#endif
}

//
// OS-dependent representation
std::string
path::osrepr(std::string _path)
{
#if defined(TIMEMORY_WINDOWS)
    filepath::replace(_path, '/', "\\");
    filepath::replace(_path, "\\\\", "\\");
#elif defined(TIMEMORY_UNIX)
    filepath::replace(_path, '\\', "/");
    filepath::replace(_path, "//", "/");
#endif
    return _path;
}

// common representation
std::string
path::canonical(std::string _path)
{
    return filepath::canonical(std::move(_path));
}
//
}  // namespace utility
}  // namespace tim
//

#endif  // !defined(TIMEMORY_UTILITY_UTILITY_CPP_)
