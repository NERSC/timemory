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

#include "timemory/bits/ctimemory.h"

#include "timemory/mpl/apply.hpp"

#include <ostream>
#include <sstream>
#include <string>

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

struct source_location
{
    enum class mode : short
    {
        blank = 0,
        basic = 1,
        full  = 2
    };

    template <typename... _Args>
    source_location(const mode& _mode, const char* _fname, const char* _func, int _line,
                    _Args&&... args)
    : m_mode(_mode)
    , m_line(_line)
    , m_filename(std::move(_fname))
    , m_function(std::move(_func))
    , m_data(apply<std::string>::join("", std::forward<_Args>(args)...).c_str())
    {
    }

    source_location()                       = delete;
    ~source_location()                      = default;
    source_location(const source_location&) = delete;
    source_location(source_location&&)      = default;
    source_location& operator=(const source_location&) = delete;
    source_location& operator=(source_location&&) = default;

    const char* filename() const { return m_filename; }
    const char* function() const { return m_function; }
    int         line() const { return m_line; }

    const char* get() const
    {
        std::stringstream ss;
        ss << *this;
        return std::move(ss.str().c_str());
    }

    friend std::ostream& operator<<(std::ostream& os, const source_location& obj)
    {
        auto filename_substr = [&]() {
#if defined(_WINDOWS)
            const char delim = '\\';
#else
            const char delim = '/';
#endif
            std::string _filename = obj.filename();
            if(_filename.find(delim) != std::string::npos)
                return _filename.substr(_filename.find_last_of(delim) + 1).c_str();
            return obj.m_filename;
        };

        switch(obj.m_mode)
        {
            case mode::blank: os << obj.m_data; break;
            case mode::basic: os << obj.function() << obj.m_data; break;
            case mode::full:
                os << obj.function() << obj.m_data << "@'" << filename_substr()
                   << "':" << obj.line();
                break;
        }
        return os;
    }

private:
    mode        m_mode;
    int         m_line;
    const char* m_filename;
    const char* m_function;
    const char* m_data;
};

}  // namespace tim

#define TIMEMORY_SOURCE_LOCATION(MODE, ...)                                              \
    ::tim::source_location(MODE, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
