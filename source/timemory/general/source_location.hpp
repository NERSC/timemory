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

/** \file general/source_location.hpp
 * \headerfile general/source_location.hpp "timemory/general/source_location.hpp"
 * Provides source location information and variadic joining of source location
 * tags
 *
 */

#pragma once

#include "timemory/general/hash.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/settings.hpp"

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
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
    //  if a const char* is passed, assume it is constant
    //
    source_location(const mode& _mode, const char* _func, int _line, const char* _fname,
                    const char* _arg1, const char* _arg2)
    : m_mode(_mode)
    {
        const char* _arg = join_type::join("", _arg1, _arg2).c_str();
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
                auto&& _label = _join(_arg);
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
            case mode::full:
            {
                compute_data(_func, _line, _fname);
                // label and hash
                auto&& _label = _join(_arg);
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

    //----------------------------------------------------------------------------------//
    //
    const captured& get_captured(const char*, const char*) { return m_captured; }

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
