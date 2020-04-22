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

#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
namespace argparse
{
template <typename... Args>
static inline void
consume_parameters(Args&&...)
{}

namespace helpers
{
static inline bool
not_is_space(int ch)
{
    return !std::isspace(ch);
}

static inline std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}
static inline std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}
static inline std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}

template <typename InputIt>
static inline std::string
join(InputIt begin, InputIt end, const std::string& separator = " ")
{
    std::ostringstream ss;
    if(begin != end)
    {
        ss << *begin++;
    }
    while(begin != end)
    {
        ss << separator;
        ss << *begin++;
    }
    return ss.str();
}

static inline bool
is_numeric(const std::string& arg)
{
    return (arg.find_first_not_of("0123456789.e+-*/") == std::string::npos);
    /*std::stringstream ss;
    ss << arg;
    float              f;
    ss >> std::noskipws >> f;
    return ss.eof() && !ss.fail();*/
}

static inline int
find_equiv(const std::string& s)
{
    for(size_t i = 0; i < s.length(); ++i)
    {
        // if find graph symbol before equal, end search
        // i.e. don't accept --asd)f=0 arguments
        // but allow --asd_f and --asd-f arguments
        if(std::ispunct(static_cast<int>(s[i])))
        {
            if(s[i] == '=')
            {
                return static_cast<int>(i);
            }
            else if(s[i] == '_' || s[i] == '-')
            {
                continue;
            }
            return -1;
        }
    }
    return -1;
}

static inline size_t
find_punct(const std::string& s)
{
    size_t i;
    for(i = 0; i < s.length(); ++i)
    {
        if(std::ispunct(static_cast<int>(s[i])))
        {
            break;
        }
    }
    return i;
}

namespace is_vector_impl
{
template <typename T>
struct is_vector : std::false_type
{};
template <typename... Args>
struct is_vector<std::vector<Args...>> : std::true_type
{};
}  // namespace is_vector_impl

// type trait to utilize the implementation type traits as well as decay the
// type
template <typename T>
struct is_vector
{
    static constexpr bool const value =
        is_vector_impl::is_vector<typename std::decay<T>::type>::value;
};
}  // namespace helpers

//
//--------------------------------------------------------------------------------------//
//
//                      argument parser
//
//--------------------------------------------------------------------------------------//
//

struct argument_parser
{
    //
    //----------------------------------------------------------------------------------//
    //
    struct arg_result
    {
        arg_result() = default;
        arg_result(std::string err) noexcept
        : m_error(true)
        , m_what(err)
        {}

        operator bool() const { return m_error; }

        friend std::ostream& operator<<(std::ostream& os, const arg_result& dt);

        const std::string& what() const { return m_what; }

    private:
        bool        m_error = false;
        std::string m_what  = {};
    };
    //
    //----------------------------------------------------------------------------------//
    //
    struct argument
    {
        using callback_t = std::function<void(void*)>;

        enum Position : int
        {
            LAST   = -1,
            IGNORE = -2
        };

        enum Count : int
        {
            ANY = -1
        };

        argument& name(const std::string& name)
        {
            m_names.push_back(name);
            return *this;
        }

        argument& names(std::vector<std::string> names)
        {
            m_names.insert(m_names.end(), names.begin(), names.end());
            return *this;
        }

        argument& description(const std::string& description)
        {
            m_desc = description;
            return *this;
        }

        argument& required(bool req)
        {
            m_required = req;
            return *this;
        }

        argument& position(int position)
        {
            if(position != Position::LAST)
            {
                // position + 1 because technically argument zero is the name of the
                // executable
                m_position = position + 1;
            }
            else
            {
                m_position = position;
            }
            return *this;
        }

        argument& count(int count)
        {
            m_count = count;
            return *this;
        }

        template <typename T>
        argument& set_default(const T& val)
        {
            m_callback = [&](void* obj) { obj = (void*) new T(val); };
            return *this;
        }

        template <typename T>
        argument& set_default(T& val)
        {
            m_callback = [&](void* obj) { obj = (void*) &val; };
            return *this;
        }

        template <typename T>
        argument& choices(const std::initializer_list<T>& _choices)
        {
            for(auto&& itr : _choices)
            {
                std::stringstream ss;
                ss << itr;
                m_choices.insert(ss.str());
            }
            return *this;
        }

        bool found() const { return m_found; }

        template <typename T>
        std::enable_if_t<helpers::is_vector<T>::value, T> get()
        {
            T                      t = T{};
            typename T::value_type vt;
            for(auto& s : m_values)
            {
                std::istringstream in(s);
                in >> vt;
                t.push_back(vt);
            }
            return t;
        }

        template <typename T>
        std::enable_if_t<!helpers::is_vector<T>::value, T> get()
        {
            auto               inp = get<std::string>();
            std::istringstream iss(inp);
            T                  t = T{};
            iss >> t >> std::ws;
            return t;
        }

    private:
        argument(const std::string& name, const std::string& desc, bool required = false)
        : m_desc(desc)
        , m_required(required)
        {
            m_names.push_back(name);
        }

        argument() {}

        arg_result check(const std::string& value)
        {
            if(m_choices.size() > 0)
            {
                if(m_choices.find(value) == m_choices.end())
                {
                    std::stringstream ss;
                    ss << "Invalid choice: '" << value << "'. Valid choices: ";
                    for(const auto& itr : m_choices)
                        ss << "'" << itr << "' ";
                    return arg_result(ss.str());
                }
            }
            return arg_result();
        }

        friend struct argument_parser;
        int                      m_position = Position::IGNORE;
        int                      m_count    = Count::ANY;
        std::vector<std::string> m_names    = {};
        std::string              m_desc     = {};
        bool                     m_found    = false;
        bool                     m_required = false;
        int                      m_index    = -1;
        void*                    m_default  = nullptr;
        callback_t               m_callback = [](void*) {};
        std::set<std::string>    m_choices  = {};

        std::vector<std::string> m_values{};
    };
    //
    //----------------------------------------------------------------------------------//
    //
    argument_parser(const std::string& desc)
    : m_desc(desc)
    {}
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument()
    {
        m_arguments.push_back({});
        m_arguments.back().m_index = static_cast<int>(m_arguments.size()) - 1;
        return m_arguments.back();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument(const std::initializer_list<std::string>& _names,
                           const std::string& desc, bool req = false)
    {
        return add_argument().names(_names).description(desc).required(req);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void print_help(const std::string& _extra = "")
    {
        std::cout << "Usage: " << m_bin;
        if(m_positional_arguments.empty())
        {
            std::cout << " [options...]"
                      << " " << _extra << std::endl;
        }
        else
        {
            int current = 0;
            for(auto& v : m_positional_arguments)
            {
                if(v.first != argument::Position::LAST)
                {
                    for(; current < v.first; ++current)
                        std::cout << " [" << current << "]";
                    std::cout
                        << " ["
                        << helpers::ltrim(
                               m_arguments[static_cast<size_t>(v.second)].m_names[0],
                               [](int c) -> bool { return c != static_cast<int>('-'); })
                        << "]";
                }
                else
                {
                    std::cout
                        << " ... ["
                        << helpers::ltrim(
                               m_arguments[static_cast<size_t>(v.second)].m_names[0],
                               [](int c) -> bool { return c != static_cast<int>('-'); })
                        << "]";
                }
            }
            if(m_positional_arguments.find(argument::Position::LAST) ==
               m_positional_arguments.end())
                std::cout << " [options...]";
            std::cout << " " << _extra << std::endl;
        }
        std::cout << "\nOptions:" << std::endl;
        for(auto& a : m_arguments)
        {
            std::string name = a.m_names[0];
            for(size_t n = 1; n < a.m_names.size(); ++n)
                name.append(", " + a.m_names[n]);
            std::stringstream ss;
            ss << name;
            if(a.m_choices.size() > 0)
            {
                ss << " [";
                auto itr = a.m_choices.begin();
                ss << " " << *itr++;
                for(; itr != a.m_choices.end(); ++itr)
                    ss << " | " << *itr;
                ss << " ] ";
            }
            std::cout << "    " << std::setw(m_width) << std::left << ss.str();

            std::cout << " " << std::setw(m_width) << a.m_desc;
            if(a.m_required)
                std::cout << " (Required)";
            std::cout << std::endl;
        }
        std::cout << '\n';
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse(int argc, char** argv)
    {
        std::vector<std::string> _args;
        for(int i = 0; i < argc; ++i)
            _args.push_back(std::string((const char*) argv[i]));
        return parse(_args);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse(const std::vector<std::string>& _args)
    {
        arg_result err;
        int        argc = _args.size();
        if(_args.size() > 1)
        {
            // build name map
            for(auto& a : m_arguments)
            {
                for(auto& n : a.m_names)
                {
                    std::string name = helpers::ltrim(
                        n, [](int c) -> bool { return c != static_cast<int>('-'); });
                    if(m_name_map.find(name) != m_name_map.end())
                        return arg_result("Duplicate of argument name: " + n);
                    m_name_map[name] = a.m_index;
                }
                if(a.m_position >= 0 || a.m_position == argument::Position::LAST)
                    m_positional_arguments[a.m_position] = a.m_index;
            }

            m_bin = _args.at(0);

            // parse
            std::string current_arg;
            size_t      arg_len;
            for(int argv_index = 1; argv_index < argc; ++argv_index)
            {
                current_arg = _args.at(argv_index);
                arg_len     = current_arg.length();
                if(arg_len == 0)
                    continue;
                if(argv_index == argc - 1 &&
                   m_positional_arguments.find(argument::Position::LAST) !=
                       m_positional_arguments.end())
                {
                    err          = end_argument();
                    arg_result b = err;
                    err          = add_value(current_arg, argument::Position::LAST);
                    if(b)
                        return b;
                    if(err)
                        return err;
                    continue;
                }
                if(arg_len >= 2 && !helpers::is_numeric(current_arg))
                {
                    // ignores the case if the arg is just a -
                    // look for -a (short) or --arg (long) args
                    if(current_arg[0] == '-')
                    {
                        err = end_argument();
                        if(err)
                            return err;
                        // look for --arg (long) args
                        if(current_arg[1] == '-')
                        {
                            err = begin_argument(current_arg.substr(2), true, argv_index);
                            if(err)
                                return err;
                        }
                        else
                        {  // short args
                            err =
                                begin_argument(current_arg.substr(1), false, argv_index);
                            if(err)
                                return err;
                        }
                    }
                    else
                    {
                        // argument value
                        err = add_value(current_arg, argv_index);
                        if(err)
                            return err;
                    }
                }
                else
                {
                    // argument value
                    err = add_value(current_arg, argv_index);
                    if(err)
                        return err;
                }
            }
        }
        if(m_help_enabled && exists("help"))
        {
            return arg_result("");
        }
        err = end_argument();
        if(err)
            return err;
        for(auto& a : m_arguments)
        {
            if(a.m_required && !a.m_found)
            {
                return arg_result("Required argument not found: " + a.m_names[0]);
            }
            if(a.m_position >= 0 && argc >= a.m_position && !a.m_found)
            {
                return arg_result("argument " + a.m_names[0] + " expected in position " +
                                  std::to_string(a.m_position));
            }
        }
        return arg_result();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void enable_help()
    {
        add_argument().names({ "-h", "--help" }).description("Shows this page");
        m_help_enabled = true;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    bool exists(const std::string& name) const
    {
        std::string n = helpers::ltrim(
            name, [](int c) -> bool { return c != static_cast<int>('-'); });
        auto itr = m_name_map.find(n);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].m_found;
        return false;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename T>
    T get(const std::string& name)
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].get<T>();
        return T{};
    }

private:
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result begin_argument(const std::string& arg, bool longarg, int position)
    {
        auto it = m_positional_arguments.find(position);
        if(it != m_positional_arguments.end())
        {
            arg_result err = end_argument();
            argument&  a   = m_arguments[static_cast<size_t>(it->second)];
            a.m_values.push_back(arg);
            a.m_found = true;
            return err;
        }
        if(m_current != -1)
        {
            return arg_result("Current argument left open");
        }
        size_t      name_end = helpers::find_punct(arg);
        std::string arg_name = arg.substr(0, name_end);
        if(longarg)
        {
            int  equal_pos = helpers::find_equiv(arg);
            auto nmf       = m_name_map.find(arg_name);
            if(nmf == m_name_map.end())
            {
                return arg_result("Unrecognized command line option '" + arg_name + "'");
            }
            m_current                                             = nmf->second;
            m_arguments[static_cast<size_t>(nmf->second)].m_found = true;
            if(equal_pos == 0 || (equal_pos < 0 && arg_name.length() < arg.length()))
            {
                // malformed argument
                return arg_result("Malformed argument: " + arg);
            }
            else if(equal_pos > 0)
            {
                std::string arg_value = arg.substr(name_end + 1);
                add_value(arg_value, position);
            }
        }
        else
        {
            arg_result r;
            if(arg_name.length() == 1)
            {
                return begin_argument(arg, true, position);
            }
            else
            {
                for(char& c : arg_name)
                {
                    r = begin_argument(std::string(1, c), true, position);
                    if(r)
                    {
                        return r;
                    }
                    r = end_argument();
                    if(r)
                    {
                        return r;
                    }
                }
            }
        }
        return arg_result();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result add_value(const std::string& value, int location)
    {
        auto unnamed = [&]() {
            auto itr = m_positional_arguments.find(location);
            if(itr != m_positional_arguments.end())
            {
                argument& a = m_arguments[static_cast<size_t>(itr->second)];
                a.m_values.push_back(value);
                a.m_found = true;
            }
        };

        if(m_current >= 0)
        {
            arg_result err;
            size_t     c = static_cast<size_t>(m_current);
            consume_parameters(c);
            argument& a = m_arguments[static_cast<size_t>(m_current)];

            err = a.check(value);
            if(err)
                return err;

            if(a.m_count >= 0 && static_cast<int>(a.m_values.size()) >= a.m_count)
            {
                err = end_argument();
                if(err)
                    return err;
                unnamed();
            }

            a.m_values.push_back(value);
            if(a.m_count >= 0 && static_cast<int>(a.m_values.size()) >= a.m_count)
            {
                err = end_argument();
                if(err)
                    return err;
            }
            return arg_result();
        }
        else
        {
            unnamed();
            // TODO
            return arg_result();
        }
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result end_argument()
    {
        if(m_current >= 0)
        {
            argument& a = m_arguments[static_cast<size_t>(m_current)];
            m_current   = -1;
            if(static_cast<int>(a.m_values.size()) < a.m_count)
                return arg_result("Too few arguments given for " + a.m_names[0]);
            if(a.m_count >= 0)
            {
                if(static_cast<int>(a.m_values.size()) > a.m_count)
                    return arg_result("Too many arguments given for " + a.m_names[0]);
            }
        }
        return arg_result();
    }

    bool                       m_help_enabled         = false;
    int                        m_current              = -1;
    int                        m_width                = 30;
    std::string                m_desc                 = {};
    std::string                m_bin                  = {};
    std::vector<argument>      m_arguments            = {};
    std::map<int, int>         m_positional_arguments = {};
    std::map<std::string, int> m_name_map             = {};
};
//
//--------------------------------------------------------------------------------------//
//
inline std::ostream&
operator<<(std::ostream& os, const argument_parser::arg_result& r)
{
    os << r.what();
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <>
inline std::string
argument_parser::argument::get<std::string>()
{
    return helpers::join(m_values.begin(), m_values.end());
}
//
//--------------------------------------------------------------------------------------//
//
template <>
inline std::vector<std::string>
argument_parser::argument::get<std::vector<std::string>>()
{
    return m_values;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace argparse
}  // namespace tim
