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
#include <cassert>
#include <cctype>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <locale>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
namespace argparse
{
namespace helpers
{
//
//--------------------------------------------------------------------------------------//
//
static inline bool
not_is_space(int ch)
{
    return !std::isspace(ch);
}
//
//--------------------------------------------------------------------------------------//
//
static inline uint64_t
lcount(const std::string& s, bool (*f)(int) = not_is_space)
{
    uint64_t c = 0;
    for(size_t i = 0; i < s.length(); ++i, ++c)
    {
        if(f(s.at(i)))
            break;
    }
    return c;
}
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
static inline char*
strdup(const char* s)
{
    auto slen   = strlen(s);
    auto result = new char[slen];
    if(result)
        memcpy(result, s, slen + 1);
    return result;
}
//
//--------------------------------------------------------------------------------------//
//
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
//
//--------------------------------------------------------------------------------------//
//
static inline bool
is_numeric(const std::string& arg)
{
    auto _nidx = arg.find_first_of("0123456789");
    auto _oidx = arg.find_first_not_of("0123456789.Ee+-*/");

    // must have number somewhere
    if(_nidx == std::string::npos)
        return false;

    // if something other than number or scientific notation
    if(_oidx != std::string::npos)
        return false;

    // numbers + possible scientific notation
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
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
//
//--------------------------------------------------------------------------------------//
//
static inline size_t
find_punct(const std::string& s)
{
    size_t i;
    for(i = 0; i < s.length(); ++i)
    {
        if(std::ispunct(static_cast<int>(s[i])) && s[i] != '-')
        {
            break;
        }
    }
    return i;
}
//
//--------------------------------------------------------------------------------------//
//
namespace is_container_impl
{
//
template <typename T>
struct is_container : std::false_type
{};
//
template <typename... Args>
struct is_container<std::vector<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::set<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::deque<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::list<Args...>> : std::true_type
{};
//
template <typename T>
struct is_initializing_container : is_container<T>::type
{};
//
template <typename... Args>
struct is_initializing_container<std::initializer_list<Args...>> : std::true_type
{};
}  // namespace is_container_impl
//
//--------------------------------------------------------------------------------------//
//
// type trait to utilize the implementation type traits as well as decay the type
template <typename T>
struct is_container
{
    static constexpr bool const value =
        is_container_impl::is_container<decay_t<T>>::value;
};
//
//--------------------------------------------------------------------------------------//
//
// type trait to utilize the implementation type traits as well as decay the type
template <typename T>
struct is_initializing_container
{
    static constexpr bool const value =
        is_container_impl::is_initializing_container<decay_t<T>>::value;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace helpers

//
//--------------------------------------------------------------------------------------//
//
//                      argument vector
//
//--------------------------------------------------------------------------------------//
//
/// \struct argument_vector
/// \brief This class exists to simplify creating argument arrays compatible with execv*
/// routines and MPI_Comm_spawn/MPI_Comm_spawn_multiple
///
struct argument_vector : std::vector<std::string>
{
    struct c_args : std::tuple<int, char**, std::string>
    {
        using base_type = std::tuple<int, char**, std::string>;
        template <typename... Args>
        c_args(Args&&... args)
        : base_type(std::forward<Args>(args)...)
        {}

        auto& argc() { return std::get<0>(*this); }
        auto& argv() { return std::get<1>(*this); }
        auto& args() { return std::get<2>(*this); }

        const auto& argc() const { return std::get<0>(*this); }
        const auto& argv() const { return std::get<1>(*this); }
        const auto& args() const { return std::get<2>(*this); }

        void clear()
        {
            // uses comma operator to execute delete and return nullptr
            for(int i = 0; i < argc(); ++i)
                argv()[i] = (delete[] argv()[i], nullptr);
            argv() = (delete[] argv(), nullptr);
        }
    };

    using base_type = std::vector<std::string>;
    using cargs_t   = c_args;

    template <typename... Args>
    argument_vector(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}

    explicit argument_vector(int& argc, char**& argv)
    : base_type()
    {
        reserve(argc);
        for(int i = 0; i < argc; ++i)
            push_back(argv[i]);
    }

    explicit argument_vector(int& argc, const char**& argv)
    : base_type()
    {
        reserve(argc);
        for(int i = 0; i < argc; ++i)
            push_back(argv[i]);
    }

    explicit argument_vector(int& argc, const char* const*& argv)
    {
        reserve(argc);
        for(int i = 0; i < argc; ++i)
            push_back(argv[i]);
    }

    cargs_t get_execv(const base_type& _prepend, size_t _beg = 0,
                      size_t _end = std::numeric_limits<size_t>::max()) const
    {
        std::stringstream cmdss;
        // find the end if not specified
        _end = std::min<size_t>(size(), _end);
        // determine the number of arguments
        auto _argc = (_end - _beg) + _prepend.size();
        // create the new C argument array, add an extra entry at the end which will
        // always be a null pointer because that is how execv determines the end
        char** _argv = new char*[_argc + 1];

        // ensure all arguments are null pointers initially
        for(size_t i = 0; i < _argc + 1; ++i)
            _argv[i] = nullptr;

        // add the prepend list
        size_t _idx = 0;
        for(const auto& itr : _prepend)
            _argv[_idx++] = helpers::strdup(itr.c_str());

        // copy over the arguments stored internally from the range specified
        for(auto i = _beg; i < _end; ++i)
            _argv[_idx++] = helpers::strdup(this->at(i).c_str());

        // add check that last argument really is a nullptr
        assert(_argv[_argc] == nullptr);

        // create the command string
        for(size_t i = 0; i < _argc; ++i)
            cmdss << " " << _argv[i];
        auto cmd = cmdss.str().substr(1);

        // return a new (int argc, char** argv) and subtract 1 bc nullptr in last entry
        // does not count as argc
        return cargs_t(_argc - 1, _argv, cmd);
    }

    cargs_t get_execv(size_t _beg = 0,
                      size_t _end = std::numeric_limits<size_t>::max()) const
    {
        return get_execv(base_type{}, _beg, _end);
    }

    // helper function to free the memory created by get_execv, pass by reference
    // so that we can set values to nullptr and avoid multiple delete errors
    static void free_execv(cargs_t& itr) { itr.clear(); }
};
//
//--------------------------------------------------------------------------------------//
//
//                      argument parser
//
//--------------------------------------------------------------------------------------//
//
struct argument_parser
{
    struct arg_result;

    using this_type     = argument_parser;
    using result_type   = arg_result;
    using bool_func_t   = std::function<bool(this_type&)>;
    using action_func_t = std::function<void(this_type&)>;
    using action_pair_t = std::pair<bool_func_t, action_func_t>;
    using error_func_t  = std::function<void(this_type&, arg_result&)>;
    using known_args_t  = std::tuple<arg_result, int, char**>;
    using strvec_t      = std::vector<std::string>;
    using strset_t      = std::set<std::string>;
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
            LastArgument   = -1,
            IgnoreArgument = -2
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
            for(auto itr : names)
                m_names.push_back(itr);
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
            if(position != Position::LastArgument)
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

        argument& max_count(int count)
        {
            m_max_count = count;
            return *this;
        }

        argument& min_count(int count)
        {
            m_min_count = count;
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

        template <template <typename...> class ContainerT, typename T, typename... ExtraT,
                  typename ContT = ContainerT<T, ExtraT...>,
                  enable_if_t<helpers::is_container<ContT>::value> = 0>
        argument& choices(const ContainerT<T, ExtraT...>& _choices)
        {
            for(auto&& itr : _choices)
            {
                std::stringstream ss;
                ss << itr;
                m_choices.insert(ss.str());
            }
            return *this;
        }

        template <typename ActionFuncT>
        argument& action(ActionFuncT&& _func)
        {
            m_actions.push_back(std::forward<ActionFuncT>(_func));
            return *this;
        }

        bool found() const { return m_found; }

        template <typename T>
        std::enable_if_t<helpers::is_container<T>::value, T> get()
        {
            T                      t = T{};
            typename T::value_type vt;
            for(auto& s : m_values)
            {
                std::istringstream in(s);
                in >> vt;
                t.insert(t.end(), vt);
            }
            return t;
        }

        template <typename T>
        std::enable_if_t<
            !helpers::is_container<T>::value && !std::is_same<T, bool>::value, T>
        get()
        {
            auto               inp = get<std::string>();
            std::istringstream iss(inp);
            T                  t = T{};
            iss >> t >> std::ws;
            return t;
        }

        template <typename T>
        std::enable_if_t<std::is_same<T, bool>::value, T> get()
        {
            if(m_count == 0)
                return found();

            auto inp = get<std::string>();
            if(inp.empty())
                return found();

            namespace regex_const             = std::regex_constants;
            const auto        regex_constants = regex_const::egrep | regex_const::icase;
            const std::string y_pattern       = "^(on|true|yes|y|t|[1-9]+)$";
            const std::string n_pattern       = "^(off|false|no|n|f|0)$";
            auto is_y = std::regex_match(inp, std::regex(y_pattern, regex_constants));
            auto is_n = std::regex_match(inp, std::regex(n_pattern, regex_constants));
            if(is_y)
                return true;
            if(is_n)
                return false;
            return found();
        }

        size_t size() const { return m_values.size(); }

        std::string get_name() const
        {
            std::stringstream ss;
            for(auto& itr : m_names)
                ss << "/" << itr;
            return ss.str().substr(1);
        }

    private:
        argument(const std::string& name, const std::string& desc, bool required = false)
        : m_desc(desc)
        , m_required(required)
        {
            m_names.push_back(name);
        }

        argument() {}

        arg_result check_choice(const std::string& value)
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
            return arg_result{};
        }

        void execute_actions(argument_parser& p)
        {
            for(auto& itr : m_actions)
                itr(p);
        }

        friend std::ostream& operator<<(std::ostream& os, const argument& arg)
        {
            std::stringstream ss;
            ss << "names: ";
            for(auto itr : arg.m_names)
                ss << itr << " ";
            ss << ", index: " << arg.m_index << ", count: " << arg.m_count
               << ", min count: " << arg.m_min_count << ", max count: " << arg.m_max_count
               << ", found: " << std::boolalpha << arg.m_found
               << ", required: " << std::boolalpha << arg.m_required
               << ", position: " << arg.m_position << ", values: ";
            for(auto itr : arg.m_values)
                ss << itr << " ";
            os << ss.str();
            return os;
        }

        friend struct argument_parser;
        int                        m_position  = Position::IgnoreArgument;
        int                        m_count     = Count::ANY;
        int                        m_min_count = Count::ANY;
        int                        m_max_count = Count::ANY;
        std::vector<std::string>   m_names     = {};
        std::string                m_desc      = {};
        bool                       m_found     = false;
        bool                       m_required  = false;
        int                        m_index     = -1;
        void*                      m_default   = nullptr;
        callback_t                 m_callback  = [](void*) {};
        std::set<std::string>      m_choices   = {};
        std::vector<std::string>   m_values    = {};
        std::vector<action_func_t> m_actions   = {};
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
    argument& add_argument(const std::vector<std::string>& _names,
                           const std::string& desc, bool req = false)
    {
        return add_argument().names(_names).description(desc).required(req);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename BoolFuncT, typename ActionFuncT>
    this_type& add_action(BoolFuncT&& _b, ActionFuncT& _act)
    {
        m_actions.push_back(
            { std::forward<BoolFuncT>(_b), std::forward<ActionFuncT>(_act) });
        return *this;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ActionFuncT>
    this_type& add_action(const std::string& _name, ActionFuncT& _act)
    {
        auto _b = [=](this_type& p) { return p.exists(_name); };
        m_actions.push_back({ _b, std::forward<ActionFuncT>(_act) });
        return *this;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void print_help(const std::string& _extra = "")
    {
        std::cerr << "Usage: " << m_bin;
        if(m_positional_arguments.empty())
        {
            std::cerr << " [" << demangle<this_type>() << " arguments...]"
                      << " " << _extra << std::endl;
        }
        else
        {
            int current = 0;
            for(auto& v : m_positional_arguments)
            {
                if(v.first != argument::Position::LastArgument)
                {
                    for(; current < v.first; ++current)
                        std::cerr << " [" << current << "]";
                    std::cerr
                        << " ["
                        << helpers::ltrim(
                               m_arguments[static_cast<size_t>(v.second)].m_names[0],
                               [](int c) -> bool { return c != static_cast<int>('-'); })
                        << "]";
                }
                else
                {
                    std::cerr
                        << " ... ["
                        << helpers::ltrim(
                               m_arguments[static_cast<size_t>(v.second)].m_names[0],
                               [](int c) -> bool { return c != static_cast<int>('-'); })
                        << "]";
                }
            }
            if(m_positional_arguments.find(argument::Position::LastArgument) ==
               m_positional_arguments.end())
                std::cerr << " [" << demangle<this_type>() << " arguments...]";
            std::cerr << " " << _extra << std::endl;
        }
        std::cerr << "\nOptions:" << std::endl;
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
            std::cerr << "    " << std::setw(m_width) << std::left << ss.str();

            std::cerr << " " << std::setw(m_width) << a.m_desc;
            if(a.m_required)
                std::cerr << " (Required)";
            std::cerr << std::endl;
        }
        std::cerr << '\n';
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int argc, char** argv, const std::string& delim,
    ///                                 int verb)
    /// \param[in,out] argc Number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Array of strings (i.e. command-line)
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief Basic variant of \ref parse_known_args which does not replace argc/argv
    /// and does not provide an array of strings that it processed
    ///
    known_args_t parse_known_args(int argc, char** argv, const std::string& _delim = "--",
                                  int verbose_level = 0)
    {
        strvec_t _args{};
        return parse_known_args(argc, argv, _args, _delim, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse_known_args(int* argc, char*** argv, const std::string& _delim = "--",
                                int verbose_level = 0)
    {
        strvec_t args{};
        return parse_known_args(argc, argv, args, _delim, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int* argc, char*** argv, const std::string& delim,
    ///                                 int verb)
    /// \param[in,out] argc Pointer to number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Pointer to array of strings (i.e. command-line)
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief This variant calls \ref parse_known_args and replaces argc and argv with
    /// the argv[0] + anything after delimiter (if the delimiter is provided). If the
    /// delimiter does not exist, argc and argv are unchanged.
    ///
    arg_result parse_known_args(int* argc, char*** argv, strvec_t& _args,
                                const std::string& _delim = "--", int verbose_level = 0)
    {
        // check for help flag
        auto help_check = [&](int _argc, char** _argv) {
            strset_t help_args = { "-h", "--help", "-?" };
            auto     _help_req = (exists("help") ||
                              (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
            if(_help_req && !exists("help"))
            {
                for(auto hitr : help_args)
                {
                    auto hstr = hitr.substr(hitr.find_first_not_of('-'));
                    auto itr  = m_name_map.find(hstr);
                    if(itr != m_name_map.end())
                        m_arguments[static_cast<size_t>(itr->second)].m_found = true;
                }
            }
            return _help_req;
        };

        // check for a dash in th command line
        bool _pdash = false;
        for(int i = 1; i < *argc; ++i)
        {
            if((*argv)[i] == std::string("--"))
                _pdash = true;
        }

        // parse the known args and get the remaining argc/argv
        auto _pargs = parse_known_args(*argc, *argv, _args, _delim, verbose_level);
        auto _perrc = std::get<0>(_pargs);
        auto _pargc = std::get<1>(_pargs);
        auto _pargv = std::get<2>(_pargs);

        // check if help was requested before the dash (if dash exists)
        if(help_check((_pdash) ? 0 : _pargc, _pargv))
            return arg_result{ "help requested" };

        // assign the argc and argv
        *argc = _pargc;
        *argv = _pargv;

        return _perrc;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int argc, char** argv, strvec_t& args, const
    ///                                 std::string& delim, int verb)
    /// \param[in,out] argc Number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Array of strings (i.e. command-line)
    /// \param[in,out] args Array of strings processed by this parser
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief Parses all options until argv[argc-1] or delimiter is found.
    /// Returns a tuple containing an argument error object (operator bool will return
    /// true if there was an error) and the new argc and argv after the known arguments
    /// have been processed. This is slightly different from the Python
    /// argparse.ArgumentParser.parse_known_args: if the delimiter is not found, it will
    /// not remove the arguments that it recognizes.
    /// To distinguish this parsers options from user arguments, use the syntax:
    ///
    ///    ./<CMD> <PARSER_OPTIONS> -- <USER_ARGS>
    ///
    /// And std::get<1>(...) on the return value will be the new argc.
    /// and std::get<2>(...) on the return value will be the new argv.
    /// Other valid usages:
    ///
    ///    ./<CMD> --help       (will report this parser's help message)
    ///    ./<CMD> -- --help    (will report the applications help message, if supported)
    ///    ./<CMD> <USER_ARGS>
    ///    ./<CMD> <PARSER_OPTIONS>
    ///    ./<CMD> <PARSER_OPTIONS> <USER_ARGS> (intermixed)
    ///
    /// will not remove any of the known options.
    /// In other words, this will remove all arguments after <CMD> until the first "--" if
    /// reached and everything after the "--" will be placed in argv[1:]
    ///
    known_args_t parse_known_args(int argc, char** argv, strvec_t& _args,
                                  const std::string& _delim = "--", int verbose_level = 0)
    {
        int    _cmdc = argc;  // the argc after known args removed
        char** _cmdv = argv;  // the argv after known args removed
        // _cmdv and argv are same pointer unless delimiter is found

        if(argc > 0)
        {
            m_bin = std::string((const char*) argv[0]);
            _args.push_back(std::string((const char*) argv[0]));
        }

        for(int i = 1; i < argc; ++i)
        {
            std::string _arg = argv[i];
            if(_arg == _delim)
            {
                _cmdc        = argc - i;
                _cmdv        = new char*[_cmdc + 1];
                _cmdv[_cmdc] = nullptr;
                _cmdv[0]     = helpers::strdup(argv[0]);
                int k        = 1;
                for(int j = i + 1; j < argc; ++j, ++k)
                    _cmdv[k] = helpers::strdup(argv[j]);
                break;
            }
            else
            {
                _args.push_back(std::string((const char*) argv[i]));
            }
        }

        auto cmd_string = [](int _ac, char** _av) {
            std::stringstream ss;
            for(int i = 0; i < _ac; ++i)
                ss << _av[i] << " ";
            return ss.str();
        };

        if((_cmdc > 0 && verbose_level > 0) || verbose_level > 1)
            std::cerr << "\n";

        if(verbose_level > 1)
        {
            std::cerr << "[original]> " << cmd_string(argc, argv) << std::endl;
            std::cerr << "[cfg-args]> ";
            for(auto& itr : _args)
                std::cerr << itr << " ";
            std::cerr << std::endl;
        }

        if(_cmdc > 0 && verbose_level > 0)
            std::cerr << "[command]>  " << cmd_string(_cmdc, _cmdv) << "\n\n";

        return known_args_t{ parse(_args, verbose_level), _cmdc, _cmdv };
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename... Args>
    arg_result parse_args(Args&&... args)
    {
        return parse(std::forward<Args>(args)...);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse(int argc, char** argv, int verbose_level = 0)
    {
        std::vector<std::string> _args;
        for(int i = 0; i < argc; ++i)
            _args.push_back(std::string((const char*) argv[i]));
        return parse(_args, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse(const std::vector<std::string>& args, int verb)
    /// \param[in] args Array of strings (i.e. command-line arguments)
    /// \param[in] verb Verbosity
    ///
    /// \brief This is the primary function for parsing the command line arguments.
    /// This is where the map of the options is built and the loop over the
    /// arguments is performed.
    arg_result parse(const std::vector<std::string>& _args, int verbose_level = 0)
    {
        if(verbose_level > 0)
        {
            std::cerr << "[argparse::parse]> parsing '";
            for(const auto& itr : _args)
                std::cerr << itr << " ";
            std::cerr << "'" << '\n';
        }

        using argmap_t = std::map<std::string, argument*>;

        argmap_t   m_arg_map = {};
        arg_result err;
        int        argc = _args.size();
        // the set of options which use a single leading dash but are longer than
        // one character, e.g. -LS ...
        std::set<std::string> long_short_opts;
        if(_args.size() > 1)
        {
            auto is_leading_dash = [](int c) -> bool {
                return c != static_cast<int>('-');
            };
            // build name map
            for(auto& a : m_arguments)
            {
                for(auto& n : a.m_names)
                {
                    auto        nleading_dash = helpers::lcount(n, is_leading_dash);
                    std::string name          = helpers::ltrim(n, is_leading_dash);
                    if(m_name_map.find(name) != m_name_map.end())
                        return arg_result("Duplicate of argument name: " + n);
                    m_name_map[name] = a.m_index;
                    m_arg_map[name]  = &a;
                    if(nleading_dash == 1 && name.length() > 1)
                        long_short_opts.insert(name);
                }
                if(a.m_position >= 0 || a.m_position == argument::Position::LastArgument)
                    m_positional_arguments.at(a.m_position) = a.m_index;
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
                   m_positional_arguments.find(argument::Position::LastArgument) !=
                       m_positional_arguments.end())
                {
                    err          = end_argument();
                    arg_result b = err;
                    err = add_value(current_arg, argument::Position::LastArgument);
                    if(b)
                        return b;
                    // return (m_error_func(*this, b), b);
                    if(err)
                        return (m_error_func(*this, err), err);
                    continue;
                }

                // count number of leading dashes
                auto nleading_dash = helpers::lcount(current_arg, is_leading_dash);
                // ignores the case if the arg is just a '-'
                // look for -a (short) or --arg (long) args
                bool is_arg =
                    (nleading_dash > 0 && arg_len > 1 && arg_len != nleading_dash)
                        ? true
                        : false;

                if(is_arg && !helpers::is_numeric(current_arg))
                {
                    err = end_argument();
                    if(err)
                        return (m_error_func(*this, err), err);

                    auto name   = current_arg.substr(nleading_dash);
                    auto islong = (nleading_dash > 1 || long_short_opts.count(name) > 0);
                    err         = begin_argument(name, islong, argv_index);
                    if(err)
                        return (m_error_func(*this, err), err);
                }
                else if(current_arg.length() > 0)
                {
                    // argument value
                    err = add_value(current_arg, argv_index);
                    if(err)
                        return (m_error_func(*this, err), err);
                }
            }
        }

        // return the help
        if(m_help_enabled && exists("help"))
            return arg_result("help requested");

        err = end_argument();
        if(err)
            return (m_error_func(*this, err), err);

        // check requirements
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

        // check all the counts have been satisfied
        for(auto& a : m_arguments)
        {
            if(a.m_found)
            {
                auto cnt_err = check_count(a);
                if(cnt_err)
                    return cnt_err;
            }
        }

        // execute the global actions
        for(auto& itr : m_actions)
        {
            if(itr.first(*this))
                itr.second(*this);
        }

        // execute the argument-specific actions
        for(auto& itr : m_arg_map)
        {
            if(exists(itr.first))
                itr.second->execute_actions(*this);
        }

        return arg_result{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn argument& enable_help()
    /// \brief Add a help command
    argument& enable_help()
    {
        m_help_enabled = true;
        return add_argument()
            .names({ "-h", "-?", "--help" })
            .description("Shows this page")
            .count(0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn bool exists(const std::string& name) const
    /// \brief Returns whether or not an option was found in the arguments. Only
    /// useful after a call to \ref parse or \ref parse_known_args.
    ///
    /// \code[.cpp]
    ///
    /// int main(int argc, char** argv)
    /// {
    ///     argument_parser p{ argv[0] };
    ///     p.add_argument()
    ///         .names({ "-h", "--help"})
    ///         .description("Help message")
    ///         .count(0);
    ///
    ///     auto ec = p.parse(argc, argv);
    ///     if(ec)
    ///     {
    ///         std::cerr << "Error: " << ec << std::endl;
    ///         exit(EXIT_FAILURE);
    ///     }
    ///
    ///     if(p.exists("help"))
    ///     {
    ///         p.print_help();
    ///         exit(EXIT_FAILURE);
    ///     }
    ///
    ///     // ...
    /// }
    /// \endcode
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
    /// \fn T get(const std::string& name)
    /// \tparam T Data type to convert the argument into
    /// \param[in] name An identifier of the option
    ///
    /// \brief Get the value(s) associated with an argument. If option, it should
    /// be used in conjunction with \ref exists(name). Only useful after a call to \ref
    /// parse or \ref parse_known_args.
    ///
    /// \code[.cpp]
    ///
    /// int main(int argc, char** argv)
    /// {
    ///     argument_parser p{ argv[0] };
    ///     p.add_argument()
    ///         .names({ "-n", "--iterations"})
    ///         .description("Number of iterations")
    ///         .count(1);
    ///
    ///     // ... etc.
    ///
    ///     auto nitr = p.get<size_t>("iteration");
    ///
    ///     // ... etc.
    /// }
    /// \endcode
    template <typename T>
    T get(const std::string& name)
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].get<T>();
        return T{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int64_t get_count(const std::string& name)
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].size();
        return 0;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int64_t get_count(argument& a) { return a.m_values.size(); }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ErrorFuncT>
    void on_error(ErrorFuncT&& _func)
    {
        on_error_sfinae(std::forward<ErrorFuncT>(_func), 0);
    }

private:
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename FuncT = std::function<bool(int, int)>>
    arg_result check_count(argument& a, const std::string& _do_str = "111",
                           const FuncT& _func = std::not_equal_to<int>{})
    {
        int _sz  = static_cast<int>(a.m_values.size());
        int _cnt = a.m_count;
        int _max = a.m_max_count;
        int _min = a.m_min_count;

        std::bitset<3> _do{ _do_str };
        std::bitset<3> _checks;
        _checks.reset();  // set all to false
        // if <val> > ANY AND <val does not satisfies condition> -> true
        _checks.set(0, _do.test(0) && _cnt > argument::Count::ANY && _func(_sz, _cnt));
        _checks.set(1, _do.test(1) && _max > argument::Count::ANY && _sz > _max);
        _checks.set(2, _do.test(2) && _min > argument::Count::ANY && _sz < _min);
        // if no checks failed, return non-error
        if(_checks.none())
            return arg_result{};
        // otherwise, compose an error message
        std::stringstream msg;
        msg << "Argument: " << a.get_name() << " failed to satisfy its argument count "
            << "requirements. Number of arguments: " << _sz << ".";
        if(_checks.test(0))
        {
            msg << "\n[" << a.get_name() << "]> Requires exactly " << _cnt << " values.";
        }
        else
        {
            if(_checks.test(1))
                msg << "\n[" << a.get_name() << "]> Requires less than " << _max + 1
                    << " values.";
            if(_checks.test(2))
                msg << "\n[" << a.get_name() << "]> Requires more than " << _min - 1
                    << " values.";
        }
        return arg_result(msg.str());
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename FuncT = std::function<bool(int, int)>>
    arg_result check_count(const std::string& name, const std::string& _do = "111",
                           const FuncT& _func = std::not_equal_to<int>{})
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return check_count(m_arguments[static_cast<size_t>(itr->second)], _do, _func);
        return arg_result{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ErrorFuncT>
    auto on_error_sfinae(ErrorFuncT&& _func, int)
        -> decltype(_func(std::declval<this_type&>(), std::declval<result_type>()),
                    void())
    {
        m_error_func = std::forward<ErrorFuncT>(_func);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ErrorFuncT>
    auto on_error_sfinae(ErrorFuncT&& _func, long)
        -> decltype(_func(std::declval<result_type>()), void())
    {
        auto _wrap_func = [=](this_type&, result_type ret) { _func(ret); };
        m_error_func    = _wrap_func;
    }
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
                arg_name = arg.substr(0, equal_pos);
                nmf      = m_name_map.find(arg_name);
            }
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
        return arg_result{};
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

            err = a.check_choice(value);
            if(err)
                return err;

            auto num_values = [&]() { return static_cast<int>(a.m_values.size()); };

            // check {m_count, m_max_count} > COUNT::ANY && m_values.size() >= {value}
            if((a.m_count >= 0 && num_values() >= a.m_count) ||
               (a.m_max_count >= 0 && num_values() >= a.m_max_count))
            {
                err = end_argument();
                if(err)
                    return err;
                unnamed();
                return arg_result{};
            }

            a.m_values.push_back(value);

            // check {m_count, m_max_count} > COUNT::ANY && m_values.size() >= {value}
            if((a.m_count >= 0 && num_values() >= a.m_count) ||
               (a.m_max_count >= 0 && num_values() >= a.m_max_count))
            {
                err = end_argument();
                if(err)
                    return err;
            }
            return arg_result{};
        }
        else
        {
            unnamed();
            // TODO
            return arg_result{};
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
            if(a.m_max_count >= 0)
            {
                if(static_cast<int>(a.m_values.size()) > a.m_max_count)
                    return arg_result("Too many arguments given for " + a.m_names[0]);
            }
            else if(a.m_count >= 0)
            {
                if(static_cast<int>(a.m_values.size()) > a.m_count)
                    return arg_result("Too many arguments given for " + a.m_names[0]);
            }
        }
        return arg_result{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
protected:
    bool                       m_help_enabled         = false;
    int                        m_current              = -1;
    int                        m_width                = 30;
    std::string                m_desc                 = {};
    std::string                m_bin                  = {};
    error_func_t               m_error_func           = [](this_type&, result_type) {};
    std::vector<argument>      m_arguments            = {};
    std::map<int, int>         m_positional_arguments = {};
    std::map<std::string, int> m_name_map             = {};
    std::vector<action_pair_t> m_actions              = {};
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
