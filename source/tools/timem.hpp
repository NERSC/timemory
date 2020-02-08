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

#define TIMEM_DEBUG
#define TIMEMORY_DISABLE_BANNER

#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//
//  configure to record statistics
//
namespace trait
{
template <>
struct record_statistics<component::page_rss> : std::true_type
{
    using type = double;
};
/*
template <>
struct record_statistics<component::virtual_memory> : std::true_type
{};
template <>
struct record_statistics<component::read_bytes> : std::true_type
{};
template <>
struct record_statistics<component::written_bytes> : std::true_type
{};
*/
}  // namespace trait
}  // namespace tim

#include "timemory/timemory.hpp"

// C includes
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_USE_LIBEXPLAIN)
#    include <libexplain/execvp.h>
#endif

#if defined(_UNIX)
#    include <unistd.h>
extern "C"
{
    extern char** environ;
}
#endif

// C++ includes
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

//--------------------------------------------------------------------------------------//

#if defined(__GNUC__) || defined(__clang__)
#    define declare_attribute(attr) __attribute__((attr))
#elif defined(_WIN32)
#    define declare_attribute(attr) __declspec(attr)
#endif

template <typename _Tp>
using vector_t = std::vector<_Tp>;
using string_t = std::string;

template <typename _Tp>
using vector_t = std::vector<_Tp>;
using string_t = std::string;

using namespace tim::component;

//--------------------------------------------------------------------------------------//

#define TIMEM_SIGNAL SIGALRM
#define TIMEM_ITIMER ITIMER_REAL

//--------------------------------------------------------------------------------------//
// create a custom component tuple printer
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
struct custom_print
{
    using value_type = typename _Tp::value_type;
    using base_type  = component::base<_Tp, value_type>;

    custom_print(std::size_t _N, std::size_t /*_Ntot*/, base_type& obj, std::ostream& os,
                 bool /*endline*/)
    {
        std::stringstream ss;
        if(_N == 0)
            ss << std::endl;
        ss << "    " << obj << std::endl;
        os << ss.str();
    }
};

//--------------------------------------------------------------------------------------//

namespace operation
{
template <typename _Tp, bool _Sample = ::tim::trait::file_sampler<_Tp>::value>
struct file_sample;

template <typename _Tp>
struct file_sample<_Tp, true>
{
    explicit file_sample(_Tp& obj) { obj.measure(); }
};

template <typename _Tp>
struct file_sample<_Tp, false>
{
    explicit file_sample(_Tp&) {}
};
}  // namespace operation

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
class custom_component_tuple : public component_tuple<Types...>
{
public:
    using apply_print_t = modifiers<custom_print, Types...>;
    using base_type     = component_tuple<Types...>;

    static std::string label();

public:
    explicit custom_component_tuple(const string_t& key)
    : base_type(key, true, true)
    , printed(false)
    {}

    ~custom_component_tuple()
    {
        if(!printed)
        {
            component_tuple<Types...>::stop();
            std::cerr << *this << std::endl;
        }
    }

    using base_type::get;
    using base_type::get_labeled;
    using base_type::m_data;
    using base_type::record;
    using base_type::reset;
    using base_type::start;
    using base_type::stop;
    using apply_v   = typename base_type::apply_v;
    using impl_type = typename base_type::impl_type;

    template <typename... _Types>
    struct opsample;
    template <template <typename...> class _TypeL, typename... _Types>
    struct opsample<_TypeL<_Types...>>
    {
        using type =
            _TypeL<operation::file_sample<_Types, trait::file_sampler<_Types>::value>...>;
    };

    void sample()
    {
        using apply_sample_t = typename opsample<impl_type>::type;
        apply<void>::access<apply_sample_t>(this->m_data);
    }

    //----------------------------------------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&                           os,
                                    const custom_component_tuple<Types...>& obj)
    {
        obj.printed = true;
        std::stringstream ssp;
        std::stringstream ssd;
        auto&&            data  = obj.m_data;
        auto&&            key   = obj.key();
        auto&&            width = obj.output_width();

        apply<void>::access_with_indices<apply_print_t>(data, std::ref(ssd), false);

        ssp << std::setw(width) << std::left << key;
        os << ssp.str() << ssd.str();

        return os;
    }

    mutable bool printed = false;
};

}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//
using comp_tuple_t = tim::custom_component_tuple<
    real_clock, user_clock, system_clock, cpu_clock, cpu_util, peak_rss, page_rss,
    virtual_memory, num_minor_page_faults, num_major_page_faults, num_signals,
    voluntary_context_switch, priority_context_switch, read_bytes, written_bytes>;

//--------------------------------------------------------------------------------------//

inline comp_tuple_t*&
get_measure()
{
    static comp_tuple_t* _instance = nullptr;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool
use_shell()
{
    static bool _instance = tim::get_env("TIMEM_USE_SHELL", true);
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool
debug()
{
    static bool _instance = tim::get_env("TIMEM_DEBUG", false);
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline int
verbose()
{
    static int _instance = tim::get_env("TIMEM_VERBOSE", 0);
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline std::string&
command()
{
    static std::string _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::string
tim::custom_component_tuple<Types...>::label()
{
    return command();
}

//--------------------------------------------------------------------------------------//

inline pid_t&
master_pid()
{
    static pid_t _instance = getpid();
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline pid_t&
worker_pid()
{
    static pid_t _instance = getpid();
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline struct sigaction&
timem_signal_action()
{
    static struct sigaction _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline struct itimerval&
timem_itimer()
{
    static struct itimerval _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline void
sampler(int signum)
{
    if(signum == TIMEM_SIGNAL)
    {
        get_measure()->sample();
        if((debug() && verbose() > 1) || (verbose() > 2))
            std::cerr << "[SAMPLE][" << getpid() << "]> " << *get_measure() << std::endl;
        else if(debug())
            fprintf(stderr, "[%i]> sampling...\n", getpid());
    }
    else
    {
        perror("timem sampler caught signal that was not TIMEM_SIGNAL...");
        signal(signum, SIG_DFL);
        raise(signum);
    }
}

//--------------------------------------------------------------------------------------//

inline void
sampler(int signum, siginfo_t*, void*)
{
    if(signum == TIMEM_SIGNAL)
    {
        get_measure()->sample();
        if((debug() && verbose() > 1) || (verbose() > 2))
            std::cerr << "[SAMPLE][" << getpid() << "]> " << *get_measure() << std::endl;
        else if(debug())
            fprintf(stderr, "[%i]> sampling...\n", getpid());
    }
    else
    {
        perror("timem sampler caught signal that was not TIMEM_SIGNAL...");
        signal(signum, SIG_DFL);
        raise(signum);
    }
}

//--------------------------------------------------------------------------------------//

inline void
explain(int ret, const char* pathname, char** argv)
{
    if(ret < 0)
    {
#if defined(TIMEMORY_USE_LIBEXPLAIN)
        fprintf(stderr, "%s\n", explain_execvp(pathname, argv));
#else
        fprintf(stderr, "Return code: %i : %s\n", ret, pathname);
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
#endif
    }
    else if(debug() || verbose() > 0)
    {
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

inline int
diagnose_status(int status, bool log_msg = true)
{
    if(verbose() > 2 || debug())
        fprintf(stderr, "[%i]> program (PID: %i) diagnosing status %i...\n",
                (int) master_pid(), (int) worker_pid(), status);

    if(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS)
    {
        if(verbose() > 2 || (debug() && verbose() > 0))
            fprintf(stderr, "[%i]> program (PID: %i) terminated normally with %i\n",
                    (int) master_pid(), (int) worker_pid(), WEXITSTATUS(status));
        return 0;
    }

    int ret = WEXITSTATUS(status);

    if(WIFSTOPPED(status))
    {
        if(log_msg)
        {
            int sig = WSTOPSIG(status);
            fprintf(stderr, "[%i]> program (PID: %i) stopped with signal %i: %i\n",
                    (int) master_pid(), (int) worker_pid(), sig, ret);
        }
    }
    else if(WCOREDUMP(status))
    {
        if(log_msg)
            fprintf(stderr,
                    "[%i]> program (PID: %i)  terminated and produced a core dump: %i\n",
                    (int) master_pid(), (int) worker_pid(), ret);
    }
    else if(WIFSIGNALED(status))
    {
        if(log_msg)
            fprintf(stderr,
                    "[%i]> program (PID: %i)  terminated because it received a signal "
                    "(%i) that was not handled: %i\n",
                    (int) master_pid(), (int) worker_pid(), WTERMSIG(status), ret);
        ret = WTERMSIG(status);
    }
    else if(WIFEXITED(status) && WEXITSTATUS(status))
    {
        if(log_msg)
        {
            if(ret == 127)
                fprintf(stderr, "[%i]> execv failed\n", (int) master_pid());
            else
                fprintf(
                    stderr,
                    "[%i]> program (PID: %i)  terminated with a non-zero status: %i\n",
                    (int) master_pid(), (int) worker_pid(), ret);
        }
    }
    else
    {
        if(log_msg)
            fprintf(stderr, "[%i]> program (PID: %i)  terminated abnormally.\n",
                    (int) master_pid(), (int) worker_pid());
        ret = -1;
    }

    return ret;
}

//--------------------------------------------------------------------------------------//

inline int
waitpid_eintr(int& status)
{
    pid_t pid    = 0;
    int   errval = 0;
    while((pid = waitpid(WAIT_ANY, &status, 0)) == -1)
    {
        errval = errno;
        if(errno != errval)
            perror("Unexpected error in waitpid_eitr");

        int ret = diagnose_status(status, debug());

        if(debug())
            fprintf(stderr, "[%i]> return code: %i\n", pid, ret);

        if(errval == EINTR)
            continue;
        break;
    }
    return errval;
}

//--------------------------------------------------------------------------------------//
