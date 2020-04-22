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

/** \file timemory/utility/signals.hpp
 * \headerfile utility/signals.hpp "timemory/utility/signals.hpp"
 * Handles signals emitted by application
 *
 */

//======================================================================================//
/// This global method should be used on LINUX or MacOSX platforms with gcc,
/// clang, or intel compilers for activating signal detection and forcing
/// exception being thrown that can be handled when detected.
//======================================================================================//

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/signals.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#include <cfenv>
#include <csignal>

#if defined(SIGNAL_AVAILABLE)
#    include <dlfcn.h>
#endif

//======================================================================================//

namespace tim
{
// No    Name         Default Action       Description
// 1     SIGHUP       terminate process    terminal line hangup
// 2     SIGINT       terminate process    interrupt program
// 3     SIGQUIT      create core image    quit program
// 4     SIGILL       create core image    illegal instruction
// 5     SIGTRAP      create core image    trace trap
// 6     SIGABRT      create core image    abort program (formerly SIGIOT)
// 7     SIGEMT       create core image    emulate instruction executed
// 8     SIGFPE       create core image    floating-point exception
// 9     SIGKILL      terminate process    kill program
// 10    SIGBUS       create core image    bus error
// 11    SIGSEGV      create core image    segmentation violation
// 12    SIGSYS       create core image    non-existent system call invoked
// 13    SIGPIPE      terminate process    write on a pipe with no reader
// 14    SIGALRM      terminate process    real-time timer expired
// 15    SIGTERM      terminate process    software termination signal
// 16    SIGURG       discard signal       urgent condition present on socket
// 18    SIGTSTP      stop process         stop signal generated from keyboard
// 24    SIGXCPU      terminate process    cpu time limit exceeded (see
// setrlimit(2)) 25    SIGXFSZ      terminate process    file size limit
// exceeded (see setrlimit(2)) 26    SIGVTALRM    terminate process    virtual
// time alarm (see setitimer(2)) 27    SIGPROF      terminate process profiling
// timer alarm (see setitimer(2))

//--------------------------------------------------------------------------------------//

enum class sys_signal : int
{
    Hangup       = SIGHUP,   // 1
    Interrupt    = SIGINT,   // 2
    Quit         = SIGQUIT,  // 3
    Illegal      = SIGILL,
    Trap         = SIGTRAP,
    Abort        = SIGABRT,
    Emulate      = SIGEMT,
    FPE          = SIGFPE,
    Kill         = SIGKILL,
    Bus          = SIGBUS,
    SegFault     = SIGSEGV,
    System       = SIGSYS,
    Pipe         = SIGPIPE,
    Alarm        = SIGALRM,
    Terminate    = SIGTERM,
    Urgent       = SIGURG,
    Stop         = SIGTSTP,
    CPUtime      = SIGXCPU,
    FileSize     = SIGXFSZ,
    VirtualAlarm = SIGVTALRM,
    ProfileAlarm = SIGPROF,
    User1        = SIGUSR1,
    User2        = SIGUSR2
};

//--------------------------------------------------------------------------------------//

class signal_settings
{
public:
    using signal_set_t      = std::set<sys_signal>;
    using signal_function_t = std::function<void(int)>;

public:
    static bool        is_active();
    static void        set_active(bool val);
    static void        enable(const sys_signal&);
    static void        disable(const sys_signal&);
    static std::string str(const sys_signal&);
    static std::string str();
    static void        check_environment();
    static void        set_exit_action(signal_function_t _f);
    static void        exit_action(int errcode);

    static const signal_set_t& enabled();
    static const signal_set_t& disabled();
    static const signal_set_t& get_enabled();
    static const signal_set_t& get_disabled();
    static const signal_set_t& get_default();

protected:
    struct signals_data_t
    {
        signals_data_t();
        bool              signals_active;
        signal_set_t      signals_default;
        signal_set_t      signals_enabled;
        signal_set_t      signals_disabled;
        signal_function_t signals_exit_func;
    };

    static signals_data_t& f_signals()
    {
        static signal_settings::signals_data_t instance;
        return instance;
    }
};

//--------------------------------------------------------------------------------------//

// declarations
inline bool enable_signal_detection(
    signal_settings::signal_set_t = signal_settings::get_default());

//--------------------------------------------------------------------------------------//

inline void
disable_signal_detection();

//--------------------------------------------------------------------------------------//

inline void
update_signal_detection(const signal_settings::signal_set_t& _signals)
{
    if(settings::allow_signal_handler())
    {
        disable_signal_detection();
        enable_signal_detection(_signals);
    }
}

//--------------------------------------------------------------------------------------//
#if defined(SIGNAL_AVAILABLE)
static void
termination_signal_message(int sig, siginfo_t* sinfo, std::ostream& message);
#endif
//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#if defined(SIGNAL_AVAILABLE)

inline std::string
timemory_stack_demangle(const std::string& name)
{
    // PRINT_HERE("%s", "");
    size_t found_end = name.find_first_of("+)", 0, 2);
    if(found_end == std::string::npos)
    {
        found_end = name.size();
    }
    size_t found_parenthesis = name.find_first_of("(");
    size_t start             = found_parenthesis + 1;
    if(found_parenthesis == std::string::npos)
        start = 0;

    // PRINT_HERE("%s", "substr");
    std::string s = name.substr(start, found_end - start);

    if(s.length() != 0)
    {
        int    status        = 0;
        char*  output_buffer = nullptr;
        size_t length        = s.length();
        char*  d = abi::__cxa_demangle(s.c_str(), output_buffer, &length, &status);
        if(status == 0 && d != nullptr)
        {
            s = d;
            free(d);
        }
    }
    // PRINT_HERE("%s", "special-case");
    // Special cases for "main" and "start" on Mac
    if(s.length() == 0)
    {
        if(name == "main" || name == "start")
        {
            s = name;
        }
    }
    // PRINT_HERE("%s", "returning");
    return s;
}
//--------------------------------------------------------------------------------------//

inline void
timemory_stack_backtrace(std::ostream& os)
{
    using size_type = std::string::size_type;
    // PRINT_HERE("%s", "");

    //   from http://linux.die.net/man/3/backtrace_symbols_fd
#    define BSIZE 100
    void* buffer[BSIZE];
    for(size_type j = 0; j < BSIZE; ++j)
        buffer[j] = nullptr;
    size_type nptrs   = backtrace(buffer, BSIZE);
    char**    strings = backtrace_symbols(buffer, nptrs);
    if(strings == NULL)
    {
        perror("backtrace_symbols");
        return;
    }

    std::vector<std::vector<std::string>> dmang_buf;
    std::vector<size_type>                dmang_len;

    // lambda for demangling a string when delimiting
    auto _transform = [](std::string s) { return tim::demangle(s); };

    dmang_buf.resize(nptrs, std::vector<std::string>(0, ""));

    for(size_type j = 0; j < nptrs; ++j)
    {
        std::string _str = const_cast<const char*>(strings[j]);

        auto _delim = tim::delimit(_str, " +;\t\n\r()[]");

        if(_delim.size() > 0)
            _delim[0] = _transform(_delim[0]);

        /*
        if(_delim.size() > 1)
        {
            int _line = 0;
            std::stringstream ss;
            ss << std::hex << _delim[1];
            ss >> _line;
            _delim[1] = std::to_string(_line);
        }

        if(_delim.size() > 2)
        {
            std::string _file = "";
            std::stringstream ss;
            ss << std::hex << _delim[2];
            ss >> _file;
            _delim[2] = _file;
        }
        */

        for(auto& itr : _delim)
            itr = _transform(itr);

        /*
        std::vector<std::string> _dladdr;
        for(const auto& itr : _delim)
        {
            auto idx = itr.find("(+");
            if(idx == std::string::npos)
            {
                _dladdr.push_back(itr);
            }
            else
            {
                auto edx = itr.find_last_of(')');
                if(edx == std::string::npos)
                {
                    _dladdr.push_back(itr);
                }
                else
                {
                    auto _funcn = itr.substr(0, idx);
                    auto _remain = itr.substr(idx+2, edx);
                    while(_remain.find(')') != std::string::npos)
                        _remain.erase(_remain.find(')'), 1);

                    _dladdr.push_back(_funcn);
                    _dladdr.push_back(_remain);

                    // PRINT_HERE("%s", "dlopen");
                    auto _dlopen = dlopen(NULL, RTLD_NOW);
                    if(_dlopen)
                    {
                        // PRINT_HERE("%s", "dlsym");
                        auto _dlsym = dlsym(_dlopen, _remain.c_str());
                        // PRINT_HERE("%s", "dladdr");
                        Dl_info _info;
                        auto _ret = dladdr(_dlsym, &_info);
                        // PRINT_HERE("ret: %i", (int) _ret);
                        if(_ret != 0 && _info.dli_fname != NULL)
                        {
                            // PRINT_HERE("%s", _info.dli_fname);
                            _dladdr.push_back(std::string(_info.dli_fname));
                        }
                    }
                }
            }
        }*/

        // PRINT_HERE("iteration %i - accumulate", (int) j);
        dmang_len.resize(std::max(dmang_len.size(), _delim.size()), 0);

        // accumulate the max lengths of the strings
        for(size_type i = 0; i < _delim.size(); ++i)
            dmang_len[i] = std::max(dmang_len[i], _delim[i].length());

        // add
        dmang_buf[j] = _delim;
    }

    // PRINT_HERE("%s", "");
    free(strings);

    std::stringstream _oss;

    _oss << std::endl << "Call Stack:" << std::endl;
    int nwidth = std::max(2, static_cast<int32_t>(std::log10(nptrs)) + 1);
    for(size_type j = 0; j < nptrs; ++j)
    {
        // print the back-trace numver
        _oss << "[" << std::setw(nwidth) << (nptrs - j - 1) << "/" << std::setw(nwidth)
             << nptrs << "] : ";

        // loop over fields
        for(size_type i = 0; i < dmang_len.size(); ++i)
        {
            std::stringstream _ss;
            // if last param, don't set width
            int mwidth = (i + 1 < dmang_len.size()) ? dmang_len.at(i) : 0;
            _ss << std::setw(mwidth) << std::left
                << ((i < dmang_buf.at(j).size()) ? dmang_buf.at(j).at(i)
                                                 : std::string(" "));
            _oss << tim::demangle(_ss.str()) << "  ";
            // std::cout << _ss.str() << "  ";
        }
        _oss << std::endl;
        // std::cout << std::endl;
    }

    _oss << std::flush;
    os << _oss.str() << std::flush;
    // c++filt can demangle:
    // http://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html
}

//--------------------------------------------------------------------------------------//

static void
timemory_termination_signal_handler(int sig, siginfo_t* sinfo, void* /* context */)
{
    // PRINT_HERE("%s", "");
    tim::sys_signal _sig = (tim::sys_signal)(sig);

    if(tim::signal_settings::get_enabled().find(_sig) ==
       tim::signal_settings::get_enabled().end())
    {
        std::stringstream ss;
        ss << "signal " << sig << " not caught";
        throw std::runtime_error(ss.str());
    }
    {
        std::stringstream message;
        tim::termination_signal_message(sig, sinfo, message);
        std::cerr << message.str() << std::flush;
    }

    tim::disable_signal_detection();

    std::stringstream message;
    message << "\n\n";

#    if defined(PSIGINFO_AVAILABLE)
    if(sinfo)
    {
        psiginfo(sinfo, message.str().c_str());
    }
    else
    {
        std::cerr << message.str() << std::endl;
    }
#    else
    std::cerr << message.str() << std::endl;
#    endif
    // std::raise(sig);
    exit(sig);
}

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

static struct sigaction&
tim_signal_termaction()
{
    static struct sigaction timemory_sigaction_instance_new;
    return timemory_sigaction_instance_new;
}

//--------------------------------------------------------------------------------------//

static struct sigaction&
tim_signal_oldaction()
{
    static struct sigaction timemory_sigaction_instance_old;
    return timemory_sigaction_instance_old;
}

//--------------------------------------------------------------------------------------//

static void
termination_signal_message(int sig, siginfo_t* sinfo, std::ostream& os)
{
    // PRINT_HERE("%s", "");
    std::stringstream message;
    sys_signal        _sig = (sys_signal)(sig);

    message << "\n### ERROR ### ";
    if(dmp::is_initialized())
        message << " [ rank : " << dmp::rank() << " ] ";
    message << "Error code : " << sig;
    if(sinfo)
        message << " @ " << sinfo->si_addr;
    message << " : " << signal_settings::str(_sig);

    if(sig == SIGSEGV)
    {
        if(sinfo)
        {
            switch(sinfo->si_code)
            {
                case SEGV_MAPERR: message << "Address not mapped to object."; break;
                case SEGV_ACCERR:
                    message << "Invalid permissions for mapped object.";
                    break;
                default:
                    message << "Unknown segmentation fault error: " << sinfo->si_code
                            << ".";
                    break;
            }
        }
        else
        {
            message << "Segmentation fault (unknown).";
        }
    }
    else if(sig == SIGFPE)
    {
        if(sinfo)
        {
            switch(sinfo->si_code)
            {
                case FE_DIVBYZERO: message << "Floating point divide by zero."; break;
                case FE_OVERFLOW: message << "Floating point overflow."; break;
                case FE_UNDERFLOW: message << "Floating point underflow."; break;
                case FE_INEXACT: message << "Floating point inexact result."; break;
                case FE_INVALID: message << "Floating point invalid operation."; break;
                default:
                    message << "Unknown floating point exception error: "
                            << sinfo->si_code << ".";
                    break;
            }
        }
        else
        {
            message << "Unknown error.";
        }
    }

    message << std::endl;
    try
    {
        signal_settings::disable(_sig);
        signal_settings::exit_action(sig);
    } catch(std::exception& e)
    {
        std::cerr << "signal_settings::exit_action(" << sig << ") threw an exception"
                  << std::endl;
        std::cerr << e.what() << std::endl;
    }

    timemory_stack_backtrace(message);
    os << message.str() << std::flush;
}

//--------------------------------------------------------------------------------------//

inline bool
enable_signal_detection(signal_settings::signal_set_t operations)
{
    if(!settings::allow_signal_handler())
    {
        if(signal_settings::is_active())
            disable_signal_detection();
        return false;
    }

    // don't re-enable
    if(signal_settings::is_active())
        return false;

    if(operations.empty())
        operations = signal_settings::enabled();
    else
    {
        for(auto& itr : signal_settings::get_enabled())
            signal_settings::disable(itr);
        signal_settings::check_environment();
        for(auto& itr : operations)
            signal_settings::enable(itr);
    }

    std::set<int> _signals;
    for(auto itr = operations.cbegin(); itr != operations.cend(); ++itr)
        _signals.insert(static_cast<int>(*itr));

    sigfillset(&tim_signal_termaction().sa_mask);
    for(auto& itr : _signals)
        sigdelset(&tim_signal_termaction().sa_mask, itr);
    tim_signal_termaction().sa_sigaction = timemory_termination_signal_handler;
    tim_signal_termaction().sa_flags     = SA_SIGINFO;
    for(auto& itr : _signals)
    {
        sigaction(itr, &tim_signal_termaction(), &tim_signal_oldaction());
    }
    signal_settings::set_active(true);

    if(settings::verbose() > 0 || settings::debug())
        std::cout << signal_settings::str() << std::endl;

    return true;
}

//--------------------------------------------------------------------------------------//

inline void
disable_signal_detection()
{
    // don't re-disable
    if(!signal_settings::is_active())
        return;

    sigemptyset(&tim_signal_termaction().sa_mask);
    tim_signal_termaction().sa_handler = SIG_DFL;

    auto _disable = [](signal_settings::signal_set_t _set) {
        for(auto itr = _set.cbegin(); itr != _set.cend(); ++itr)
        {
            int _itr = static_cast<int>(*itr);
            sigaction(_itr, &tim_signal_termaction(), 0);
        }
    };

    _disable(signal_settings::get_enabled());
    _disable(signal_settings::get_disabled());

    signal_settings::set_active(false);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#else /* Not a supported architecture */

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

inline bool enable_signal_detection(signal_settings::signal_set_t) { return false; }

//--------------------------------------------------------------------------------------//

inline void
disable_signal_detection()
{}

//--------------------------------------------------------------------------------------//

inline void
timemory_stack_backtrace(std::ostream& os)
{
    os << "timemory_stack_backtrace() not available." << std::endl;
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#endif

#include "timemory/utility/bits/signals.hpp"
