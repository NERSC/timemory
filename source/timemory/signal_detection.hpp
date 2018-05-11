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

/** \file timemory/signal_detection.hpp
 * \headerfile signal_detection.hpp "timemory/signal_detection.hpp"
 * Handles signals emitted by application
 *
 */

//============================================================================//
/// This global method should be used on LINUX or MacOSX platforms with gcc,
/// clang, or intel compilers for activating signal detection and forcing
/// exception being thrown that can be handled when detected.
//============================================================================//

#ifndef signal_detection_hpp_
#define signal_detection_hpp_ 1

#include "timemory/macros.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <iostream>
#include <cstdlib>
#include <stdlib.h>  /* abort(), exit() */
#include <set>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <functional>
#include <deque>
#include <cmath>
#include <vector>

#include "timemory/macros.hpp"
#include "timemory/utility.hpp"
#include "timemory/mpi.hpp"

#if defined(_UNIX)
#   include <cfenv>
#   include <cxxabi.h>
#   include <execinfo.h> // for StackBacktrace()
#endif

// compatible compiler
#if (defined(__GNUC__) || defined(__clang__) || defined(_INTEL_COMPILER))
#   if !defined(SIGNAL_COMPAT_COMPILER)
#       define SIGNAL_COMPAT_COMPILER
#   endif
#endif

// compatible operating system
#if (defined(__linux__) || defined(__MACH__))
#   if !defined(SIGNAL_COMPAT_OS)
#       define SIGNAL_COMPAT_OS
#   endif
#endif

#if defined(SIGNAL_COMPAT_COMPILER) && defined(SIGNAL_COMPAT_OS)
#   if !defined(SIGNAL_AVAILABLE)
#       define SIGNAL_AVAILABLE
#   endif
#endif

#if defined(__linux__)
#   include <features.h>
#   include <csignal>
#elif defined(__MACH__)      /* MacOSX */
#   include <signal.h>
#endif

//============================================================================//
//  these are not in the original POSIX.1-1990 standard so we are defining
//  them in case the OS hasn't
//  POSIX-1.2001
#ifndef SIGTRAP
#   define SIGTRAP 5
#endif
//  not specified in POSIX.1-2001, but nevertheless appears on most other
//  UNIX systems, where its default action is typically to terminate the
//  process with a core dump.
#ifndef SIGEMT
#   define SIGEMT 7
#endif
//  POSIX-1.2001
#ifndef SIGURG
#   define SIGURG 16
#endif
//  POSIX-1.2001
#ifndef SIGXCPU
#   define SIGXCPU 24
#endif
//  POSIX-1.2001
#ifndef SIGXFSZ
#   define SIGXFSZ 25
#endif
//  POSIX-1.2001
#ifndef SIGVTALRM
#   define SIGVTALRM 26
#endif
//  POSIX-1.2001
#ifndef SIGPROF
#   define SIGPROF 27
#endif
//  POSIX-1.2001
#ifndef SIGINFO
#   define SIGINFO 29
#endif

//============================================================================//

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
// 24    SIGXCPU      terminate process    cpu time limit exceeded (see setrlimit(2))
// 25    SIGXFSZ      terminate process    file size limit exceeded (see setrlimit(2))
// 26    SIGVTALRM    terminate process    virtual time alarm (see setitimer(2))
// 27    SIGPROF      terminate process    profiling timer alarm (see setitimer(2))


//----------------------------------------------------------------------------//

enum class sys_signal : int
{
    sHangup = SIGHUP, // 1
    sInterrupt = SIGINT, // 2
    sQuit = SIGQUIT, // 3
    sIllegal = SIGILL,
    sTrap = SIGTRAP,
    sAbort = SIGABRT,
    sEmulate = SIGEMT,
    sFPE = SIGFPE,
    sKill = SIGKILL,
    sBus = SIGBUS,
    sSegFault = SIGSEGV,
    sSystem = SIGSYS,
    sPipe = SIGPIPE,
    sAlarm = SIGALRM,
    sTerminate = SIGTERM,
    sUrgent = SIGURG,
    sStop = SIGTSTP,
    sCPUtime = SIGXCPU,
    sFileSize = SIGXFSZ,
    sVirtualAlarm = SIGVTALRM,
    sProfileAlarm = SIGPROF
};

//----------------------------------------------------------------------------//

class tim_api signal_settings
{
public:
    typedef std::set<sys_signal> signal_set_t;
    //typedef void (*signal_function_t)(int errcode);
    typedef std::function<void(int)> signal_function_t;

public:
    static bool is_active();
    static void set_active(bool val);
    static void enable(const sys_signal&);
    static void disable(const sys_signal&);
    static std::string str(const sys_signal&);
    static std::string str();
    static void check_environment();
    static void set_exit_action(signal_function_t _f);
    static void exit_action(int errcode);

    static const signal_set_t& enabled();
    static const signal_set_t& disabled();
    static const signal_set_t& get_enabled();
    static const signal_set_t& get_disabled();
    static const signal_set_t& get_default();

protected:
    struct signals_data_t
    {
        signals_data_t();
        bool                 signals_active;
        signal_set_t         signals_default;
        signal_set_t         signals_enabled;
        signal_set_t         signals_disabled;
        signal_function_t    signals_exit_func;
    };

    static signals_data_t f_signals;
};

//----------------------------------------------------------------------------//

} // namespace tim

//============================================================================//

#if defined(SIGNAL_AVAILABLE)

//============================================================================//

namespace tim
{

//----------------------------------------------------------------------------//

static struct sigaction tim_signal_termaction, tim_signal_oldaction;

// declarations
inline bool enable_signal_detection(signal_settings::signal_set_t ops
                                    = signal_settings::signal_set_t());
inline void disable_signal_detection();

//----------------------------------------------------------------------------//

inline void stack_backtrace(std::ostream& ss)
{
    typedef std::string::size_type          size_type;

    //   from http://linux.die.net/man/3/backtrace_symbols_fd
#   define BSIZE 50
    void* buffer[ BSIZE ];
    size_type nptrs = backtrace( buffer, BSIZE );
    char** strings = backtrace_symbols( buffer, nptrs );
    if(strings == NULL)
    {
        perror( "backtrace_symbols" );
        return;
    }

    std::deque<std::deque<std::string>>     dmang_buf;
    std::deque<size_type>                   dmang_len;

    // lambda for demangling a string when delimiting
    auto _transform = [] (std::string _str)
    {
        int _ret = 0;
        char* _demang = abi::__cxa_demangle(_str.c_str(), 0, 0, &_ret);
        if(_demang && _ret == 0)
            return std::string(const_cast<const char*>(_demang));
        else
            return _str;
    };

    for(size_type j = 0; j < nptrs; ++j)
    {
        std::string str = strings[j];
        if(str.find("+") != std::string::npos)
            str.replace(str.find_last_of("+"), 1, " +");

        auto _delim = delimit(str, " \t\n\r()");
        // found a GCC compiler bug when passing _transform to delimit
        for(auto& itr : _delim)
            itr = _transform(itr);

        // find trailing " + ([0-9]+)"
        auto itr = _delim.begin();
        for(; itr != _delim.end(); ++itr)
            if(*itr == "+")
                break;

        // get rid of trailing " + ([0-9]+)"
        if(itr != _delim.end())
            _delim.erase(itr, _delim.end());

        // get rid of hex strings if not last param
        for(itr = _delim.begin(); itr != _delim.end(); ++itr)
            if(itr->substr(0, 2) == "0x"  ||
               itr->substr(0, 3) == "[0x" ||
               itr->substr(0, 3) == "+0x" )
            {
                if(itr+1 == _delim.end())
                    continue;
                _delim.erase(itr);
                --itr;
            }

        // accumulate the max lengths of the strings
        for(size_type i = 0; i < _delim.size(); ++i)
        {
            dmang_len.resize(std::max(dmang_len.size(), _delim.size()), 0);
            dmang_len[i] = std::max(dmang_len[i], _delim[i].length());
        }

        // add
        dmang_buf.push_back(_delim);
    }

    free(strings);

    ss << std::endl << "Call Stack:" << std::endl;
    int nwidth = std::max(2, static_cast<int32_t>(std::log10(nptrs))+1);
    for(size_type j = 0; j < nptrs; ++j)
    {
        // print the back-trace numver
        ss << "["<< std::setw(nwidth) << nptrs-j-1 << "/"
           << std::setw(nwidth) << nptrs << "] : ";
        // loop over fields
        for(size_type i = 0; i < dmang_len.size(); ++i)
        {
            std::stringstream _ss;
            // if last param, don't set width
            int mwidth = (i+1 < dmang_len.size()) ? dmang_len.at(i) : 0;
            _ss << std::setw(mwidth) << std::left
                << ((i < dmang_buf.at(j).size())
                    ? dmang_buf.at(j).at(i)
                    : std::string(" "));
            ss << _ss.str() << "  ";
        }
        ss << std::endl;
    }

    // c++filt can demangle:
    // http://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html
}

//----------------------------------------------------------------------------//

inline void termination_signal_message(int sig, siginfo_t* sinfo,
                                       std::ostream& message)
{
    sys_signal _sig = (sys_signal) (sig);

    message << "\n" << "### ERROR ### ";
    if(mpi_is_initialized())
        message << " [ MPI rank : " << mpi_rank() << " ] ";
    message << "Error code : " << sig;
    if(sinfo)
        message << " @ " << sinfo->si_addr;
    message << " : " << signal_settings::str(_sig);

    if(sig == SIGSEGV)
    {
        if(sinfo)
            switch (sinfo->si_code)
            {
                case SEGV_MAPERR:
                    message << "Address not mapped to object.";
                    break;
                case SEGV_ACCERR:
                    message << "Invalid permissions for mapped object.";
                    break;
                default:
                    message << "Unknown segmentation fault error: "
                            << sinfo->si_code << ".";
                    break;
            }
        else
            message << "Segmentation fault (unknown).";
    }
    else if(sig == SIGFPE)
    {
        if(sinfo)
            switch (sinfo->si_code)
            {
                case FE_DIVBYZERO:
                    message << "Floating point divide by zero.";
                    break;
                case FE_OVERFLOW:
                    message << "Floating point overflow.";
                    break;
                case FE_UNDERFLOW:
                    message << "Floating point underflow.";
                    break;
                case FE_INEXACT:
                    message << "Floating point inexact result.";
                    break;
                case FE_INVALID:
                    message << "Floating point invalid operation.";
                    break;
                default:
                    message << "Unknown floating point exception error: "
                            << sinfo->si_code << ".";
                    break;

            }
        else
            message << "Unknown error: " << sinfo->si_code << ".";
    }

    message << std::endl;
    try
    {
        signal_settings::disable(_sig);
        signal_settings::exit_action(sig);
    }
    catch(std::exception& e)
    {
        std::cerr << "signal_settings::exit_action(" << sig
                  << ") threw an exception" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    stack_backtrace(message);
}

//----------------------------------------------------------------------------//

inline void termination_signal_handler(int sig, siginfo_t* sinfo,
                                       void* /* context */)
{
    sys_signal _sig = (sys_signal) (sig);

    if(signal_settings::get_enabled().find(_sig) ==
       signal_settings::get_enabled().end())
        return;

    std::stringstream message;
    termination_signal_message(sig, sinfo, message);

    if(signal_settings::enabled().find(sys_signal::sAbort) !=
       signal_settings::enabled().end())
    {
        signal_settings::disable(sys_signal::sAbort);
        signal_settings::disable(_sig);
        disable_signal_detection();
        enable_signal_detection(signal_settings::enabled());
        std::cerr << message.str() << std::endl;
    }

    // throw an exception instead of ::abort() so it can be caught
    // if the error can be ignored if desired
#if defined(TIMEMORY_EXCEPTIONS)
    throw std::runtime_error(message.str());
#else
    std::cerr << message.str() << std::endl;
    exit(sig);
#endif
}

//----------------------------------------------------------------------------//

inline bool enable_signal_detection(signal_settings::signal_set_t operations)
{
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

    sigfillset(&tim_signal_termaction.sa_mask);
    for(auto& itr : _signals)
        sigdelset(&tim_signal_termaction.sa_mask, itr);
    tim_signal_termaction.sa_sigaction = termination_signal_handler;
    tim_signal_termaction.sa_flags = SA_SIGINFO;
    for(auto& itr : _signals)
        sigaction(itr, &tim_signal_termaction, &tim_signal_oldaction);

    signal_settings::set_active(true);

    if(get_env<int32_t>("TIMEMORY_VERBOSE", 0) > 1)
        std::cout << signal_settings::str() << std::endl;

    return true;
}

//----------------------------------------------------------------------------//

inline void disable_signal_detection()
{
    // don't re-disable
    if(!signal_settings::is_active())
        return;

    sigemptyset(&tim_signal_termaction.sa_mask);
    tim_signal_termaction.sa_handler = SIG_DFL;

    auto _disable = [] (const signal_settings::signal_set_t& _set)
    {
        for(auto itr = _set.cbegin(); itr != _set.cend(); ++itr)
        {
            int _itr = static_cast<int>(*itr);
            sigaction(_itr, &tim_signal_termaction, 0);
        }
    };

    _disable(signal_settings::get_enabled());
    _disable(signal_settings::get_disabled());

    signal_settings::set_active(false);
}

//----------------------------------------------------------------------------//

} // namespace tim

//============================================================================//

#else /* Not a supported architecture */

//============================================================================//

namespace tim
{
inline bool enable_signal_detection(signal_settings::signal_set_t
                                    = signal_settings::signal_set_t())
{ return false; }
inline void disable_signal_detection() { }
inline void stack_backtrace(std::ostream& os)
{
    os << "stack_backtrace() not available." << std::endl;
}

} // namespace tim

//============================================================================//

#endif

#pragma GCC diagnostic pop

#endif /* signal_detection_hpp_ */
