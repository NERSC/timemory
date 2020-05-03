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

#include "timem.hpp"
#include "timemory/utility/argparse.hpp"

//--------------------------------------------------------------------------------------//

void
parent_process(pid_t pid, int ret)
{
    if((debug() && verbose() > 1) || verbose() > 2)
        std::cerr << "[AFTER STOP][" << pid << "]> " << *get_measure() << std::endl;

    std::stringstream _oss;
    _oss << "\n" << *get_measure() << std::flush;

    if(tim::settings::file_output())
    {
        std::string label = "timem";
        if(tim::settings::text_output())
        {
            auto          fname = tim::settings::compose_output_filename(label, ".txt");
            std::ofstream ofs(fname.c_str());
            if(ofs)
            {
                fprintf(stderr, "[timem]> Outputting '%s'...\n", fname.c_str());
                ofs << _oss.str();
                ofs.close();
            }
            else
            {
                std::cerr << "[timem]>  opening output file '" << fname << "'...\n";
                std::cerr << _oss.str();
            }
        }

        if(tim::settings::json_output())
        {
            auto jname = tim::settings::compose_output_filename(label, ".json");
            fprintf(stderr, "[timem]> Outputting '%s'...\n", jname.c_str());
            tim::generic_serialization(jname, *get_measure());
        }
    }
    else
    {
        std::cerr << _oss.str() << std::endl;
    }

    exit(ret);
}

//--------------------------------------------------------------------------------------//

declare_attribute(noreturn) void child_process(uint64_t argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    std::stringstream shell_cmd;
    char**            argv_list = (char**) malloc(sizeof(char**) * (argc));
    for(uint64_t i = 0; i < argc - 1; i++)
    {
        argv_list[i] = strdup(argv[i + 1]);
        shell_cmd << argv[i + 1] << " ";
    }
    argv_list[argc - 1] = nullptr;

    // launches the command with the shell, this is the default because it enables aliases
    auto launch_using_shell = [&]() {
        std::string _shell          = get_config().shell;
        auto        flags_str       = get_config().shell_flags;
        auto        flags_vec       = tim::delimit(flags_str, " ");
        int         ret             = -1;
        uint64_t    argc_shell      = 4 + flags_vec.size();
        char**      argv_shell_list = new char*[argc_shell + 1];

        if(debug() || verbose() > 0)
            fprintf(stderr, "using shell: %s\n", _shell.c_str());

        argv_shell_list[argc_shell - 1] = nullptr;
        if(_shell.length() > 0)
        {
            if(debug())
                PRINT_HERE("%s", "");

            std::string _command   = "-c";
            size_t      idx        = 0;
            argv_shell_list[idx++] = strdup(_shell.c_str());
            for(const auto& itr : flags_vec)
                argv_shell_list[idx++] = strdup(itr.c_str());
            argv_shell_list[idx++] = strdup(_command.c_str());
            argv_shell_list[idx++] = strdup(shell_cmd.str().c_str());
            argv_shell_list[idx++] = nullptr;

            if(debug())
                PRINT_HERE("%s", "");

            explain(0, argv_shell_list[0], argv_shell_list);
            ret = execvp(argv_shell_list[0], argv_shell_list);
            // ret = execv(argv_shell_list[0], argv_shell_list);
            explain(ret, argv_shell_list[0], argv_shell_list);

            if(ret != 0)
            {
                PRINT_HERE("return code: %i", ret);
                explain(ret, argv_shell_list[0], argv_shell_list);
                ret = execv(argv_shell_list[0], argv_shell_list);
            }

            if(ret != 0)
            {
                PRINT_HERE("return code: %i", ret);
                explain(ret, argv_shell_list[0], argv_shell_list);
                ret = execve(argv_shell_list[0], argv_shell_list, environ);
            }

            if(debug())
                PRINT_HERE("return code: %i", ret);
        }
        else
        {
            fprintf(stderr, "getusershell failed!\n");
        }

        if(debug())
            PRINT_HERE("%s", "");

        return ret;
    };

    // this will launch the process and inherit the environment but aliases will not
    // be available
    auto launch_without_shell = [&]() {
        int ret = execvp(argv_list[0], argv_list);
        // explain error if enabled
        explain(ret, argv_list[0], argv_list);
        return ret;
    };

    // default return code
    int ret = -1;

    // determine if the shell should be tested first
    bool try_shell = use_shell();

    if(try_shell)
    {
        // launch the command with shell. If that fails, launch without shell
        ret = launch_using_shell();
        if(ret < 0)
        {
            if(debug())
                puts("Error launching with shell! Trying without shell...");
            ret = launch_without_shell();
        }
    }
    else
    {
        // launch the command without shell. If that fails, launch with shell
        ret = launch_without_shell();
        if(ret < 0)
        {
            if(debug())
                puts("Error launching without shell! Trying with shell...");
            ret = launch_using_shell();
        }
    }

    explain(ret, argv_list[0], argv_list);

    exit(ret);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    // disable banner if not specified
    // setenv("TIMEMORY_BANNER", "OFF", 0);
    master_pid() = getpid();

    // set some defaults
    tim::settings::banner()      = false;
    tim::settings::file_output() = false;
    tim::settings::scientific()  = false;
    tim::settings::width()       = 12;
    tim::settings::precision()   = 6;

    tim::argparse::argument_parser parser(argv[0]);

    parser.enable_help();
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
        .max_count(1);
    parser.add_argument().names({ "--debug" }).description("Debug output").count(0);
    parser
        .add_argument({ "-s", "--shell" }, "Enable launching command via a shell command "
                                           "(if no arguments, $SHELL is used)")
        .max_count(1);
    parser
        .add_argument({ "--shell-flags" },
                      "Set the shell flags to use (pass as single "
                      "string as leading dashes can confuse parser) [default: -i]")
        .count(1);
    parser
        .add_argument({ "-d", "--sample-delay" },
                      "Set the delay before the sampler starts (seconds)")
        .count(1);
    parser
        .add_argument({ "-f", "--sample-freq" },
                      "Set the frequency of the sampler (1/seconds)")
        .count(1);

    auto _args = parser.parse_known_args(argc, argv);
    auto _aerr = std::get<0>(_args);
    auto _argc = std::get<1>(_args);
    auto _argv = std::get<2>(_args);

    if(_aerr)
    {
        std::cerr << _aerr << std::endl;
        parser.print_help("-- <CMD> <ARGS>");
        return EXIT_FAILURE;
    }

    if(parser.exists("help"))
    {
        parser.print_help("-- <CMD> <ARGS>");
        return EXIT_FAILURE;
    }

    // make sure config is instantiated
    auto& _config = get_config();

    if(parser.exists("verbose"))
    {
        if(parser.get_count("verbose") == 0)
            verbose() = 1;
        else
            verbose() = parser.get<int>("verbose");
    }

    if(parser.exists("debug"))
        debug() = true;

    if(parser.exists("shell"))
    {
        use_shell() = true;
        if(parser.get_count("shell") > 0)
            _config.shell = parser.get<std::string>("shell");
    }

    if(parser.exists("sample-delay"))
        _config.sample_delay = parser.get<double>("sample-delay");

    if(parser.exists("sample-freq"))
        _config.sample_freq = parser.get<double>("sample-freq");

    if(parser.exists("shell-flags"))
        _config.shell_flags = parser.get<std::string>("shell-flags");

    // parse for settings configurations
    if(argc > 1)
        tim::timemory_init(argc, argv);

    // override a some settings
    tim::settings::suppress_parsing() = true;
    tim::settings::auto_output()      = false;
    tim::settings::output_prefix()    = "";

    // tim::manager::get_storage<comp_tuple_t>::initialize();
    tim::manager::instance()->set_write_metadata(-1);

    auto compose_prefix = [&]() {
        std::stringstream ss;
        ss << "[" << command().c_str() << "] measurement totals:";
        return ss.str();
    };

    if(argc > 1)
    {
        command() = std::string(const_cast<const char*>(_argv[1]));
    }
    else
    {
        command()              = std::string(const_cast<const char*>(_argv[0]));
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        exit(EXIT_SUCCESS);
    }

    if(tim::settings::papi_events().empty())
        tim::settings::papi_events() = "PAPI_TOT_CYC,PAPI_TOT_INS";

    tim::get_rusage_type() = RUSAGE_CHILDREN;
    pid_t pid              = fork();

    if(pid != 0)
    {
        worker_pid()                  = pid;
        tim::process::get_target_id() = pid;
        tim::settings::papi_attach()  = true;
        get_sampler()                 = new sampler_t(compose_prefix(), { TIMEM_SIGNAL });
    }

    uint64_t nargs = static_cast<uint64_t>(_argc);
    if(pid == -1)  // pid == -1 means error occured
    {
        puts("failure forking, error occured!");
        exit(EXIT_FAILURE);
    }
    else if(pid == 0)  // pid == 0 means child process created
    {
        child_process(nargs, _argv);
    }
    else  // means parent process
    {
        /// \param TIMEM_SAMPLE_DELAY
        /// \brief Environment variable, expressed in seconds, that sets the length
        /// of time the timem executable waits before starting sampling of the relevant
        /// measurements (components that read from child process status files)
        ///
        double fdelay = get_config().sample_delay;

        /// \param TIMEM_SAMPLE_FREQ
        /// \brief Environment variable, expressed in 1/seconds, that sets the frequency
        /// that the timem executable samples the relevant measurements (components
        /// that read from child process status files)
        ///
        double frate = get_config().sample_freq;

        sampler_t::set_delay(fdelay);
        sampler_t::set_rate(frate);

        sampler_t::configure(TIMEM_SIGNAL);

        // pause until first interrupt delivered
        pause();

        auto status = sampler_t::wait(verbose(), debug());

        if((debug() && verbose() > 1) || verbose() > 2)
            std::cerr << "[BEFORE STOP][" << pid << "]> " << *get_measure() << std::endl;

        get_sampler()->stop();
        sampler_t::ignore();

        parent_process(pid, status);
    }

    delete get_sampler();
}

//--------------------------------------------------------------------------------------//

// signal(TIMEM_SIGNAL, SIG_IGN);
// signal(SIGTERM, SIG_IGN);
/*
struct sigaction sa;
sigset_t mask;

sa.sa_handler = &dummy_sampler; // Intercept and ignore TIMEM_SIGNAL
sa.sa_sigaction = &dummy_sampler;
sa.sa_flags = SA_RESTART; // Remove the handler after first signal
sigfillset(&sa.sa_mask);
sigaction(TIMEM_SIGNAL, &sa, NULL);
sigaction(SIGTERM, &sa, NULL);

// Get the current signal mask
// sigprocmask(0, NULL, &mask);

// Unblock TIMEM_SIGNAL
// sigdelset(&mask, TIMEM_SIGNAL);
*/

/*
// struct sigaction& sa = timem_signal_action();
struct sigaction sa;

// Install timer_handler as the signal handler for TIMEM_SIGNAL.
memset(&sa, 0, sizeof(sa));
sa.sa_handler = &dummy_sampler;
sa.sa_sigaction = &sampler;
// sa.sa_flags = SA_RESTART;
sigaction(TIMEM_SIGNAL, &sa, nullptr);

// itimerval& _timer = timem_itimer();
struct itimerval _timer;

// Configure the timer to expire after 500 msec...
_timer.it_value.tv_sec  = 2;
_timer.it_value.tv_usec = 50000;

// ... and every 500 msec after that
_timer.it_interval.tv_sec  = 0;
_timer.it_interval.tv_usec = 50000;

int ret = setitimer(ITIMER_REAL, &_timer, nullptr);
fprintf(stderr, "Sample configuration return value: %i\n", ret);
*/
