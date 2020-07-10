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
childpid_catcher(int);
void
parent_process(pid_t pid);
void
      child_process(int argc, char** argv) declare_attribute(noreturn);
pid_t read_pid(pid_t);
static bool&
timem_mpi_was_finalized();

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    // disable banner if not specified
    // setenv("TIMEMORY_BANNER", "OFF", 0);
    master_pid() = getpid();

    // parallel settings
    tim::settings::mpi_init()       = false;
    tim::settings::mpi_finalize()   = false;
    tim::settings::upcxx_init()     = false;
    tim::settings::upcxx_finalize() = false;
    // other settings
    tim::settings::banner()      = false;
    tim::settings::auto_output() = false;
    tim::settings::file_output() = false;
    tim::settings::scientific()  = false;
    tim::settings::width()       = 16;
    tim::settings::precision()   = 6;
    tim::manager::instance()->set_write_metadata(-1);

    auto _mpi_argc = 1;
    auto _mpi_argv = argv;
    tim::mpi::initialize(_mpi_argc, _mpi_argv);

    using parser_t     = tim::argparse::argument_parser;
    using parser_err_t = typename parser_t::result_type;

    auto help_check = [](parser_t& p, int _argc, char** _argv) {
        std::set<std::string> help_args = { "-h", "--help", "-?" };
        return (p.exists("help") || _argc == 1 ||
                (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
    };

    auto help_action = [](parser_t& p) {
        if(tim::dmp::rank() == 0)
            p.print_help("-- <CMD> <ARGS>");
        exit(EXIT_FAILURE);
    };

    auto parser = parser_t(argv[0]);

    parser.enable_help();
    parser.on_error([=](parser_t& p, parser_err_t _err) {
        std::cerr << _err << std::endl;
        help_action(p);
    });

    parser.add_argument()
        .names({ "--debug" })
        .description("Debug output")
        .count(0)
        .action([](parser_t&) { debug() = true; });
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
        .max_count(1)
        .action([](parser_t& p) {
            if(p.get_count("verbose") == 0)
                verbose() = 1;
            else
                verbose() = p.get<int>("verbose");
        });
    parser
        .add_argument({ "-d", "--sample-delay" },
                      "Set the delay before the sampler starts (seconds)")
        .count(1)
        .action([](parser_t& p) { sample_delay() = p.get<double>("sample-delay"); });
    parser
        .add_argument({ "-f", "--sample-freq" },
                      "Set the frequency of the sampler (1/seconds)")
        .count(1)
        .action([](parser_t& p) { sample_freq() = p.get<double>("sample-freq"); });
    parser.add_argument({ "-e", "--events", "--papi-events" },
                        "Set the hardware counter events to record");
    parser
        .add_argument(
            { "--mpi" },
            "Launch processes via MPI_Comm_spawn_multiple (reduced functionality)")
        .count(0);
    parser.add_argument({ "--disable-papi" }, "Disable hardware counters")
        .count(0)
        .action([](parser_t&) { use_papi() = false; });
    parser.add_argument({ "--disable-mpi" }, "Disable MPI_Finalize")
        .count(0)
        .action([](parser_t&) { timem_mpi_was_finalized() = true; });
    parser
        .add_argument({ "-i", "--indiv" },
                      "Output individual results for each process (i.e. rank) instead of "
                      "reporting the aggregation")
        .count(0)
        .action([](parser_t&) { tim::settings::collapse_processes() = false; });
    parser
        .add_argument({ "-s", "--shell" }, "Enable launching command via a shell command "
                                           "(if no arguments, $SHELL is used)")
        .max_count(1)
        .action([](parser_t& p) {
            use_shell() = true;
            if(p.get_count("shell") > 0)
                shell() = p.get<std::string>("shell");
        });
    parser
        .add_argument({ "--shell-flags" },
                      "Set the shell flags to use (pass as single "
                      "string as leading dashes can confuse parser) [default: -i]")
        .count(1)
        .action([](parser_t& p) { shell_flags() = p.get<std::string>("shell-flags"); });

    auto _args = parser.parse_known_args(argc, argv);
    auto _argc = std::get<1>(_args);
    auto _argv = std::get<2>(_args);

    if(help_check(parser, _argc, _argv))
        help_action(parser);

    if(!debug() && verbose() < 1)
        tim::settings::papi_quiet() = true;

    // make sure config is instantiated
    tim::consume_parameters(get_config());

    if(parser.exists("mpi"))
    {
#if !defined(TIMEMORY_USE_MPI)
        throw std::runtime_error("Error! timemory was not built with MPI support");
#else
        use_mpi() = true;
#endif
    }

    if(parser.exists("events"))
    {
        if(!tim::trait::is_available<papi_array_t>::value)
            throw std::runtime_error("Error! timemory was not built with PAPI support");

        auto              evts = parser.get<std::vector<std::string>>("events");
        std::stringstream ss;
        for(const auto& itr : evts)
            ss << itr << ",";
        tim::settings::papi_events() = ss.str().substr(0, ss.str().length() - 1);
    }

    // parse for settings configurations
    if(argc > 1)
        tim::timemory_init(argc, argv);

    // override a some settings
    tim::settings::suppress_parsing() = true;
    tim::settings::auto_output()      = false;
    tim::settings::output_prefix()    = "";

    auto compose_prefix = [&]() {
        std::stringstream ss;
        ss << "[" << command().c_str() << "] measurement totals";
        if(use_mpi())
            ss << " (# ranks = " << tim::mpi::size() << "):";
        else if(tim::dmp::size() > 1)
            ss << " (# ranks = " << tim::dmp::size() << "):";
        else
            ss << ":";
        return ss.str();
    };

    if(argc > 1)
    {
        command() = std::string(const_cast<const char*>(_argv[0]));
    }
    else
    {
        command()              = std::string(const_cast<const char*>(_argv[0]));
        tim::get_rusage_type() = RUSAGE_CHILDREN;
        exit(EXIT_SUCCESS);
    }

    if(tim::settings::papi_events().empty())
    {
        if(use_papi())
            tim::settings::papi_events() = "PAPI_TOT_CYC,PAPI_TOT_INS";
        else
            tim::trait::runtime_enabled<papi_array_t>::set(false);
    }

    tim::get_rusage_type() = RUSAGE_CHILDREN;

    // set the signal handler on this process if using mpi so that we can read
    // the file providing the PID. If not, fork provides the PID so this is
    // unnecessary
    if(use_mpi())
        signal(TIMEM_PID_SIGNAL, childpid_catcher);
    else
        signal_delivered() = true;

    //----------------------------------------------------------------------------------//
    //
    //          Create subprocesses
    //
    //----------------------------------------------------------------------------------//
    pid_t pid = (use_mpi()) ? master_pid() : fork();

    using comm_t        = tim::mpi::comm_t;
    comm_t comm_world_v = tim::mpi::comm_world_v;
    comm_t comm_child_v;

    if(use_mpi())
    {
        tim::trait::apply<tim::trait::runtime_enabled>::set<
            user_clock, system_clock, cpu_clock, cpu_util, peak_rss,
            num_major_page_faults, num_minor_page_faults, priority_context_switch,
            voluntary_context_switch, user_mode_time, kernel_mode_time, papi_array_t>(
            false);

        using info_t      = tim::mpi::info_t;
        using argvector_t = tim::argparse::argument_vector;
        using cargs_t     = typename argvector_t::cargs_t;

        string_t pidexe = argv[0];
        // name of the executable should be some path + "timem"
        if(pidexe.substr(pidexe.find_last_of('/') + 1) == "timem")
        {
            // append "timem" + "ory-pid" to form "timemory-pid"
            pidexe += "ory-pid";
        }
        else
        {
            if(verbose() > -1)
                fprintf(
                    stderr,
                    "Warning! Executable '%s' was expected to be 'timem'. Using "
                    "'timemory-pid' instead of adding 'ory-pid' to name of executable",
                    argv[0]);
            // otherwise, assume it can find 'timemory-pid'
            pidexe = "timemory-pid";
        }

        auto comm_size  = tim::mpi::size();
        auto comm_rank  = tim::mpi::rank();
        auto pids       = std::vector<int>(comm_size, 0);
        auto procs      = std::vector<int>(comm_size, 1);
        auto errcodes   = std::vector<int>(comm_size, 0);
        auto infos      = std::vector<info_t>(comm_size, tim::mpi::info_null_v);
        auto margv      = std::vector<argvector_t>(comm_size, argvector_t(_argc, _argv));
        auto cargv      = std::vector<cargs_t>{};
        auto cargv_arg0 = std::vector<char*>{};
        auto cargv_argv = std::vector<char**>{};

        pids.at(comm_rank) = pid;
        tim::mpi::gather(&pid, 1, MPI_INT, pids.data(), 1, MPI_INT, 0,
                         tim::mpi::comm_world_v);

        if(debug() && comm_rank == 0)
        {
            std::stringstream ss;
            std::cout << "[timem]> parent pids: ";
            for(const auto& itr : pids)
                ss << ", " << itr;
            std::cout << ss.str().substr(2) << '\n';
        };

        // create the
        for(decltype(comm_size) i = 0; i < comm_size; ++i)
        {
            auto _cargv =
                margv.at(i).get_execv({ pidexe, std::to_string(pids.at(i)) }, 1);
            cargv.push_back(_cargv);
            cargv_arg0.push_back(_cargv.argv()[0]);
            cargv_argv.push_back(_cargv.argv() + 1);
            if(debug() && comm_rank == 0)
                fprintf(stderr, "[timem][rank=%i]> cmd :: %s\n", (int) i,
                        _cargv.args().c_str());
        }

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::mpi::barrier(comm_world_v);

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::mpi::comm_spawn_multiple(comm_size, cargv_arg0.data(), cargv_argv.data(),
                                      procs.data(), infos.data(), 0, comm_world_v,
                                      &comm_child_v, errcodes.data());
        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");

        // clean up
        for(auto& itr : cargv)
            itr.clear();
        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
    }
    //
    //----------------------------------------------------------------------------------//
    CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");

    if(pid != 0)
    {
        if(use_mpi())
        {
            CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
            if(!signal_delivered())
                // wait for timemory-pid to signal
                pause();
            worker_pid() = read_pid(master_pid());
        }
        else
        {
            CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
            worker_pid() = pid;
        }

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::process::get_target_id() = worker_pid();
        tim::settings::papi_attach()  = true;
        get_sampler()                 = new sampler_t(compose_prefix(), { TIMEM_SIGNAL });
    }

    auto failed_fork = [&]() {
        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        // pid == -1 means error occured
        bool cond = (pid == -1);
        if(pid == -1)
            puts("failure forking, error occured!");
        return cond;
    };

    auto is_child = [&]() {
        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        // pid == 0 means child process created if not using MPI
        bool cond = (pid == 0 && !use_mpi());
        return cond;
    };

    int ec = 0;

    if(failed_fork())
    {
        exit(EXIT_FAILURE);
    }
    else if(is_child())
    {
        child_process(_argc, _argv);
    }
    else
    {
        // means parent process
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

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        // pause until first interrupt delivered
        pause();

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "target pid = %i",
                               (int) worker_pid());
        auto status = sampler_t::wait(worker_pid(), verbose(), debug());

        if((debug() && verbose() > 1) || verbose() > 2)
            std::cerr << "[BEFORE STOP][" << pid << "]> " << *get_measure() << std::endl;

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        get_sampler()->stop();

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        sampler_t::ignore();

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        tim::mpi::barrier(comm_world_v);

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");
        parent_process(pid);

        CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "exit code = %i", status);
        ec = status;
    }

    delete get_sampler();

    CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "Completed");
    if(use_mpi() || (!timem_mpi_was_finalized() && tim::dmp::size() == 1))
        tim::mpi::finalize();

    return ec;
}

//--------------------------------------------------------------------------------------//

pid_t
read_pid(pid_t _master)
{
    pid_t _worker = 0;
    // get the child pid from a file
    {
        std::stringstream fname;
        fname << tim::get_env<std::string>("TMPDIR", "/tmp") << "/.timemory-pid-"
              << _master;
        std::ifstream ifs(fname.str().c_str());
        if(ifs)
            ifs >> _worker;
        else
            fprintf(stderr, "Error opening '%s'...\n", fname.str().c_str());
        ifs.close();
    }
    if(debug())
        printf("[%s]> pid :: %i -> %i\n", __FUNCTION__, _master, _worker);
    return _worker;
}

//--------------------------------------------------------------------------------------//

void
childpid_catcher(int)
{
    signal_delivered() = true;
    signal(TIMEM_PID_SIGNAL, SIG_DFL);
    int _worker                   = read_pid(master_pid());
    worker_pid()                  = _worker;
    tim::process::get_target_id() = _worker;
    if(debug())
        printf("[%s][pid=%i]> worker_pid() = %i, worker = %i, target_id() = %i\n",
               __FUNCTION__, getpid(), worker_pid(), _worker,
               tim::process::get_target_id());
}

//--------------------------------------------------------------------------------------//

void
parent_process(pid_t pid)
{
    // auto comm_size = tim::mpi::size();
    // auto comm_rank = tim::mpi::rank();

    if((debug() && verbose() > 1) || verbose() > 2)
        std::cerr << "[AFTER STOP][" << pid << "]> " << *get_measure() << std::endl;

    std::vector<timem_bundle_t> _measurements;

    if(use_mpi() || tim::mpi::size() > 1)
        _measurements = get_measure()->mpi_get();
    else
        _measurements = { *get_measure() };

    if(_measurements.empty())
        return;

    std::stringstream _oss;
    for(size_t i = 0; i < _measurements.size(); ++i)
    {
        auto& itr = _measurements.at(i);
        if(itr.empty())
            continue;
        if(_measurements.size() > 0 && (use_mpi() || tim::mpi::size() > 1))
            itr.set_rank(i);
        _oss << "\n" << itr << std::flush;
    }

    if(_oss.str().empty())
        return;

    auto _file_out = tim::settings::file_output();
    auto _text_out = tim::settings::text_output();
    auto _json_out = tim::settings::json_output();

    if(_file_out)
    {
        std::string label = "timem";
        if(_text_out)
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

        if(_json_out)
        {
            auto jname = tim::settings::compose_output_filename(label, ".json");
            fprintf(stderr, "[timem]> Outputting '%s'...\n", jname.c_str());
            tim::generic_serialization(jname, _measurements);
        }
    }
    else
    {
        std::cerr << _oss.str() << std::endl;
    }

    std::cerr << std::flush;

    // tim::mpi::barrier();
    // tim::mpi::finalize();
}

//--------------------------------------------------------------------------------------//

void
child_process(int argc, char** argv)
{
    if(argc < 2)
        exit(0);

    // the argv list first argument should point to filename associated
    // with file being executed the array pointer must be terminated by
    // NULL pointer

    // launches the command with the shell, this is the default because it enables aliases
    auto launch_using_shell = [&]() {
        int ret = -1;
        if(get_config().shell.length() > 0)
        {
            if(debug() || verbose() > 0)
                fprintf(stderr, "using shell: %s\n", get_config().shell.c_str());

            auto argpv = tim::argparse::argument_vector(argc, argv);
            auto argpc = argpv.get_execv(1);
            argpv.clear();
            argpv.resize(1, argpc.args());
            argpc.clear();
            std::vector<std::string> shellp = { get_config().shell, "-c" };
            for(auto&& itr : tim::delimit(get_config().shell_flags, " "))
                shellp.push_back(itr);
            auto shellc = argpv.get_execv(shellp);

            if(debug())
                CONDITIONAL_PRINT_HERE((debug() && verbose() > 1),
                                       "[timem]> command :: %s", shellc.args().c_str());

            explain(0, shellc.argv()[0], shellc.argv());

            ret = execvp(shellc.argv()[0], shellc.argv());
            explain(ret, shellc.argv()[0], shellc.argv());

            if(ret != 0)
            {
                CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "return code: %i",
                                       ret);
                explain(ret, shellc.argv()[0], shellc.argv());
                ret = execv(shellc.argv()[0], shellc.argv());
            }

            if(ret != 0)
            {
                CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "return code: %i",
                                       ret);
                explain(ret, shellc.argv()[0], shellc.argv());
                ret = execve(shellc.argv()[0], shellc.argv(), environ);
            }

            if(debug())
                CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "return code: %i",
                                       ret);
        }
        else
        {
            fprintf(stderr, "getusershell failed!\n");
        }

        if(debug())
            CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "%s", "");

        return ret;
    };

    // this will launch the process and inherit the environment but aliases will not
    // be available
    auto launch_without_shell = [&]() {
        auto argpv = tim::argparse::argument_vector(argc, argv);
        auto argpc = argpv.get_execv(1);
        if(debug())
            CONDITIONAL_PRINT_HERE((debug() && verbose() > 1), "[timem]> command :: '%s'",
                                   argpc.args().c_str());
        int ret = execvp(argpc.argv()[0], argpc.argv());
        // explain error if enabled
        explain(ret, argpc.argv()[0], argpc.argv());
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

    exit(ret);
}

//--------------------------------------------------------------------------------------//

static bool&
timem_mpi_was_finalized()
{
    static bool _instance = false;
    return _instance;
}

//--------------------------------------------------------------------------------------//
