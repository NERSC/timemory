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
//

#include "timemory-run.hpp"

template class BPatch_Vector<BPatch_variableExpr*>;

//======================================================================================//
//
// entry point
//
//======================================================================================//
//
int
main(int argc, char** argv)
{
#if defined(DYNINST_API_RT)
    tim::set_env<std::string>("DYNINSTAPI_RT_LIB", DYNINST_API_RT, 0);
#endif

    argv0                  = argv[0];
    auto env_collect_paths = tim::get_env<std::string>("TIMEMORY_COLLECTION_PATH", "");
    prefix_collection_path(env_collect_paths, collection_paths);

    bool loadlib = false;      // do we have a library loaded? default false
    char mutname[MUTNAMELEN];  // variable to hold mutator name (ie timemory-run)
    char outfile[MUTNAMELEN];  // variable to hold output file
    char fname[FUNCNAMELEN];
    char libname[FUNCNAMELEN];
    char sharedlibname[FUNCNAMELEN];  // function name and library name variables
    char staticlibname[FUNCNAMELEN];
    BPatch_process*              appThread = nullptr;  // application thread
    BPatch_Vector<BPatch_point*> mpiinit;
    BPatch_function*             mpiinitstub = nullptr;
    bpatch                                   = new BPatch;  // create a new version.
    std::string        inputlib              = "";
    tim::process::id_t _pid                  = -1;

    // bpatch->setTrampRecursive(true); /* enable C++ support */
    // bpatch->setBaseTrampDeletion(true);
    bpatch->setTypeChecking(false);
    bpatch->setMergeTramp(true);
    bpatch->setSaveFPR(true);

    int _argc = argc;
    int _cmdc = 0;

    char** _argv = new char*[_argc];
    char** _cmdv = nullptr;

    for(int i = 0; i < argc; ++i)
        _argv[i] = nullptr;

    auto copy_str = [](char*& _dst, const char* _src) { _dst = strdup(_src); };

    copy_str(_argv[0], argv[0]);

    for(int i = 1; i < argc; ++i)
    {
        std::string _arg = argv[i];
        if(_arg.length() == 2 && _arg == "--")
        {
            _argc        = i;
            _cmdc        = argc - i - 1;
            _cmdv        = new char*[_cmdc + 1];
            _cmdv[_cmdc] = nullptr;
            int k        = 0;
            for(int j = i + 1; j < argc; ++j, ++k)
            {
                copy_str(_cmdv[k], argv[j]);
            }
            strcpy(mutname, _cmdv[0]);
            break;
        }
        else
        {
            copy_str(_argv[i], argv[i]);
        }
    }

    auto cmd_string = [](int _ac, char** _av) {
        std::stringstream ss;
        for(int i = 0; i < _ac; ++i)
            ss << _av[i] << " ";
        return ss.str();
    };

    if(debugPrint)
    {
        std::cout << "[original]: " << cmd_string(argc, argv) << std::endl;
        std::cout << "[cfg-args]: " << cmd_string(_argc, _argv) << std::endl;
    }

    if(_cmdc > 0)
        std::cout << "\n [command]: " << cmd_string(_cmdc, _cmdv) << "\n\n";

    // now can loop through the options.  If the first character is '-', then we know we
    // have an option.  Check to see if it is one of our options and process it.  If it is
    // unrecognized, then set the errflag to report an error.  When we come to a non '-'
    // charcter, then we must be at the application name.
    tim::argparse::argument_parser parser("timemory-run");

    parser.enable_help();
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
        .count(0);
    parser.add_argument()
        .names({ "-e", "--error" })
        .description("All warnings produce runtime errors")
        .count(0);
    parser.add_argument()
        .names({ "-o", "--output" })
        .description("Enable binary-rewrite to new executable")
        .count(1);
    parser.add_argument()
        .names({ "-R", "--regex-include" })
        .description("Regex for selecting functions");
    parser.add_argument()
        .names({ "-E", "--regex-exclude" })
        .description("Regex for excluding functions");
    parser.add_argument()
        .names({ "-m", "--main-function" })
        .description("The primary function to instrument around, e.g. 'main'")
        .count(1);
    parser.add_argument()
        .names({ "-l", "--instrument-loops" })
        .description("Instrument at the loop level")
        .count(0);
    parser.add_argument()
        .names({ "-C", "--collection" })
        .description("Include text file(s) listing function to include or exclude "
                     "(prefix with '!' to exclude)");
    parser.add_argument()
        .names({ "-P", "--collection-path" })
        .description("Additional path(s) to folders containing collection files");
    parser.add_argument()
        .names({ "-s", "--stubs" })
        .description("Instrument with library stubs for LD_PRELOAD")
        .count(0);
    parser.add_argument()
        .names({ "-L", "--library" })
        .description("Library with instrumentation routines (default: \"libtimemory\")")
        .count(1);
    parser.add_argument()
        .names({ "-S", "--stdlib" })
        .description(
            "Enable instrumentation of C++ standard library functions. Use with caution! "
            "May causes deadlocks because timemory uses the STL internally. Use the "
            "'-E/--regex-exclude' option to exclude any deadlocking functions")
        .count(0);
    parser.add_argument()
        .names({ "-p", "--pid" })
        .description("Connect to running process")
        .count(1);
    parser.add_argument()
        .names({ "-d", "--default-components" })
        .description("Default components to instrument");
    parser.add_argument()
        .names({ "-M", "--mode" })
        .description("Instrumentation mode. 'trace' mode is immutable, 'region' mode is "
                     "mutable by timemory library interface")
        .choices({ "trace", "region" })
        .count(1);
    parser.add_argument({ "--mpi" },
                        "Enable MPI support (requires timemory built w/ MPI and GOTCHA "
                        "support)");

    if(_cmdc == 0)
    {
        parser.add_argument()
            .names({ "-c", "--command" })
            .description("Input executable and arguments (if '-- <CMD>' not provided)")
            .required(true);
    }

    std::string extra_help = "-- <CMD> <ARGS>";
    auto        err        = parser.parse(_argc, _argv);
    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help(extra_help);
        return -1;
    }

    if(parser.exists("h") || parser.exists("help"))
    {
        parser.print_help(extra_help);
        return 0;
    }

    if(parser.exists("e"))
        werror = true;

    if(parser.exists("v"))
        debugPrint = 1;

    if(parser.exists("m"))
        main_fname = parser.get<std::string>("m");

    if(parser.exists("l"))
        loop_level_instr = true;

    if(_cmdc == 0 && parser.exists("c"))
    {
        using argtype = std::vector<std::string>;
        auto keys     = parser.get<argtype>("c");
        if(keys.empty())
        {
            parser.print_help(extra_help);
            return EXIT_FAILURE;
        }
        strcpy(mutname, keys.at(0).c_str());
        _cmdc = keys.size();
        _cmdv = new char*[_cmdc];
        for(int i = 0; i < _cmdc; ++i)
        {
            copy_str(_cmdv[i], keys.at(i).c_str());
        }
    }

    if(parser.exists("o"))
    {
        binaryRewrite = true;
        auto key      = parser.get<std::string>("o");
        strcpy(outfile, key.c_str());
    }

    if(parser.exists("s") && !parser.exists("L"))
        inputlib += "stubs/";

    if(!parser.exists("L"))
        inputlib += "libtimemory";

    if(parser.exists("S"))
        stl_func_instr = true;

    if(parser.exists("mpi"))
        use_mpi = true;

    if(parser.exists("p"))
        _pid = parser.get<int>("p");

    if(parser.exists("d"))
    {
        using argtype      = std::vector<std::string>;
        auto _components   = parser.get<argtype>("d");
        default_components = "";
        for(size_t i = 0; i < _components.size(); ++i)
        {
            if(_components.at(i) == "none")
            {
                default_components = "";
                break;
            }
            default_components += _components.at(i);
            if(i + 1 < _components.size())
                default_components += ",";
        }
    }

    if(parser.exists("M"))
    {
        auto _mode = parser.get<std::string>("M");
        if(_mode == "trace")
        {
            instr_push_func = "timemory_push_trace";
            instr_pop_func  = "timemory_pop_trace";
        }
        else if(_mode == "region")
        {
            instr_push_func = "timemory_push_region";
            instr_pop_func  = "timemory_pop_region";
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //                              REGEX OPTIONS
    //
    //----------------------------------------------------------------------------------//
    //
    //  Helper function for adding regex expressions
    //
    auto add_regex = [](auto& regex_array, const std::string& regex_expr) {
        auto regex_constants =
            std::regex_constants::ECMAScript | std::regex_constants::icase;
        if(!regex_expr.empty())
            regex_array.push_back(std::regex(regex_expr, regex_constants));
    };

    add_regex(regex_include, tim::get_env<std::string>("TIMEMORY_REGEX_INCLUDE", ""));
    add_regex(regex_exclude, tim::get_env<std::string>("TIMEMORY_REGEX_EXCLUDE", ""));

    using regex_arg_t = std::vector<std::string>;

    if(parser.exists("R"))
    {
        auto keys = parser.get<regex_arg_t>("R");
        for(const auto& itr : keys)
            add_regex(regex_include, itr);
    }

    if(parser.exists("E"))
    {
        auto keys = parser.get<regex_arg_t>("E");
        for(const auto& itr : keys)
            add_regex(regex_exclude, itr);
    }
    //
    //----------------------------------------------------------------------------------//

    //----------------------------------------------------------------------------------//
    //
    //                              COLLECTION OPTIONS
    //
    //----------------------------------------------------------------------------------//
    //
    if(parser.exists("P"))
    {
        auto tmp = parser.get<std::vector<std::string>>("P");
        for(auto itr : tmp)
            prefix_collection_path(itr, collection_paths);
    }

    if(parser.exists("C"))
    {
        auto tmp = parser.get<std::vector<std::string>>("C");
        for(auto itr : tmp)
        {
            if(itr.find('!') == 0)
                read_collection(itr.substr(1), collection_excludes);
            else
                read_collection(itr, collection_includes);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //                              MAIN
    //
    //----------------------------------------------------------------------------------//

    char* bindings = (char*) malloc(1024);
    dprintf("mutatee name = %s\n", mutname);

    // did we load a library?  if not, load the default
    auto generate_libname = [](char* _targ, std::string _base, std::string _ext) {
        sprintf(_targ, "%s%s", _base.c_str(), _ext.c_str());
    };

    if(!loadlib)
    {
        generate_libname(libname, inputlib, "");
        generate_libname(sharedlibname, inputlib, ".so");
        generate_libname(staticlibname, inputlib, ".a");
        loadlib = true;
    }

    if(_cmdc == 0)
        return EXIT_FAILURE;

    // Register a callback function that prints any error messages
    bpatch->registerErrorCallback(error_func_real);

    if(binaryRewrite)
    {
        dprintf("Before timemory_rewrite_binary\n");
        timemory_rewrite_binary(bpatch, mutname, outfile, (char*) sharedlibname,
                                (char*) staticlibname, bindings);
        char cwd[FUNCNAMELEN];
        auto ret = getcwd(cwd, FUNCNAMELEN);
        consume_parameters(ret);
        // exit from the application
        printf("The instrumented executable image is stored in '%s/%s'\n", cwd, outfile);
        delete bpatch;
        return 0;
    }

    if(_pid >= 0)
    {
        dprintf("Before processAttach\n");
        char* _cmdv0 = (_cmdc > 0) ? _cmdv[0] : nullptr;
        appThread    = bpatch->processAttach(_cmdv0, _pid);
        if(!appThread)
            fprintf(stderr, "[timemory-run]> Failed to connect to process %i\n",
                    (int) _pid);
    }
    else
    {
        dprintf("Before processCreate\n");
        appThread = bpatch->processCreate(_cmdv[0], (const char**) _cmdv, nullptr);
        if(!appThread)
            fprintf(stderr, "[timemory-run]> createProcess failed\n");
    }

    if(!appThread)
        exit(EXIT_FAILURE);

    dprintf("Before getImage and getModules\n");

    // get image
    BPatch_image*                  appImage = appThread->getImage();
    BPatch_Vector<BPatch_module*>* m_full   = appImage->getModules();
    BPatch_Vector<BPatch_module*>  m        = *m_full;

    char                           modulename[512];
    BPatch_Vector<BPatch_snippet*> initArgs;

    dprintf("Before modules loop\n");
    std::map<std::string, std::vector<function_signature>> functions;
    for(size_t j = 0; j < m.size(); j++)
    {
        if(!m[j]->getProcedures())
            continue;

        sprintf(modulename, "Module %s\n", m[j]->getName(fname, FUNCNAMELEN));
        BPatch_Vector<BPatch_function*>* p = m[j]->getProcedures();

        if(!module_constraint(fname))
        {
            for(size_t i = 0; i < p->size(); ++i)
            {
                // For all procedures within the module, iterate
                p->at(i)->getName(fname, FUNCNAMELEN);
                auto name = get_func_file_line_info(appImage, p->at(i));
                if(!routine_constraint(fname) && !name.get().empty() &&
                   !routine_constraint(name.m_name.c_str()))
                {
                    // routines that are ok to instrument
                    // get full source information
                    functions[fname].emplace_back(name);
                }
            }
        }
    }

    dprintf("Module size before loading instrumentation library: %lu\n",
            (long unsigned) m.size());

    // Load the library that has entry and exit routines.
    // Do not load the timemory library if we're rewriting the binary. Use LD_PRELOAD
    // instead. The library may be loaded at a different location.
    if(loadlib == true)
    {
        dprintf("Before appThread->loadLibrary\n");
        // try and load the library
        bool ret = appThread->loadLibrary(sharedlibname, true);

        if(ret == true)
        {
            // now, check to see if the library is listed as a module in the
            // application image
            char name[FUNCNAMELEN];
            bool found = false;
            for(size_t i = 0; i < m_full->size(); ++i)
            {
                m_full->at(i)->getName(name, sizeof(name));
                auto _name = std::string(name);
                if(_name.find(libname) != std::string::npos)
                {
                    found = true;
                    break;
                }
            }
            if(found)
            {
                dprintf("%s loaded properly\n", sharedlibname);
            }
            else
            {
                fprintf(stderr, "Error in loading library %s\n", sharedlibname);
                appThread->terminateExecution();
                return EXIT_FAILURE;
            }
        }
        else
        {
            fprintf(stderr, "ERROR:%s not loaded properly. \n", sharedlibname);
            fprintf(stderr, "Please make sure that its path is in your LD_LIBRARY_PATH "
                            "environment variable.\n");
            appThread->terminateExecution();
            return EXIT_FAILURE;
        }
        dprintf("Load library: %i\n", (int) ret);
    }

    dprintf("Module size after loading instrumentation library: %lu\n",
            (long unsigned) m.size());

    dprintf("Before find_function\n");

    auto* enterstub     = find_function(appImage, instr_push_func.c_str());
    auto* exitstub      = find_function(appImage, instr_pop_func.c_str());
    auto* setupstub     = find_function(appImage, "timemory_trace_init");
    auto* terminatestub = find_function(appImage, "timemory_trace_finalize");
    auto* mpistub       = find_function(appImage, "timemory_trace_set_mpi");

    if(use_mpi && !mpistub)
    {
        throw std::runtime_error("MPI support was requested but timemory was not built "
                                 "with MPI and GOTCHA support");
    }
    else if(use_mpi)
    {
        BPatch_Vector<BPatch_snippet*> mpiArgs;
        mpiArgs.push_back(new BPatch_constExpr(use_mpi));
        BPatch_funcCallExpr mpiExpr(*mpistub, mpiArgs);
        appThread->oneTimeCode(mpiExpr);
    }

    BPatch_Vector<BPatch_snippet*> setupArgs;
    setupArgs.push_back(new BPatch_constExpr(default_components.c_str()));
    BPatch_funcCallExpr setupExpr(*setupstub, setupArgs);
    appThread->oneTimeCode(setupExpr);

    BPatch_Vector<BPatch_snippet*> terminateArgs;
    BPatch_funcCallExpr            terminateExpr(*terminatestub, terminateArgs);
    terminate_expr = &terminateExpr;

    auto exit_callback = bpatch->registerExitCallback(&timemory_thread_exit);
    consume_parameters(exit_callback);

    initialize(appThread, appImage, initArgs);

    dprintf("Did initialize\n");

    // In our tests, the routines started execution concurrently with the
    // one time code. To avoid this, we first start the one time code and then
    // iterate through the list of routines to select for instrumentation and
    // instrument these. So, we need to iterate twice.
    appThread->beginInsertionSet();
    for(size_t j = 0; j < m.size(); j++)
    {
        if(!m[j]->getProcedures())
            continue;
        sprintf(modulename, "Module %s\n", m[j]->getName(fname, FUNCNAMELEN));
        BPatch_Vector<BPatch_function*>* p = m[j]->getProcedures();

        // auto mitr = functions.find(modulename);
        // if(mitr == functions.end())
        //    continue;

        dprintf("%s", modulename);

        if(!module_constraint(fname))
        {
            for(size_t i = 0; i < p->size(); ++i)
            {
                // For all procedures within the module, iterate
                p->at(i)->getName(fname, FUNCNAMELEN);
                auto inFunc = p->at(i);
                auto name   = get_func_file_line_info(appImage, inFunc);

                if(!routine_constraint(fname) && !name.get().empty() &&
                   !routine_constraint(name.m_name.c_str()))
                {
                    // routines that are ok to instrument
                    dprintf("Instrumenting |> [ %s ]\n", name.m_name.c_str());

                    auto callee_entry_args = new BPatch_Vector<BPatch_snippet*>();
                    auto callee_exit_args  = new BPatch_Vector<BPatch_snippet*>();

                    callee_entry_args->push_back(
                        new BPatch_constExpr(name.get().c_str()));
                    callee_exit_args->push_back(new BPatch_constExpr(name.get().c_str()));

                    invoke_routine_in_func(appThread, appImage, inFunc, BPatch_entry,
                                           enterstub, callee_entry_args);
                    invoke_routine_in_func(appThread, appImage, inFunc, BPatch_exit,
                                           exitstub, callee_exit_args);
                }
            }
        }
    }

    BPatch_function* exitpoint = find_function(appImage, "exit");
    if(exitpoint == nullptr)
        exitpoint = find_function(appImage, "_exit");

    if(exitpoint == nullptr)
    {
        fprintf(stderr, "UNABLE TO FIND exit()\n");
        exit(1);
    }
    else
    {
        // When _exit is invoked, call termination routine
        BPatch_Vector<BPatch_snippet*>* exitargs = new BPatch_Vector<BPatch_snippet*>();
        BPatch_constExpr*               name     = new BPatch_constExpr("_exit");
        exitargs->push_back(name);
        invoke_routine_in_func(appThread, appImage, exitpoint, BPatch_entry,
                               terminatestub, exitargs);
        delete exitargs;
        delete name;
    }

    if(!mpiinit.size())
    {
        dprintf("*** MPI_Init not found. This is not an MPI Application! \n");
    }
    else if(mpiinitstub)
    {
        // we've found either MPI_Comm_rank or PMPI_Comm_rank!
        dprintf("FOUND MPI_Comm_rank or PMPI_Comm_rank! \n");
        auto             mpistubargs = new BPatch_Vector<BPatch_snippet*>();
        BPatch_paramExpr paramRank(1);
        mpistubargs->push_back(&paramRank);
        invoke_routine_in_func(appThread, appImage, mpiinit, mpiinitstub, mpistubargs);
        delete mpistubargs;
    }

    auto success = appThread->finalizeInsertionSet(true);

    if(!success)
    {
        fprintf(stderr,
                "Instrumentation failure! Detaching from process and exiting...\n");
        appThread->detach(true);
        exit(EXIT_FAILURE);
    }

    printf("Executing...\n");

    auto _continue_exec = [&]() {
        if(!appThread->continueExecution())
            fprintf(stderr, "continueExecution failed\n");
    };

    auto _wait_exec = [&]() {
        while(!appThread->isTerminated())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            bpatch->waitForStatusChange();
            if(appThread->isStopped())
                _continue_exec();
        }
    };

    appThread->continueExecution();
    _continue_exec();
    _wait_exec();

    if(!appThread->isTerminated())
        appThread->terminateExecution();

    if(appThread->terminationStatus() == ExitedNormally)
    {
        if(appThread->isTerminated())
            printf("End of application\n");
    }
    else if(appThread->terminationStatus() == ExitedViaSignal)
    {
        auto sign = appThread->getExitSignal();
        fprintf(stderr, "Application exited with signal: %i\n", int(sign));
    }

    auto code = appThread->getExitCode();

    // cleanup
    for(int i = 0; i < argc; ++i)
        delete[] _argv[i];
    delete[] _argv;
    for(int i = 0; i < _cmdc; ++i)
        delete[] _cmdv[i];
    delete[] _cmdv;
    delete bpatch;
    return code;
}

//======================================================================================//
//
int
timemory_rewrite_binary(BPatch* bpatch, const char* mutateeName, char* outfile,
                        char* sharedlibname, char* staticlibname, char* bindings)
{
    dprintf("Inside timemory_rewrite_binary, name=%s, out=%s\n", mutateeName, outfile);
    BPatch_binaryEdit* mutatee_addr_space = bpatch->openBinary(mutateeName, false);

    if(mutatee_addr_space == nullptr)
    {
        fprintf(stderr, "Failed to open binary %s\n", mutateeName);
        throw std::runtime_error("Failed to open binary");
    }

    BPatch_image*                   mutateeImage = mutatee_addr_space->getImage();
    BPatch_Vector<BPatch_function*> allFuncs     = *mutateeImage->getProcedures();
    bool isStaticExecutable = mutatee_addr_space->isStaticExecutable();

    if(isStaticExecutable)
    {
        bool result = mutatee_addr_space->loadLibrary(staticlibname);
        dprintf("staticlibname loaded result = %d\n", result);
        assert(result);
    }
    else
    {
        bool result = mutatee_addr_space->loadLibrary(sharedlibname);
        if(!result)
        {
            printf("Error: loadLibrary(%s) failed. Please ensure that timemory's lib "
                   "directory is in your LD_LIBRARY_PATH environment variable and retry."
                   "\n",
                   sharedlibname);
        }
        assert(result);
    }

    auto* entryTrace  = find_function(mutateeImage, instr_push_func.c_str());
    auto* exitTrace   = find_function(mutateeImage, instr_pop_func.c_str());
    auto* setupFunc   = find_function(mutateeImage, "timemory_trace_init");
    auto* cleanupFunc = find_function(mutateeImage, "timemory_trace_finalize");
    auto* mainFunc    = find_function(mutateeImage, main_fname.c_str());
    auto* mpiFunc     = find_function(mutateeImage, "timemory_trace_set_mpi");

    dprintf("Instrumenting with '%s' and '%s'...\n", instr_push_func.c_str(),
            instr_pop_func.c_str());

    if(!mainFunc)
    {
        fprintf(stderr, "Couldn't find %s(), aborting\n", main_fname.c_str());
        throw std::runtime_error("Could not find main function");
    }

    if(!entryTrace || !exitTrace || !setupFunc || !cleanupFunc)
    {
        fprintf(stderr, "Couldn't find entry/exit/setup/cleanup functions, aborting\n");
        throw std::runtime_error("Not entry/exit/setup/cleanup functions");
    }

    if(use_mpi && !mpiFunc)
    {
        throw std::runtime_error("MPI support was requested but timemory was not built "
                                 "with MPI and GOTCHA support");
    }

    // This heuristic guesses that debugging info. is available if main
    // is not defined in the DEFAULT_MODULE
    bool           hasDebuggingInfo = false;
    BPatch_module* mainModule       = mainFunc->getModule();
    if(nullptr != mainModule)
    {
        char moduleName[MUTNAMELEN];
        mainModule->getName(moduleName, MUTNAMELEN);
        if(strcmp(moduleName, "DEFAULT_MODULE") != 0)
            hasDebuggingInfo = true;
    }

    BPatch_Vector<BPatch_point*>* main_init_entry = mainFunc->findPoint(BPatch_entry);
    BPatch_Vector<BPatch_point*>* main_fini_entry = mainFunc->findPoint(BPatch_exit);

    assert(main_init_entry);
    assert(main_init_entry->size());
    assert((*main_init_entry)[0]);

    assert(main_fini_entry);
    assert(main_fini_entry->size());
    assert((*main_fini_entry)[0]);

    // begin insertion
    mutatee_addr_space->beginInsertionSet();

    BPatch_Vector<BPatch_snippet*> mpi_params;
    BPatch_Vector<BPatch_snippet*> init_params;
    BPatch_Vector<BPatch_snippet*> fini_params;

    mpi_params.push_back(new BPatch_constExpr(use_mpi));
    init_params.push_back(new BPatch_constExpr(default_components.c_str()));

    BPatch_funcCallExpr* mpi_call = nullptr;
    if(mpiFunc)
        mpi_call = new BPatch_funcCallExpr(*mpiFunc, mpi_params);

    BPatch_funcCallExpr setup_call(*setupFunc, init_params);
    BPatch_funcCallExpr cleanup_call(*cleanupFunc, fini_params);

    if(mpi_call)
        init_names.push_back(mpi_call);
    init_names.push_back(&setup_call);
    fini_names.push_back(&cleanup_call);

    for(auto itr = allFuncs.begin(); itr != allFuncs.end(); ++itr)
    {
        char fname[FUNCNAMELEN];
        (*itr)->getName(fname, FUNCNAMELEN);

        if(!(*itr)->isInstrumentable())
        {
            dprintf("Skipping uninstrumentable function: %s\n", fname);
            continue;
        }

        // STATIC EXECUTABLE FUNCTION EXCLUDE
        // Temporarily avoid some functions -- this isn't a solution
        // -- it appears that something like module_constraint would work
        // well here
        if(isStaticExecutable)
        {
            // Always instrument _fini to ensure instrumentation disabled correctly
            if(hasDebuggingInfo && strcmp(fname, "_fini") != 0)
            {
                BPatch_module* funcModule = (*itr)->getModule();
                if(funcModule != nullptr)
                {
                    char moduleName[MUTNAMELEN];
                    funcModule->getName(moduleName, MUTNAMELEN);
                    if(strcmp(moduleName, "DEFAULT_MODULE") == 0)
                        continue;
                }
            }
        }

        char modname[1024];
        (*itr)->getModuleName(modname, 1024);
        if(strstr(modname, "libdyninstAPI_RT"))
            continue;

        if(module_constraint(modname))
            continue;

        auto name = get_func_file_line_info(mutatee_addr_space->getImage(), *itr);
        if(name.get().empty() || routine_constraint(fname) ||
           routine_constraint(name.m_name.c_str()) || !instrument_entity(name.m_name) ||
           !instrument_entity(name.get()))
            continue;

        insert_trace(*itr, mutatee_addr_space, entryTrace, exitTrace, name);

        if(loop_level_instr)
        {
            dprintf("Generating CFG at loop level: %s\n", fname);
            BPatch_flowGraph*                     flow = (*itr)->getCFG();
            BPatch_Vector<BPatch_basicBlockLoop*> basicLoop;
            dprintf("Generating outer loop info : %s\n", fname);
            flow->getOuterLoops(basicLoop);
            dprintf("Before instrumenting at loop level: %s\n", fname);

            for(auto litr = basicLoop.begin(); litr != basicLoop.end(); litr++)
            {
                dprintf("Instrumenting at the loop level: %s\n", fname);
                insert_trace(*itr, mutatee_addr_space, entryTrace, exitTrace, flow, *litr,
                             name);
            }
        }
    }

    BPatch_sequence init_sequence(init_names);
    mutatee_addr_space->insertSnippet(init_sequence, *main_init_entry, BPatch_callBefore,
                                      BPatch_firstSnippet);
    BPatch_sequence fini_sequence(fini_names);
    mutatee_addr_space->insertSnippet(fini_sequence, *main_fini_entry, BPatch_callAfter,
                                      BPatch_firstSnippet);

    // finalize insertion
    mutatee_addr_space->finalizeInsertionSet(false, nullptr);

    if(isStaticExecutable)
    {
        bool loadResult = load_dependent_libraries(mutatee_addr_space, bindings);
        if(!loadResult)
        {
            fprintf(stderr,
                    "Failed to load dependent libraries need for binary rewrite\n");
            return -1;
        }
    }

    std::string modifiedFileName(outfile);
    int         ret = chdir("result");
    if(ret == 0)
        fprintf(stderr, "Error chdir('result') = %i\n", ret);

    mutatee_addr_space->writeFile(modifiedFileName.c_str());
    if(!isStaticExecutable)
    {
        // unlink(sharedlibname);
    }
    return 0;
}

//======================================================================================//
//
void
read_collection(const std::string& fname, std::set<std::string>& collection_set)
{
    std::string           searched_paths;
    std::set<std::string> prefixes = { ".", get_absolute_path(argv0.c_str()) };
    for(const auto& pitr : prefixes)
    {
        for(auto itr : collection_paths)
        {
            itr = TIMEMORY_JOIN("/", pitr, itr);
            searched_paths += itr;
            searched_paths += ", ";
            auto fpath = TIMEMORY_JOIN("/", itr, fname);

            dprintf("trying to read collection file @ %s...", fpath.c_str());
            std::ifstream ifs(fpath.c_str());

            if(!ifs)
            {
                dprintf("trying to read collection file @ %s...", fpath.c_str());
                fpath = TIMEMORY_JOIN("/", itr, to_lower(fname));
                ifs.open(fpath.c_str());
            }

            if(ifs)
            {
                dprintf(">    reading collection file @ %s...", fpath.c_str());
                std::string tmp;
                while(!ifs.eof())
                {
                    ifs >> tmp;
                    if(ifs.good())
                        collection_set.insert(tmp);
                }
                return;
            }
        }
    }
    searched_paths = searched_paths.substr(0, searched_paths.length() - 2);
    std::stringstream ss;
    ss << "Unable to find \"" << fname << "\". Searched paths: " << searched_paths;
    if(werror)
        throw std::runtime_error(ss.str());
    else
        std::cerr << ss.str() << "\n";
}

//======================================================================================//

bool
process_file_for_instrumentation(const std::string& file_name)
{
    std::regex ext_regex("\\.(c|C|S)$");
    std::regex userlib_regex("^lib(timemory|caliper|gotcha|papi|cupti|TAU|likwid|"
                             "profiler|tcmalloc|dyninst|pfm|nvtx|upcxx|pthread)");
    std::regex corelib_regex("^lib(rt-|m-|dl-|m-|util-|gcc|c-|stdc++|c++|python)");
    // these are all due to TAU
    std::regex prefix_regex("^(Tau|Profiler|Rts|Papi|Py|Comp_xl\\.cpp|Comp_gnu\\.cpp|"
                            "UserEvent\\.cpp|FunctionInfo\\.cpp|PthreadLayer\\.cpp|"
                            "Comp_intel[0-9]\\.cpp|Tracer\\.cpp)");
    if(check_project_source_file_for_instrumentation(file_name) ||
       std::regex_search(file_name, ext_regex) ||
       std::regex_search(file_name, corelib_regex) ||
       std::regex_search(file_name, prefix_regex) ||
       std::regex_search(file_name, userlib_regex))
        return false;

    auto is_include = [&]() {
        if(regex_include.empty())
            return true;
        for(auto& itr : regex_include)
        {
            if(std::regex_search(file_name, itr))
                return true;
        }
        return false;
    };

    auto is_exclude = [&]() {
        for(auto& itr : regex_exclude)
        {
            if(std::regex_search(file_name, itr))
                return true;
        }
        return false;
    };

    bool use = is_include() && !is_exclude();
    if(use)
    {
        static std::set<std::string> already_reported;
        if(already_reported.count(file_name) == 0)
        {
            dprintf("%s |> [ %s ]\n", __FUNCTION__, file_name.c_str());
            already_reported.insert(file_name);
        }
    }
    return use;
}

//======================================================================================//

bool
instrument_entity(const std::string& function_name)
{
    std::regex exclude(
        "(timemory|tim::|cereal|N3tim|MPI_Init|MPI_Finalize|\\{lambda|::_["
        "A-Z]|::__[A-Za-z]|std::max|std::min|std::fill|std::forward|std::get)");
    std::regex leading("^(_init|_fini|__|_dl_|_start|_exit|frame_dummy|\\(\\(|\\(__|_"
                       "GLOBAL|targ|PMPI_|new|delete|std::allocator|std::move|nvtx|gcov|"
                       "main\\.cold\\.|TAU|tau|Tau)");
    std::regex stlfunc("^std::");

    if(!stl_func_instr && std::regex_search(function_name, stlfunc))
    {
        dprintf("Skipping stl function '%s'...\n", function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude))
        return false;

    // don't instrument the functions when key is found at the start of the function name
    if(std::regex_search(function_name, leading))
        return false;

    auto is_include = [&]() {
        if(regex_include.empty())
            return true;
        for(auto& itr : regex_include)
        {
            if(std::regex_search(function_name, itr))
                return true;
        }
        return false;
    };

    auto is_exclude = [&]() {
        for(auto& itr : regex_exclude)
        {
            if(std::regex_search(function_name, itr))
                return true;
        }
        return false;
    };

    bool use = is_include() && !is_exclude();
    return use;
}

//======================================================================================//
// InsertTrace function for loop-level instrumentation.
// Bug exists at the moment that the second line number is
// the last command at the outermost loop's level. So, if the outer
// loop has a nested loop inside, with blank lines afterwards,
// only the lines from the beginning of the outer loop to the
// beginning of the outer loop are counted.
//
void
insert_trace(BPatch_function* funcToInstr, BPatch_addressSpace* mutatee,
             BPatch_function* traceEntryFunc, BPatch_function* traceExitFunc,
             BPatch_flowGraph* cfGraph, BPatch_basicBlockLoop* loopToInstrument,
             function_signature& name)
{
    BPatch_module* module = funcToInstr->getModule();
    if(!module)
        return;

    BPatch_Vector<BPatch_point*>* loopEntr =
        cfGraph->findLoopInstPoints(BPatch_locLoopEntry, loopToInstrument);
    BPatch_Vector<BPatch_point*>* loopExit =
        cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);

    if(loopEntr == nullptr || loopExit == nullptr)
        return;
    if(loopEntr->empty() || loopExit->empty())
        return;

    dprintf("Instrumenting |> [ %s ]\n", name.m_name.c_str());

    BPatch_Vector<BPatch_snippet*> entryTraceArgs;
    BPatch_Vector<BPatch_snippet*> exitTraceArgs;

    entryTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));
    exitTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));

    BPatch_funcCallExpr entryTrace(*traceEntryFunc, entryTraceArgs);
    BPatch_funcCallExpr exitTrace(*traceExitFunc, exitTraceArgs);

    for(size_t i = 0; i < loopEntr->size(); ++i)
    {
        if(loopEntr->at(i))
            mutatee->insertSnippet(entryTrace, *loopEntr->at(i), BPatch_callBefore,
                                   BPatch_firstSnippet);
    }

    for(size_t i = 0; i < loopExit->size(); ++i)
    {
        if(loopExit->at(i))
            mutatee->insertSnippet(exitTrace, *loopExit->at(i), BPatch_callAfter,
                                   BPatch_lastSnippet);
    }
}

//======================================================================================//

void
insert_trace(BPatch_function* funcToInstr, BPatch_addressSpace* mutatee,
             BPatch_function* traceEntryFunc, BPatch_function* traceExitFunc,
             function_signature& name)
{
    dprintf("Instrumenting |> [ %s ]\n", name.m_name.c_str());

    BPatch_Vector<BPatch_point*> funcEntry = *funcToInstr->findPoint(BPatch_entry);
    BPatch_Vector<BPatch_point*> funcExit  = *funcToInstr->findPoint(BPatch_exit);

    BPatch_Vector<BPatch_snippet*> entryTraceArgs;
    BPatch_Vector<BPatch_snippet*> exitTraceArgs;

    entryTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));
    exitTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));

    BPatch_funcCallExpr entryTrace(*traceEntryFunc, entryTraceArgs);
    BPatch_funcCallExpr exitTrace(*traceExitFunc, exitTraceArgs);

    mutatee->insertSnippet(entryTrace, funcEntry, BPatch_callBefore, BPatch_firstSnippet);
    mutatee->insertSnippet(exitTrace, funcExit, BPatch_callAfter, BPatch_firstSnippet);
}

//======================================================================================//
//
// invoke_routine_in_func calls routine "callee" with no arguments when
// Function "function" is invoked at the point given by location
//
BPatchSnippetHandle*
invoke_routine_in_func(BPatch_process* appThread, BPatch_image* appImage,
                       BPatch_function* function, BPatch_procedureLocation loc,
                       BPatch_function*                callee,
                       BPatch_Vector<BPatch_snippet*>* callee_args)
{
    consume_parameters(appImage);
    // First create the snippet using the callee and the args
    const BPatch_snippet* snippet = new BPatch_funcCallExpr(*callee, *callee_args);

    if(snippet == nullptr)
    {
        fprintf(stderr, "Unable to create snippet to call callee\n");
        exit(EXIT_FAILURE);
    }

    // Then find the points using loc (entry/exit) for the given function
    const BPatch_Vector<BPatch_point*>* points = function->findPoint(loc);

    if(points != nullptr)
    {
        // Insert the given snippet at the given point
        if(loc == BPatch_entry)
        {
            appThread->insertSnippet(*snippet, *points, BPatch_callBefore,
                                     BPatch_firstSnippet);
        }
        else
        {
            appThread->insertSnippet(*snippet, *points);
        }
    }
    delete snippet;
    return nullptr;
}

//======================================================================================//
//
// invoke_routine_in_func calls routine "callee" with no arguments when
// Function "function" is invoked at the point given by location
//
BPatchSnippetHandle*
invoke_routine_in_func(BPatch_process* appThread, BPatch_image* appImage,
                       BPatch_Vector<BPatch_point*> points, BPatch_function* callee,
                       BPatch_Vector<BPatch_snippet*>* callee_args)
{
    consume_parameters(appImage);
    // First create the snippet using the callee and the args
    const BPatch_snippet* snippet = new BPatch_funcCallExpr(*callee, *callee_args);
    if(snippet == NULL)
    {
        fprintf(stderr, "Unable to create snippet to call callee\n");
        exit(1);
    }

    if(points.size())
    {
        // Insert the given snippet at the given point
        appThread->insertSnippet(*snippet, points);
    }
    delete snippet;
    return nullptr;
}

//======================================================================================//
//
// initialize calls InitCode, the initialization routine in the user
// application. It is executed exactly once, before any other routine.
//
void
initialize(BPatch_process* appThread, BPatch_image* appImage,
           BPatch_Vector<BPatch_snippet*>& initArgs)
{
    // Find the initialization function and call it
    BPatch_function* init_func = find_function(appImage, "timemory_trace_init");
    if(init_func == nullptr)
    {
        fprintf(stderr, "Unable to find function timemory_trace_init\n");
        exit(1);
    }

    // Find the finalization function and call it
    BPatch_function* fini_func = find_function(appImage, "timemory_trace_finalize");
    if(fini_func == nullptr)
    {
        fprintf(stderr, "Unable to find function timemory_trace_finalize\n");
        exit(1);
    }

    if(binaryRewrite)
    {
        BPatch_Vector<BPatch_snippet*> finiArgs = {};
        // check_cost(init_Expr);
        // locate the entry point for main
        BPatch_function* main_entry = find_function(appImage, main_fname.c_str());
        if(main_entry == nullptr)
        {
            fprintf(stderr, "timemory-run: Unable to find function '%s'\n",
                    main_fname.c_str());
            exit(1);
        }

        if(!main_entry->findPoint(BPatch_entry))
        {
            fprintf(stderr,
                    "timemory-run: entry points for '%s' or init_snippet for "
                    "Init are null\n",
                    main_fname.c_str());
            exit(1);
        }

        auto init_points  = main_entry->findPoint(BPatch_entry);
        auto init_snippet = BPatch_funcCallExpr(*init_func, initArgs);
        auto fini_points  = main_entry->findPoint(BPatch_exit);
        auto fini_snippet = BPatch_funcCallExpr(*fini_func, finiArgs);
        // We invoke the init_snippet before any other call in main and fini_snippet after
        // any other call in main
        // Insert the init_snippet to be called before first snippet
        appThread->insertSnippet(init_snippet, *init_points, BPatch_callBefore,
                                 BPatch_firstSnippet);
        // Insert the fini_snippet to be called before last snippet
        appThread->insertSnippet(fini_snippet, *fini_points, BPatch_callAfter,
                                 BPatch_firstSnippet);
    }
    else
    {
        BPatch_funcCallExpr init_Expr(*init_func, initArgs);
        appThread->oneTimeCode(init_Expr);
    }
}

//======================================================================================//
// Constraints for instrumentation. Returns true for those modules that
// shouldn't be instrumented.
int
module_constraint(char* fname)
{
    // fname is the name of module/file
    int len = strlen(fname);

    std::string _fname = fname;
    if(_fname.find("timemory") != std::string::npos ||
       _fname.find("tim::") != std::string::npos)
        return true;

    if(process_file_for_instrumentation(std::string(fname)))
        return false;  // ok to instrument

    if(_fname == "DEFAULT_MODULE" || _fname == "LIBRARY_MODULE")
        return false;

    // do not instrument
    return true;

    /*if(are_file_include_exclude_lists_empty())
    {
        // there are no user sepecified constraints on modules. Use our default
        // constraints
        if((strcmp(fname, "DEFAULT_MODULE") == 0) ||
           ((fname[len - 2] == '.') && (fname[len - 1] == 'c')) ||
           ((fname[len - 2] == '.') && (fname[len - 1] == 'C')) ||
           ((fname[len - 3] == '.') && (fname[len - 2] == 'c') &&
            (fname[len - 1] == 'c')) ||
           ((fname[len - 4] == '.') && (fname[len - 3] == 'c') &&
            (fname[len - 2] == 'p') && (fname[len - 1] == 'p')) ||
           ((fname[len - 4] == '.') && (fname[len - 3] == 'f') &&
            (fname[len - 2] == '9') && (fname[len - 1] == '0')) ||
           ((fname[len - 4] == '.') && (fname[len - 3] == 'F') &&
            (fname[len - 2] == '9') && (fname[len - 1] == '0')) ||
           ((fname[len - 2] == '.') && (fname[len - 1] == 'F')) ||
           ((fname[len - 2] == '.') && (fname[len - 1] == 'f')) ||
           //((fname[len-3] == '.') && (fname[len-2] == 's') && (fname[len-1] == 'o'))||
           (strcmp(fname, "LIBRARY_MODULE") == 0))
        {
            // It is ok to instrument this module. Constraint doesn't exist.
            return false;
        }
        else
        {
            return true;
        }
    }
    else
    {
        // See if the file should be instrumented
        if(process_file_for_instrumentation(std::string(fname)))
        {
            // Yes, it should be instrumented. moduleconstraint should return false!
            return false;
        }
        else
        {
            // No, the file should not be instrumented. Constraint exists return true
            return true;
        }
    }
    */
}

//======================================================================================//
// Constraint for routines. The constraint returns true for those routines that
// should not be instrumented.
int
routine_constraint(const char* fname)
{
    std::string _fname = fname;
    if(_fname.find("timemory") != std::string::npos ||
       _fname.find("tim::") != std::string::npos)
        return true;

    if((strncmp(fname, "tim", 3) == 0) || (strstr(fname, "FunctionInfo") != 0) ||
       (strncmp(fname, "RtsLayer", 8) == 0) || (strncmp(fname, "DYNINST", 7) == 0) ||
       (strncmp(fname, "PthreadLayer", 12) == 0) ||
       (strncmp(fname, "threaded_func", 13) == 0) || (strncmp(fname, "targ8", 5) == 0) ||
       (strncmp(fname, "__intel_", 8) == 0) || (strncmp(fname, "_intel_", 7) == 0) ||
       (strncmp(fname, "The", 3) == 0) ||
       // The following functions show up in static executables
       (strncmp(fname, "__mmap", 6) == 0) || (strncmp(fname, "_IO_printf", 10) == 0) ||
       (strncmp(fname, "__write", 7) == 0) || (strncmp(fname, "__munmap", 8) == 0) ||
       (strstr(fname, "_L_lock") != 0) || (strstr(fname, "_L_unlock") != 0))
    {
        return true;  // Don't instrument
    }
    else
    {
        // Should the routine fname be instrumented?
        if(instrument_entity(std::string(fname)))
        {
            // Yes it should be instrumented. Return false
            return false;
        }
        else
        {
            // No. The selective instrumentation file says: don't instrument it
            return true;
        }
    }
}

//======================================================================================//
//
bool
load_dependent_libraries(BPatch_binaryEdit* bedit, char* bindings)
{
    // Order of load matters, just like command line arguments to a standalone linker

    char deplibs[1024];
    char bindir[] = TIMEMORY_BIN_DIR;
    char cmd[1024];
    dprintf("Inside load_dependent_libraries: bindings=%s\n", bindings);
    sprintf(cmd, "%s/timemory_show_libs %s/../lib/Makefile.timemory%s", bindir, bindir,
            bindings);
    dprintf("cmd = %s\n", cmd);
    FILE* fp = popen(cmd, "r");

    if(fp == nullptr)
    {
        perror("timemory-run: Error launching timemory_show_libs to get list of "
               "dependent static libraries for static binary");
        return false;
    }

    while((fgets(deplibs, 1024, fp)) != nullptr)
    {
        int len = strlen(deplibs);
        if(deplibs[len - 2] == ',' && deplibs[len - 3] == '"' && deplibs[0] == '"')
        {
            deplibs[len - 3] = '\0';
            dprintf("LOADING %s\n", &deplibs[1]);
            if(!bedit->loadLibrary(&deplibs[1]))
            {
                fprintf(stderr, "Failed to load dependent library: %s\n", &deplibs[1]);
                return false;
            }
        }
        else
        {
            printf("WARNING: timemory_show_libs in timemory-run: Comma not found! "
                   "deplibs = %s\n",
                   deplibs);
        }
    }

    return true;
}
