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

static std::set<std::string> extra_libs = {};

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

    bool             loadlib = false;
    char             mutname[MUTNAMELEN];
    char             outfile[MUTNAMELEN];
    char             libname[FUNCNAMELEN];
    char             sharedlibname[FUNCNAMELEN];
    char             staticlibname[FUNCNAMELEN];
    address_space_t* addr_space = nullptr;
    bpatch                      = new BPatch;
    std::string        inputlib = "";
    tim::process::id_t _pid     = -1;

    // bpatch->setTrampRecursive(true);
    bpatch->setBaseTrampDeletion(false);
    // bpatch->setMergeTramp(true);
    bpatch->setTypeChecking(false);
    bpatch->setSaveFPR(true);
    bpatch->setDelayedParsing(false);

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

    if(verbose_level > 1)
    {
        std::cout << "[original]: " << cmd_string(argc, argv) << std::endl;
        std::cout << "[cfg-args]: " << cmd_string(_argc, _argv) << std::endl;
    }

    if(_cmdc > 0)
        std::cout << "\n [command]: " << cmd_string(_cmdc, _cmdv) << "\n\n";

    if(_cmdc > 0)
        cmdv0 = _cmdv[0];

    // now can loop through the options.  If the first character is '-', then we know we
    // have an option.  Check to see if it is one of our options and process it.  If it is
    // unrecognized, then set the errflag to report an error.  When we come to a non '-'
    // charcter, then we must be at the application name.
    tim::argparse::argument_parser parser("timemory-run");

    parser.enable_help();
    parser.add_argument()
        .names({ "-v", "--verbose" })
        .description("Verbose output")
        .max_count(1);
    parser.add_argument().names({ "--debug" }).description("Debug output").count(0);
    parser.add_argument()
        .names({ "-e", "--error" })
        .description("All warnings produce runtime errors")
        .count(0);
    parser.add_argument()
        .names({ "-o", "--output" })
        .description("Enable binary-rewrite to new executable")
        .count(1);
    parser.add_argument()
        .names({ "-I", "-R", "--function-include" })
        .description("Regex for selecting functions");
    parser.add_argument()
        .names({ "-E", "--function-exclude" })
        .description("Regex for excluding functions");
    parser.add_argument()
        .names({ "-MI", "-MR", "--module-include" })
        .description("Regex for selecting modules/files/libraries");
    parser.add_argument()
        .names({ "-ME", "--module-exclude" })
        .description("Regex for excluding modules/files/libraries");
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
            "'-E/--function-exclude' option to exclude any deadlocking functions")
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
    parser.add_argument()
        .names({ "--prefer" })
        .description("Prefer this library types when available")
        .choices({ "shared", "static" })
        .count(1);
    parser
        .add_argument({ "--mpi" },
                      "Enable MPI support (requires timemory built w/ MPI and GOTCHA "
                      "support)")
        .count(0);
    parser.add_argument({ "--mpip" }, "Enable MPI profiling via GOTCHA").count(0);
    parser.add_argument({ "--ompt" }, "Enable OpenMP profiling via OMPT").count(0);
    parser.add_argument({ "--load" }, "Extra libraries to load");

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
    {
        if(parser.get_count("v") == 0)
            verbose_level = 1;
        else
            verbose_level = parser.get<int>("v");
    }

    if(parser.exists("debug"))
        verbose_level = 256;

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
        binary_rewrite = true;
        auto key       = parser.get<std::string>("o");
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

    if(parser.exists("mpip"))
        use_mpip = true;

    if(parser.exists("ompt"))
        use_ompt = true;

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

    if(parser.exists("prefer"))
        prefer_library = parser.get<std::string>("prefer");

    if(parser.exists("load"))
    {
        auto _load = parser.get<std::vector<std::string>>("load");
        for(auto itr : _load)
            extra_libs.insert(itr);
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
        auto regex_constants = std::regex_constants::ECMAScript;
        if(!regex_expr.empty())
            regex_array.push_back(std::regex(regex_expr, regex_constants));
    };

    add_regex(func_include, tim::get_env<std::string>("TIMEMORY_REGEX_INCLUDE", ""));
    add_regex(func_exclude, tim::get_env<std::string>("TIMEMORY_REGEX_EXCLUDE", ""));

    using regex_arg_t = std::vector<std::string>;

    if(parser.exists("R"))
    {
        auto keys = parser.get<regex_arg_t>("R");
        for(const auto& itr : keys)
            add_regex(func_include, itr);
    }

    if(parser.exists("E"))
    {
        auto keys = parser.get<regex_arg_t>("E");
        for(const auto& itr : keys)
            add_regex(func_exclude, itr);
    }

    if(parser.exists("MI"))
    {
        auto keys = parser.get<regex_arg_t>("MI");
        for(const auto& itr : keys)
            add_regex(file_include, itr);
    }

    if(parser.exists("ME"))
    {
        auto keys = parser.get<regex_arg_t>("ME");
        for(const auto& itr : keys)
            add_regex(file_exclude, itr);
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
    verbprintf(0, "mutatee name = %s\n", mutname);

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

    addr_space =
        timemory_get_address_space(bpatch, _cmdc, _cmdv, binary_rewrite, _pid, mutname);

    if(!addr_space)
        exit(EXIT_FAILURE);

    BPatch_process*    appThread = nullptr;
    BPatch_binaryEdit* appBinary = nullptr;

    // get image
    verbprintf(1, "Before getImage and getModules\n");
    BPatch_image*                    appImage     = addr_space->getImage();
    BPatch_Vector<BPatch_module*>*   appModules   = appImage->getModules();
    BPatch_Vector<BPatch_function*>* appFunctions = appImage->getProcedures();

    BPatch_Vector<BPatch_module*>   modules;
    BPatch_Vector<BPatch_function*> functions;

    if(appModules)
    {
        modules = *appModules;
        for(auto itr : *appModules)
        {
            char modname[MUTNAMELEN];
            itr->getName(modname, MUTNAMELEN);
            available_modules.insert(modname);
            auto procedures = itr->getProcedures();
            if(procedures)
            {
                for(auto pitr : *procedures)
                {
                    char fname[FUNCNAMELEN];
                    pitr->getName(fname, FUNCNAMELEN);
                    available_procedures.insert(fname);
                }
            }
        }
    }

    if(appFunctions)
    {
        functions = *appFunctions;
        for(auto itr : *appFunctions)
        {
            char modname[MUTNAMELEN];
            char fname[FUNCNAMELEN];

            itr->getModuleName(modname, MUTNAMELEN);
            BPatch_module* mod = itr->getModule();
            if(mod)
                mod->getName(modname, MUTNAMELEN);
            else
                itr->getModuleName(modname, MUTNAMELEN);
            itr->getName(fname, FUNCNAMELEN);
            available_modules.insert(modname);
            available_procedures.insert(fname);
        }
    }

    verbprintf(1, "Module size before loading instrumentation library: %lu\n",
               (long unsigned) appModules->size());

    dump_info("available_modules.txt", available_modules, 2);
    dump_info("available_functions.txt", available_procedures, 2);

    is_static_exe = addr_space->isStaticExecutable();

    if(binary_rewrite)
        appBinary = static_cast<BPatch_binaryEdit*>(addr_space);
    else
        appThread = static_cast<BPatch_process*>(addr_space);

    auto load_library = [addr_space](std::string _libname) {
        verbprintf(0, "loading library: '%s'...\n", _libname.c_str());
        bool result = addr_space->loadLibrary(_libname.c_str());
        verbprintf(1, "loadLibrary(%s) result = %d\n", _libname.c_str(), result);
        if(!result)
        {
            fprintf(stderr,
                    "Error: 'loadLibrary(%s)' failed.\nPlease ensure that the "
                    "library's directory is in your LD_LIBRARY_PATH environment variable "
                    "or the full path is provided\n",
                    _libname.c_str());
            exit(EXIT_FAILURE);
        }
    };

    auto get_library_ext = [=](std::string lname) {
        if(lname.find(".so") != std::string::npos ||
           lname.find(".a") == lname.length() - 2)
            return lname;
        if(!prefer_library.empty())
            return (lname +
                    ((prefer_library == "static" || is_static_exe) ? ".a" : ".so"));
        else
            return (lname + ((is_static_exe) ? ".a" : ".so"));
    };

    auto* _mutatee_init = find_function(appImage, "_init");
    auto* _mutatee_fini = find_function(appImage, "_fini");

    load_library(get_library_ext(libname));

    if(use_mpip && !is_static_exe)
        load_library("libtimemory-mpip.so");

    if(use_ompt)
        load_library(get_library_ext("libtimemory-ompt"));

    for(auto itr : extra_libs)
        load_library(get_library_ext(itr));

    auto* main_func  = find_function(appImage, main_fname.c_str());
    auto* entr_trace = find_function(appImage, instr_push_func.c_str());
    auto* exit_trace = find_function(appImage, instr_pop_func.c_str());
    auto* init_func  = find_function(appImage, "timemory_trace_init");
    auto* fini_func  = find_function(appImage, "timemory_trace_finalize");
    auto* env_func   = find_function(appImage, "timemory_trace_set_env");
    auto* mpi_func   = find_function(appImage, "timemory_trace_set_mpi");

    auto* mpip_beg_stub = find_function(appImage, "timemory_register_mpip");
    auto* mpip_end_stub = find_function(appImage, "timemory_deregister_mpip");
    auto* ompt_beg_stub = find_function(appImage, "timemory_register_ompt");
    auto* ompt_end_stub = find_function(appImage, "timemory_deregister_ompt");

    auto* exit_func = find_function(appImage, "exit");
    if(!exit_func)
        exit_func = find_function(appImage, "_exit");

    verbprintf(0, "Instrumenting with '%s' and '%s'...\n", instr_push_func.c_str(),
               instr_pop_func.c_str());

    if(!main_func)
    {
        fprintf(stderr, "[timemory-run]> Couldn't find '%s'\n", main_fname.c_str());
        if(!_mutatee_init || !_mutatee_fini)
        {
            fprintf(stderr, "[timemory-run]> Couldn't find '%s' or '%s', aborting\n",
                    "_init", "_fini");
            throw std::runtime_error("Could not find main function");
        }
        else
        {
            fprintf(stderr, "[timemory-run]> using '%s' and '%s' in lieu of '%s'...",
                    "_init", "_fini", main_fname.c_str());
        }
    }

    using pair_t = std::pair<procedure_t*, std::string>;

    for(auto itr :
        { pair_t(main_func, main_fname), pair_t(entr_trace, instr_push_func),
          pair_t(exit_trace, instr_pop_func), pair_t(init_func, "timemory_trace_init"),
          pair_t(fini_func, "timemory_trace_finalize"),
          pair_t(env_func, "timemory_trace_set_env") })
    {
        if(itr.first == main_func && _mutatee_init && _mutatee_fini)
            continue;
        if(!itr.first)
        {
            std::stringstream ss;
            ss << "Error! Couldn't find '" << itr.second.c_str() << "' function";
            fprintf(stderr, "[timemory-run]> %s\n", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }

    if(use_mpi && !mpi_func)
    {
        throw std::runtime_error("MPI support was requested but timemory was not built "
                                 "with MPI and GOTCHA support");
    }

    if(use_mpip && !(mpip_beg_stub || mpip_end_stub))
    {
        throw std::runtime_error("MPIP support was requested but could not find "
                                 "timemory_{register,deregister}_mpip functions");
    }

    if(use_ompt && !(ompt_beg_stub || ompt_end_stub))
    {
        throw std::runtime_error("OMPT support was requested but could not find "
                                 "timemory_{register,deregister}_ompt functions");
    }

    // This heuristic guesses that debugging info. is available if main
    // is not defined in the DEFAULT_MODULE
    bool has_debug_info = false;
    if(main_func)
    {
        BPatch_module* main_module = main_func->getModule();
        if(main_module)
        {
            char moduleName[MUTNAMELEN];
            main_module->getName(moduleName, MUTNAMELEN);
            if(strcmp(moduleName, "DEFAULT_MODULE") != 0)
                has_debug_info = true;
        }
    }

    BPatch_Vector<BPatch_point*>* main_entr_points = nullptr;
    BPatch_Vector<BPatch_point*>* main_exit_points = nullptr;

    if(main_func)
    {
        main_entr_points = main_func->findPoint(BPatch_entry);
        main_exit_points = main_func->findPoint(BPatch_exit);
    }
    else
    {
        main_entr_points = _mutatee_init->findPoint(BPatch_entry);
        main_exit_points = _mutatee_fini->findPoint(BPatch_exit);
    }

    // begin insertion
    if(binary_rewrite)
        addr_space->beginInsertionSet();

    function_signature main_sign("int", "main", "");

    auto main_call_args = timemory_call_expr(main_sign.get());
    auto init_call_args = timemory_call_expr(default_components, true, cmdv0);
    auto fini_call_args = timemory_call_expr();
    auto umpi_call_args = timemory_call_expr(use_mpi);
    auto mpip_call_args =
        timemory_call_expr("TIMEMORY_MPIP_COMPONENTS", default_components);
    auto ompt_call_args =
        timemory_call_expr("TIMEMORY_OMPT_COMPONENTS", default_components);
    auto none_call_args = timemory_call_expr();

    auto init_call = init_call_args.get(init_func);
    auto fini_call = fini_call_args.get(fini_func);
    auto umpi_call = umpi_call_args.get(mpi_func);

    auto main_beg_call = main_call_args.get(entr_trace);
    auto main_end_call = main_call_args.get(exit_trace);

    auto mpip_env_call = mpip_call_args.get(env_func);
    auto mpip_beg_call = none_call_args.get(mpip_beg_stub);
    auto mpip_end_call = none_call_args.get(mpip_end_stub);

    auto ompt_env_call = ompt_call_args.get(env_func);
    auto ompt_beg_call = none_call_args.get(ompt_beg_stub);
    auto ompt_end_call = none_call_args.get(ompt_end_stub);

    if(use_mpip)
        verbprintf(0, "+ Adding mpip instrumentation...\n");
    if(use_ompt)
        verbprintf(0, "+ Adding ompt instrumentation...\n");

    if(use_mpip && mpip_env_call)
        init_names.push_back(mpip_env_call.get());
    if(use_ompt && ompt_env_call)
        init_names.push_back(ompt_env_call.get());

    if(use_mpip && mpip_beg_call)
        init_names.push_back(mpip_beg_call.get());
    if(use_ompt && ompt_beg_call)
        init_names.push_back(ompt_beg_call.get());

    if(use_mpi && umpi_call)
        init_names.push_back(umpi_call.get());

    init_names.push_back(init_call.get());

    if(binary_rewrite)
    {
        init_names.push_back(main_beg_call.get());
        fini_names.push_back(main_end_call.get());
    }

    fini_names.push_back(fini_call.get());

    if(use_mpip && mpip_end_call)
        fini_names.push_back(mpip_end_call.get());
    if(use_ompt && ompt_end_call)
        fini_names.push_back(ompt_end_call.get());

    if(binary_rewrite)
    {
        addr_space->insertSnippet(BPatch_sequence(init_names), *main_entr_points,
                                  BPatch_callBefore, BPatch_firstSnippet);
        addr_space->insertSnippet(BPatch_sequence(fini_names), *main_exit_points,
                                  BPatch_callAfter, BPatch_firstSnippet);
    }
    else if(appThread)
    {
        for(auto itr : init_names)
            appThread->oneTimeCode(*itr);
    }

    auto instr_procedures = [&](const procedure_vec_t& procedures) {
        for(auto itr : procedures)
        {
            char modname[MUTNAMELEN];
            char fname[FUNCNAMELEN];

            BPatch_module* mod = itr->getModule();
            if(mod)
                mod->getName(modname, MUTNAMELEN);
            else
                itr->getModuleName(modname, MUTNAMELEN);

            if(strstr(modname, "libdyninstAPI_RT"))
                continue;

            if(module_constraint(modname) || !process_file_for_instrumentation(modname))
            {
                verbprintf(1, "Skipping constrained module: '%s'\n", modname);
                continue;
            }

            itr->getName(fname, FUNCNAMELEN);

            if(!itr->isInstrumentable())
            {
                verbprintf(1, "Skipping uninstrumentable function: %s\n", fname);
                continue;
            }

            auto name = get_func_file_line_info(appImage, itr);

            if(name.get().empty())
            {
                verbprintf(1, "Skipping function [empty name]: %s\n", fname);
                continue;
            }

            if(routine_constraint(fname))
            {
                verbprintf(1, "Skipping function [constrained]: %s\n", fname);
                continue;
            }

            if(routine_constraint(name.m_name.c_str()))
            {
                verbprintf(1, "Skipping function [constrained]: %s\n",
                           name.m_name.c_str());
                continue;
            }

            if(!instrument_entity(name.m_name))
            {
                verbprintf(1, "Skipping function [excluded]: %s\n", name.m_name.c_str());
                continue;
            }

            if(!instrument_entity(name.get()))
            {
                verbprintf(1, "Skipping function [excluded]: %s\n", name.get().c_str());
                continue;
            }

            if(is_static_exe && has_debug_info && strcmp(fname, "_fini") != 0 &&
               strcmp(modname, "DEFAULT_MODULE") == 0)
            {
                verbprintf(1, "Skipping function [DEFAULT_MODULE]: %s\n", fname);
                continue;
            }

            if(instrumented_procedures.find(name.get()) != instrumented_procedures.end())
            {
                verbprintf(1, "Skipping function [duplicate]: %s\n", name.get().c_str());
                continue;
            }

            verbprintf(0, "Instrumenting |> [ %s ]\n", name.m_name.c_str());

            instrumented_modules.insert(modname);
            instrumented_procedures.insert(name.get());

            insert_instr(addr_space, itr, entr_trace, BPatch_entry, name, nullptr,
                         nullptr);
            insert_instr(addr_space, itr, exit_trace, BPatch_exit, name, nullptr,
                         nullptr);

            if(loop_level_instr)
            {
                verbprintf(0, "Instrumenting at the loop level: %s\n", fname);

                flow_graph_t*    flow = itr->getCFG();
                basic_loop_vec_t basic_loop;
                if(flow)
                    flow->getOuterLoops(basic_loop);
                for(auto litr : basic_loop)
                {
                    auto lname = get_loop_file_line_info(appImage, itr, flow, litr);
                    insert_instr(addr_space, itr, entr_trace, BPatch_entry, lname, flow,
                                 litr);
                    insert_instr(addr_space, itr, exit_trace, BPatch_exit, lname, flow,
                                 litr);
                }
            }
        }
    };

    // finalize insertion
    if(binary_rewrite)
        addr_space->finalizeInsertionSet(false, nullptr);

    if(is_static_exe)
    {
        bool loadResult = load_dependent_libraries(addr_space, bindings);
        if(!loadResult)
        {
            fprintf(stderr, "Failed to load dependent libraries\n");
            throw std::runtime_error("Failed to load dependent libraries");
        }
    }

    verbprintf(2, "Before modules loop\n");
    for(auto& m : modules)
    {
        char modname[1024];
        m->getName(modname, 1024);
        if(strstr(modname, "libdyninstAPI_RT"))
            continue;

        if(!m->getProcedures())
        {
            verbprintf(1, "Skipping module w/ no procedures: '%s'\n", modname);
            continue;
        }

        verbprintf(0, "Parsing module: %s\n", modname);
        BPatch_Vector<BPatch_function*>* p = m->getProcedures();
        if(!p)
            continue;

        instr_procedures(*p);
    }

    dump_info("instrumented_modules.txt", instrumented_modules, 1);
    dump_info("instrumented_functions.txt", instrumented_procedures, 1);

    int code = -1;
    if(binary_rewrite)
    {
        char cwd[FUNCNAMELEN];
        auto ret = getcwd(cwd, FUNCNAMELEN);
        consume_parameters(ret);

        bool success = appBinary->writeFile(outfile);
        code         = (success) ? EXIT_SUCCESS : EXIT_FAILURE;
        if(success)
            printf("\nThe instrumented executable image is stored in '%s/%s'\n", cwd,
                   outfile);
    }
    else
    {
        printf("Executing...\n");

        bpatch->setDebugParsing(false);
        bpatch->setTypeChecking(false);
        bpatch->setDelayedParsing(true);
        bpatch->setInstrStackFrames(false);
        bpatch->setLivenessAnalysis(false);

        verbprintf(4, "Registering fork callbacks...\n");
        auto _prefork  = bpatch->registerPreForkCallback(&timemory_fork_callback);
        auto _postfork = bpatch->registerPostForkCallback(&timemory_fork_callback);

        auto _wait_exec = [&]() {
            while(!appThread->isTerminated())
            {
                verbprintf(3, "Continuing execution...\n");
                appThread->continueExecution();
                verbprintf(4, "Process is not terminated...\n");
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));
                bpatch->waitForStatusChange();
                verbprintf(4, "Process status change...\n");
                if(appThread->isStopped())
                {
                    verbprintf(4, "Process is stopped, continuing execution...\n");
                    if(!appThread->continueExecution())
                    {
                        fprintf(stderr, "continueExecution failed\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        };

        verbprintf(4, "Entering wait for status change mode...\n");
        _wait_exec();

        if(appThread->terminationStatus() == ExitedNormally)
        {
            if(appThread->isTerminated())
                printf("\nEnd of timemory-run\n");
            else
                _wait_exec();
        }
        else if(appThread->terminationStatus() == ExitedViaSignal)
        {
            auto sign = appThread->getExitSignal();
            fprintf(stderr, "\nApplication exited with signal: %i\n", int(sign));
        }

        code = appThread->getExitCode();
        consume_parameters(_prefork, _postfork);
    }

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
//
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

            verbprintf(0, "trying to read collection file @ %s...", fpath.c_str());
            std::ifstream ifs(fpath.c_str());

            if(!ifs)
            {
                verbprintf(0, "trying to read collection file @ %s...", fpath.c_str());
                fpath = TIMEMORY_JOIN("/", itr, to_lower(fname));
                ifs.open(fpath.c_str());
            }

            if(ifs)
            {
                verbprintf(0, ">    reading collection file @ %s...", fpath.c_str());
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
    auto is_include = [&](bool _if_empty) {
        if(file_include.empty())
            return _if_empty;
        for(auto& itr : file_include)
        {
            if(std::regex_search(file_name, itr))
                return true;
        }
        return false;
    };

    auto is_exclude = [&]() {
        for(auto& itr : file_exclude)
        {
            if(std::regex_search(file_name, itr))
            {
                verbprintf(2, "Excluding module [user-regex] : '%s'...\n",
                           file_name.c_str());
                return true;
            }
        }
        return false;
    };

    auto _user_include = is_include(false) && !is_exclude();

    if(_user_include)
    {
        verbprintf(2, "Including module [user-regex] : '%s'...\n", file_name.c_str());
        return true;
    }

    std::string ext_str = (binary_rewrite) ? "\\.(C|S)$" : "\\.(c|C|S)$";
    std::regex  ext_regex(ext_str);
    std::regex  sys_regex("^(s|k|e|w)_[A-Za-z_0-9\\-]+\\.(c|C)$");
    std::regex  userlib_regex("^lib(timemory|caliper|gotcha|papi|cupti|TAU|likwid|"
                             "profiler|tcmalloc|dyninst|pfm|nvtx|upcxx|pthread)");
    std::regex  corelib_regex("^lib(rt-|m-|dl-|m-|util-|gcc|c-|stdc++|c++|python)");
    // these are all due to TAU
    std::regex prefix_regex("^(Tau|Profiler|Rts|Papi|Py|Comp_xl\\.cpp|Comp_gnu\\.cpp|"
                            "UserEvent\\.cpp|FunctionInfo\\.cpp|PthreadLayer\\.cpp|"
                            "Comp_intel[0-9]\\.cpp|Tracer\\.cpp)");

    if(check_if_timemory_source_file(file_name))
    {
        verbprintf(3, "Excluding instrumentation [timemory source] : '%s'...\n",
                   file_name.c_str());
        return false;
    }

    if(std::regex_search(file_name, ext_regex))
    {
        verbprintf(3, "Excluding instrumentation [file extension] : '%s'...\n",
                   file_name.c_str());
        return false;
    }

    if(std::regex_search(file_name, sys_regex))
    {
        verbprintf(3, "Excluding instrumentation [system library] : '%s'...\n",
                   file_name.c_str());
        return false;
    }

    if(std::regex_search(file_name, corelib_regex))
    {
        verbprintf(3, "Excluding instrumentation [core library] : '%s'...\n",
                   file_name.c_str());
        return false;
    }

    if(std::regex_search(file_name, userlib_regex))
    {
        verbprintf(3, "Excluding instrumentation [timemory library] : '%s'...\n",
                   file_name.c_str());
        return false;
    }

    if(std::regex_search(file_name, prefix_regex))
    {
        verbprintf(3, "Excluding instrumentation [TAU] : '%s'...\n", file_name.c_str());
        return false;
    }

    bool use = is_include(true) && !is_exclude();
    if(use)
    {
        static std::set<std::string> already_reported;
        if(already_reported.count(file_name) == 0)
        {
            verbprintf(1, "%s |> [ %s ]\n", __FUNCTION__, file_name.c_str());
            already_reported.insert(file_name);
        }
    }
    return use;
}

//======================================================================================//

bool
instrument_entity(const std::string& function_name)
{
    auto is_include = [&](bool _if_empty) {
        if(func_include.empty())
            return _if_empty;
        for(auto& itr : func_include)
        {
            if(std::regex_search(function_name, itr))
                return true;
        }
        return false;
    };

    auto is_exclude = [&]() {
        for(auto& itr : func_exclude)
        {
            if(std::regex_search(function_name, itr))
            {
                verbprintf(2, "Excluding function [user-regex] : '%s'...\n",
                           function_name.c_str());
                return true;
            }
        }
        return false;
    };

    auto _user_include = is_include(false) && !is_exclude();

    if(_user_include)
    {
        verbprintf(2, "Including function [user-regex] : '%s'...\n",
                   function_name.c_str());
        return true;
    }

    std::regex exclude("(timemory|tim::|cereal|N3tim|MPI_Init|MPI_Finalize|::__[A-Za-z]|"
                       "std::max|std::min|std::fill|std::forward|std::get|dyninst)");
    std::regex leading("^(_init|_fini|__|_dl_|_start|_exit|frame_dummy|\\(\\(|\\(__|_"
                       "GLOBAL|targ|PMPI_|new|delete|std::allocator|std::move|nvtx|gcov|"
                       "main\\.cold\\.|TAU|tau|Tau|dyn|RT|_IO|dl|sys|pthread|posix)");
    std::regex stlfunc("^std::");
    std::set<std::string> whole = { "malloc", "free", "init", "fini", "_init", "_fini" };

    if(!stl_func_instr && std::regex_search(function_name, stlfunc))
    {
        verbprintf(3, "Excluding function [stl] : '%s'...\n", function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude))
    {
        verbprintf(3, "Excluding function [critical, any match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found at the start of the function name
    if(std::regex_search(function_name, leading))
    {
        verbprintf(3, "Excluding function [critical, leading match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    if(whole.count(function_name) > 0)
    {
        verbprintf(3, "Excluding function [critical, whole match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    bool use = is_include(true) && !is_exclude();
    if(use)
        verbprintf(2, "Including function [user-regex] : '%s'...\n",
                   function_name.c_str());

    return use;
}

//======================================================================================//
// insert_instr -- generic insert instrumentation function
//
void
insert_instr(BPatch_addressSpace* mutatee, BPatch_function* funcToInstr,
             BPatch_function* traceFunc, BPatch_procedureLocation traceLoc,
             function_signature& name, BPatch_flowGraph* cfGraph,
             BPatch_basicBlockLoop* loopToInstrument)
{
    BPatch_module* module = funcToInstr->getModule();
    if(!module)
        return;

    std::string _name = name.get();

    BPatch_Vector<BPatch_point*>* _points = nullptr;

    auto _trace_args = timemory_call_expr(_name);
    auto _trace      = _trace_args.get(traceFunc);

    if(cfGraph && loopToInstrument)
    {
        if(traceLoc == BPatch_entry)
            _points = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, loopToInstrument);
        else if(traceLoc == BPatch_exit)
            _points = cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);
    }
    else
    {
        _points = funcToInstr->findPoint(traceLoc);
    }

    if(_points == nullptr)
        return;
    if(_points->empty())
        return;

    /*
    if(loop_level_instr)
    {
        BPatch_flowGraph*                     flow = funcToInstr->getCFG();
        BPatch_Vector<BPatch_basicBlockLoop*> basicLoop;
        flow->getOuterLoops(basicLoop);
        for(auto litr = basicLoop.begin(); litr != basicLoop.end(); ++litr)
        {
            BPatch_Vector<BPatch_point*>* _tmp;
            if(traceLoc == BPatch_entry)
                _tmp = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, *litr);
            else if(traceLoc == BPatch_exit)
                _tmp = cfGraph->findLoopInstPoints(BPatch_locLoopExit, *litr);
            if(!_tmp)
                continue;
            for(auto& itr : *_tmp)
                _points->push_back(itr);
        }
    }
    */

    // verbprintf(0, "Instrumenting |> [ %s ]\n", name.m_name.c_str());

    for(auto& itr : *_points)
    {
        if(!itr)
            continue;
        else if(traceLoc == BPatch_entry)
            mutatee->insertSnippet(*_trace, *itr, BPatch_callBefore, BPatch_firstSnippet);
        else if(traceLoc == BPatch_exit)
            mutatee->insertSnippet(*_trace, *itr, BPatch_callAfter, BPatch_firstSnippet);
        else
            mutatee->insertSnippet(*_trace, *itr);
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

    if((strcmp(fname, "DEFAULT_MODULE") == 0) || (strcmp(fname, "LIBRARY_MODULE") == 0) ||
       ((fname[len - 2] == '.') && (fname[len - 1] == 'c')) ||
       ((fname[len - 2] == '.') && (fname[len - 1] == 'C')) ||
       ((fname[len - 3] == '.') && (fname[len - 2] == 'c') && (fname[len - 1] == 'c')) ||
       ((fname[len - 4] == '.') && (fname[len - 3] == 'c') && (fname[len - 2] == 'p') &&
        (fname[len - 1] == 'p')) ||
       ((fname[len - 4] == '.') && (fname[len - 3] == 'f') && (fname[len - 2] == '9') &&
        (fname[len - 1] == '0')) ||
       ((fname[len - 4] == '.') && (fname[len - 3] == 'F') && (fname[len - 2] == '9') &&
        (fname[len - 1] == '0')) ||
       ((fname[len - 2] == '.') && (fname[len - 1] == 'F')) ||
       ((fname[len - 2] == '.') && (fname[len - 1] == 'f')))
    {
        //((fname[len-3] == '.') && (fname[len-2] == 's') && (fname[len-1] == 'o'))||
        return false;
    }

    if(process_file_for_instrumentation(std::string(fname)))
        return false;

    // do not instrument
    return true;
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
load_dependent_libraries(address_space_t* bedit, char* bindings)
{
    // Order of load matters, just like command line arguments to a standalone linker

    char deplibs[1024];
    char bindir[] = TIMEMORY_BIN_DIR;
    char cmd[1024];
    verbprintf(0, "Inside load_dependent_libraries: bindings=%s\n", bindings);
    sprintf(cmd, "%s/timemory_show_libs %s/../lib/Makefile.timemory%s", bindir, bindir,
            bindings);
    verbprintf(0, "cmd = %s\n", cmd);
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
            verbprintf(0, "LOADING %s\n", &deplibs[1]);
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

//======================================================================================//
//
static inline void
consume()
{
    consume_parameters(initialize_expr, error_print, debug_print, expect_error);
    if(false)
        timemory_thread_exit(nullptr, ExitedNormally);
}
