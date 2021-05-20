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

#include <sys/stat.h>
#include <sys/types.h>

static bool                                       is_driver  = false;
static size_t                                     batch_size = 50;
static strset_t                                   extra_libs = {};
static std::vector<std::pair<uint64_t, string_t>> hash_ids;
static std::map<string_t, bool>                   use_stubs;
static std::map<string_t, procedure_t*>           beg_stubs;
static std::map<string_t, procedure_t*>           end_stubs;
static strvec_t                                   init_stub_names;
static strvec_t                                   fini_stub_names;
static strset_t                                   used_stub_names;
static std::vector<call_expr_pointer_t>           env_variables;
static std::map<string_t, call_expr_pointer_t>    beg_expr;
static std::map<string_t, call_expr_pointer_t>    end_expr;
static const auto                                 npos_v          = string_t::npos;
static string_t                                   instr_mode      = "trace";
static string_t                                   instr_push_func = "timemory_push_trace";
static string_t                                   instr_pop_func  = "timemory_pop_trace";
static string_t instr_push_hash = "timemory_push_trace_hash";
static string_t instr_pop_hash  = "timemory_pop_trace_hash";

std::string
get_absolute_exe_filepath(std::string exe_name);

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
    tim::set_env<string_t>("DYNINSTAPI_RT_LIB", DYNINST_API_RT, 0);
#endif

#if defined(TIMEMORY_USE_MPI)
    auto _dmp_argc = 1;
    auto _dmp_argv = argv;
    tim::dmp::initialize(_dmp_argc, _dmp_argv);
#endif

    argv0                  = argv[0];
    auto env_collect_paths = tim::get_env<string_t>("TIMEMORY_COLLECTION_PATH", "");
    prefix_collection_path(env_collect_paths, collection_paths);

    bpatch                              = std::make_shared<patch_t>();
    bool                  is_attached   = false;
    address_space_t*      addr_space    = nullptr;
    string_t              mutname       = "";
    string_t              outfile       = "";
    std::vector<string_t> inputlib      = { "" };
    std::vector<string_t> libname       = {};
    std::vector<string_t> sharedlibname = {};
    std::vector<string_t> staticlibname = {};
    tim::process::id_t    _pid          = -1;

    bpatch->setTypeChecking(true);
    bpatch->setSaveFPR(true);
    bpatch->setDelayedParsing(true);
    bpatch->setInstrStackFrames(true);
    bpatch->setLivenessAnalysis(false);
    bpatch->setBaseTrampDeletion(false);
    bpatch->setTrampRecursive(false);
    bpatch->setMergeTramp(false);

    std::set<std::string> dyninst_defs = { "TypeChecking", "SaveFPR", "DelayedParsing",
                                           "InstrStackFrames" };

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
        string_t _arg = argv[i];
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
            mutname = _cmdv[0];
            break;
        }
        else
        {
            copy_str(_argv[i], argv[i]);
        }
    }

    auto cmd_string = [](int _ac, char** _av) {
        stringstream_t ss;
        for(int i = 0; i < _ac; ++i)
            ss << _av[i] << " ";
        return ss.str();
    };

    if(_cmdc > 0 && !mutname.empty())
    {
        auto resolved_mutname = get_absolute_exe_filepath(mutname);
        if(resolved_mutname != mutname)
        {
            mutname = resolved_mutname;
            delete _cmdv[0];
            copy_str(_cmdv[0], resolved_mutname.c_str());
        }
    }

    if(verbose_level > 1 && tim::dmp::rank() == 0)
    {
        std::cout << "[original]: " << cmd_string(argc, argv) << std::endl;
        std::cout << "[cfg-args]: " << cmd_string(_argc, _argv) << std::endl;
    }

    if(_cmdc > 0 && tim::dmp::rank() == 0)
        std::cout << "\n [command]: " << cmd_string(_cmdc, _cmdv) << "\n\n";

    if(_cmdc > 0)
        cmdv0 = _cmdv[0];

    std::stringstream jump_description;
    jump_description
        << "Instrument with function pointers in TIMEMORY_JUMP_LIBRARY (default: "
        << tim::get_env<string_t>("TIMEMORY_JUMP_LIBRARY", "jump/libtimemory.so") << ")";

    // now can loop through the options.  If the first character is '-', then we know
    // we have an option.  Check to see if it is one of our options and process it. If
    // it is unrecognized, then set the errflag to report an error.  When we come to a
    // non '-' charcter, then we must be at the application name.
    using parser_t = tim::argparse::argument_parser;
    parser_t parser("timemory-run");

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
        .names({ "-j", "--jump" })
        .description(jump_description.str())
        .count(0);
    parser.add_argument()
        .names({ "-s", "--stubs" })
        .description("Instrument with library stubs for LD_PRELOAD")
        .count(0);
    parser.add_argument()
        .names({ "-L", "--library" })
        .description(
            "Libraries with instrumentation routines (default: \"libtimemory\")");
    parser.add_argument()
        .names({ "-S", "--stdlib" })
        .description("Enable instrumentation of C++ standard library functions.")
        .count(0);
    parser.add_argument()
        .names({ "--cstdlib" })
        .description("Enable instrumentation of C standard library functions.")
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
        .names({ "--env" })
        .description(
            "Environment variables to add to the runtime in form VARIABLE=VALUE");
    parser.add_argument()
        .names({ "--prefer" })
        .description("Prefer this library types when available")
        .choices({ "shared", "static" })
        .count(1);
    parser.add_argument({ "--driver" }, "Force main or _init/_fini instrumentation")
        .count(0)
        .action([](auto&) { is_driver = true; });
    parser
        .add_argument({ "--mpi" },
                      "Enable MPI support (requires timemory built w/ MPI and GOTCHA "
                      "support)")
        .count(0);
    parser.add_argument({ "--label" }, "Labeling info for functions")
        .choices({ "file", "line", "return", "args" });
    parser.add_argument({ "--mpip" }, "Enable MPI profiling via GOTCHA").count(0);
    parser.add_argument({ "--ompt" }, "Enable OpenMP profiling via OMPT").count(0);
    parser.add_argument({ "--load" },
                        "Supplemental instrumentation library names w/o extension (e.g. "
                        "'libinstr' for 'libinstr.so' or 'libinstr.a')");
    parser.add_argument(
        { "--init-functions" },
        "Initialization function(s) for supplemental instrumentation libraries");
    parser.add_argument(
        { "--fini-functions" },
        "Finalization function(s) for supplemental instrumentation libraries");
    parser
        .add_argument(
            { "-b", "--batch-size" },
            "Dyninst supports batch insertion of multiple points. If one large batch "
            "insertion fails, this value will be used to create smaller batches")
        .count(1)
        .action([](parser_t& p) { batch_size = p.get<size_t>("batch-size"); });

    if(_cmdc == 0)
    {
        parser.add_argument()
            .names({ "-c", "--command" })
            .description("Input executable and arguments (if '-- <CMD>' not provided)")
            .count(1);
    }

    parser
        .add_argument({ "--dyninst-options" },
                      "Advanced dyninst options: BPatch::set<OPTION>(bool), e.g. "
                      "bpatch->setTrampRecursive(true)")
        .choices({ "TypeChecking", "SaveFPR", "DelayedParsing", "InstrStackFrames",
                   "TrampRecursive", "MergeTramp", "BaseTrampDeletion" });

    string_t extra_help = "-- <CMD> <ARGS>";
    auto     err        = parser.parse(_argc, _argv);
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
    {
        verbose_level = 256;
        debug_print   = true;
    }

    if(parser.exists("m"))
        main_fname = parser.get<string_t>("m");

    if(parser.exists("l"))
        loop_level_instr = true;

    if(_cmdc == 0 && parser.exists("c"))
    {
        auto keys = parser.get<strvec_t>("c");
        if(keys.empty())
        {
            parser.print_help(extra_help);
            return EXIT_FAILURE;
        }
        keys.at(0) = get_absolute_exe_filepath(keys.at(0));
        mutname    = keys.at(0);
        _cmdc      = keys.size();
        _cmdv      = new char*[_cmdc];
        for(int i = 0; i < _cmdc; ++i)
        {
            copy_str(_cmdv[i], keys.at(i).c_str());
        }
    }

    if(parser.exists("o"))
    {
        binary_rewrite = true;
        outfile        = parser.get<string_t>("o");
    }

    if(!parser.exists("L"))
    {
        for(auto& itr : inputlib)
            itr += "libtimemory";

        if(parser.exists("s"))
        {
            for(auto& itr : inputlib)
                itr += "-stubs";
        }
        else if(parser.exists("j"))
        {
            for(auto& itr : inputlib)
                itr += "-jump";
        }
    }
    else
    {
        inputlib = parser.get<strvec_t>("L");
    }

    if(parser.exists("S"))
        stl_func_instr = true;

    if(parser.exists("cstdlib"))
        cstd_func_instr = true;

    if(parser.exists("mpi"))
        use_mpi = true;

    if(parser.exists("mpip"))
        use_stubs["mpip"] = true;
    else
        use_stubs["mpip"] = false;

    if(parser.exists("ompt"))
        use_stubs["ompt"] = true;
    else
        use_stubs["ompt"] = false;

    if(parser.exists("p"))
        _pid = parser.get<int>("p");

    if(parser.exists("d"))
    {
        auto _components   = parser.get<strvec_t>("default-components");
        default_components = "";
        for(size_t i = 0; i < _components.size(); ++i)
        {
            if(_components.at(i) == "none")
            {
                default_components = "none";
                break;
            }
            default_components += _components.at(i);
            if(i + 1 < _components.size())
                default_components += ",";
        }
        if(default_components == "none")
            default_components = "";
        else
        {
            auto _strcomp = parser.get<std::string>("d");
            if(!_strcomp.empty() && default_components.empty())
                default_components = _strcomp;
        }
    }

    if(parser.exists("M"))
    {
        instr_mode      = parser.get<string_t>("M");
        instr_push_func = "timemory_push_" + instr_mode;
        instr_push_hash = "timemory_push_" + instr_mode + "_hash";
        instr_pop_func  = "timemory_pop_" + instr_mode;
        instr_pop_hash  = "timemory_pop_" + instr_mode + "_hash";
    }

    if(parser.exists("prefer"))
        prefer_library = parser.get<string_t>("prefer");

    if(parser.exists("load"))
    {
        auto _load = parser.get<strvec_t>("load");
        for(auto itr : _load)
            extra_libs.insert(itr);
    }

    if(parser.exists("label"))
    {
        auto _labels = parser.get<strvec_t>("label");
        for(auto itr : _labels)
        {
            if(std::regex_match(itr, std::regex("file", std::regex_constants::icase)))
                use_file_info = true;
            else if(std::regex_match(itr,
                                     std::regex("return", std::regex_constants::icase)))
                use_return_info = true;
            else if(std::regex_match(itr,
                                     std::regex("args", std::regex_constants::icase)))
                use_args_info = true;
            else if(std::regex_match(itr,
                                     std::regex("line", std::regex_constants::icase)))
                use_line_info = true;
        }
    }

    init_stub_names = parser.get<strvec_t>("init-functions");
    fini_stub_names = parser.get<strvec_t>("fini-functions");
    auto env_vars   = parser.get<strvec_t>("env");

    //----------------------------------------------------------------------------------//
    //
    //                              REGEX OPTIONS
    //
    //----------------------------------------------------------------------------------//
    //
    //  Helper function for adding regex expressions
    //
    auto add_regex = [](auto& regex_array, const string_t& regex_expr) {
        if(!regex_expr.empty())
            regex_array.push_back(std::regex(regex_expr, regex_opts));
    };

    add_regex(func_include, tim::get_env<string_t>("TIMEMORY_REGEX_INCLUDE", ""));
    add_regex(func_exclude, tim::get_env<string_t>("TIMEMORY_REGEX_EXCLUDE", ""));

    if(parser.exists("R"))
    {
        auto keys = parser.get<strvec_t>("R");
        for(const auto& itr : keys)
            add_regex(func_include, itr);
    }

    if(parser.exists("E"))
    {
        auto keys = parser.get<strvec_t>("E");
        for(const auto& itr : keys)
            add_regex(func_exclude, itr);
    }

    if(parser.exists("MI"))
    {
        auto keys = parser.get<strvec_t>("MI");
        for(const auto& itr : keys)
            add_regex(file_include, itr);
    }

    if(parser.exists("ME"))
    {
        auto keys = parser.get<strvec_t>("ME");
        for(const auto& itr : keys)
            add_regex(file_exclude, itr);
    }

    //----------------------------------------------------------------------------------//
    //
    //                              COLLECTION OPTIONS
    //
    //----------------------------------------------------------------------------------//
    //
    if(parser.exists("P"))
    {
        auto tmp = parser.get<strvec_t>("P");
        for(auto itr : tmp)
            prefix_collection_path(itr, collection_paths);
    }

    if(parser.exists("C"))
    {
        auto tmp = parser.get<strvec_t>("C");
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
    //                              DYNINST OPTIONS
    //
    //----------------------------------------------------------------------------------//

    int dyninst_verb = 1;
    if(parser.exists("dyninst-options"))
    {
        dyninst_defs = parser.get<std::set<std::string>>("dyninst-options");
        dyninst_verb = 0;
    }

    auto get_dyninst_option = [&](const std::string& _opt) {
        bool _ret = dyninst_defs.find(_opt) != dyninst_defs.end();
        verbprintf(dyninst_verb, "[dyninst-option]> %-20s = %4s\n", _opt.c_str(),
                   (_ret) ? "on" : "off");
        return _ret;
    };

    bpatch->setTypeChecking(get_dyninst_option("TypeChecking"));
    bpatch->setSaveFPR(get_dyninst_option("SaveFPR"));
    bpatch->setDelayedParsing(get_dyninst_option("DelayedParsing"));
    bpatch->setInstrStackFrames(get_dyninst_option("InstrStackFrames"));

    bpatch->setTrampRecursive(get_dyninst_option("TrampRecursive"));
    bpatch->setMergeTramp(get_dyninst_option("MergeTramp"));
    bpatch->setBaseTrampDeletion(get_dyninst_option("BaseTrampDeletion"));

    //----------------------------------------------------------------------------------//
    //
    //                              MAIN
    //
    //----------------------------------------------------------------------------------//

    if(_cmdc == 0)
    {
        parser.print_help(extra_help);
        fprintf(stderr, "\nError! No command for dynamic instrumentation. Use "
                        "\n\ttimemory-run <OPTIONS> -- <COMMAND> <ARGS>\nE.g. "
                        "\n\ttimemory-run -o foo.inst -- ./foo\nwill output an "
                        "instrumented version of 'foo' executable to 'foo.inst'\n");
        return EXIT_FAILURE;
    }

    verbprintf(0, "instrumentation target: %s\n", mutname.c_str());

    // did we load a library?  if not, load the default
    auto generate_libnames = [](auto& _targ, const auto& _base,
                                const std::set<string_t>& _ext) {
        for(const auto& bitr : _base)
            for(const auto& eitr : _ext)
                _targ.push_back(bitr + eitr);
    };

    generate_libnames(libname, inputlib, { "" });
    generate_libnames(sharedlibname, inputlib, { ".so" });
    generate_libnames(staticlibname, inputlib, { ".a" });

    // Register a callback function that prints any error messages
    bpatch->registerErrorCallback(error_func_real);

    //----------------------------------------------------------------------------------//
    //
    //  Start the instrumentation procedure by opening a file for binary editing,
    //  attaching to a running process, or starting a process
    //
    //----------------------------------------------------------------------------------//

    addr_space =
        timemory_get_address_space(bpatch, _cmdc, _cmdv, binary_rewrite, _pid, mutname);

    if(!addr_space)
    {
        fprintf(stderr, "[timemory-run]> Error! address space for dynamic "
                        "instrumentation was not created\n");
        exit(EXIT_FAILURE);
    }

    process_t*     app_thread = nullptr;
    binary_edit_t* app_binary = nullptr;

    // get image
    verbprintf(1, "Getting the address space image, modules, and procedures...\n");
    image_t*                  app_image     = addr_space->getImage();
    bpvector_t<module_t*>*    app_modules   = app_image->getModules();
    bpvector_t<procedure_t*>* app_functions = app_image->getProcedures();
    bpvector_t<module_t*>     modules;
    bpvector_t<procedure_t*>  functions;

    //----------------------------------------------------------------------------------//
    //
    //  Generate a log of all the available procedures and modules
    //
    //----------------------------------------------------------------------------------//
    std::set<std::string> module_names;

    if(app_modules && app_modules->size() > 0)
    {
        modules = *app_modules;
        for(auto itr : *app_modules)
        {
            auto procedures = itr->getProcedures();
            if(procedures)
            {
                for(auto pitr : *procedures)
                {
                    auto _modfn = module_function(itr, pitr);
                    module_names.insert(_modfn.module);
                    available_module_functions.insert(std::move(_modfn));
                }
            }
        }
    }
    else
    {
        verbprintf(0, "Warning! No modules in application...\n");
    }

    if(app_functions && app_functions->size() > 0)
    {
        functions = *app_functions;
        for(auto itr : *app_functions)
        {
            module_t* mod = itr->getModule();
            if(mod)
            {
                auto _modfn = module_function(mod, itr);
                module_names.insert(_modfn.module);
                available_module_functions.insert(std::move(_modfn));
            }
        }
    }
    else
    {
        verbprintf(0, "Warning! No functions in application...\n");
    }

    verbprintf(0, "Module size before loading instrumentation library: %lu\n",
               (long unsigned) app_modules->size());

    if(debug_print || verbose_level > 1)
    {
        module_function::reset_width();
        for(const auto& itr : available_module_functions)
            module_function::update_width(itr);

        auto mwid = module_function::get_width().at(0);
        auto ncol = 240 / std::min<size_t>(mwid, 240);
        std::cout << "### MODULES ###\n| ";
        for(size_t i = 0; i < module_names.size(); ++i)
        {
            auto itr = module_names.begin();
            std::advance(itr, i);
            std::cout << std::setw(mwid) << *itr << " | ";
            if(i % ncol == ncol - 1)
                std::cout << "\n| ";
        }
        std::cout << '\n' << std::endl;
    }

    dump_info("available_module_functions.txt", available_module_functions, 1);

    //----------------------------------------------------------------------------------//
    //
    //  Get the derived type of the address space
    //
    //----------------------------------------------------------------------------------//

    is_static_exe = addr_space->isStaticExecutable();

    if(binary_rewrite)
        app_binary = static_cast<BPatch_binaryEdit*>(addr_space);
    else
        app_thread = static_cast<BPatch_process*>(addr_space);

    is_attached = (_pid >= 0 && app_thread);

    if(!app_binary && !app_thread)
    {
        fprintf(stderr, "No application thread or binary!...\n");
        throw std::runtime_error("Nullptr to BPatch_binaryEdit* and BPatch_process*");
    }

    //----------------------------------------------------------------------------------//
    //
    //  Helper functions for library stuff
    //
    //----------------------------------------------------------------------------------//

    auto load_library = [addr_space](const std::vector<string_t>& _libnames) {
        bool result = false;
        // track the tried library names
        string_t _tried_libs;
        for(const auto& _libname : _libnames)
        {
            verbprintf(0, "loading library: '%s'...\n", _libname.c_str());
            result = addr_space->loadLibrary(_libname.c_str());
            verbprintf(1, "loadLibrary(%s) result = %s\n", _libname.c_str(),
                       (result) ? "success" : "failure");
            if(result)
                break;
            _tried_libs += string_t("|") + _libname;
        }
        if(!result)
        {
            fprintf(stderr,
                    "Error: 'loadLibrary(%s)' failed.\nPlease ensure that the "
                    "library directory is in LD_LIBRARY_PATH environment variable "
                    "or absolute path is provided\n",
                    _tried_libs.substr(1).c_str());
            exit(EXIT_FAILURE);
        }
    };

    auto get_library_ext = [=](const std::vector<string_t>& linput) {
        auto lnames           = linput;
        auto _get_library_ext = [](string_t lname) {
            if(lname.find(".so") != string_t::npos ||
               lname.find(".a") == lname.length() - 2)
                return lname;
            if(!prefer_library.empty())
                return (lname +
                        ((prefer_library == "static" || is_static_exe) ? ".a" : ".so"));
            else
                return (lname + ((is_static_exe) ? ".a" : ".so"));
        };
        for(auto& lname : lnames)
            lname = _get_library_ext(lname);
        return lnames;
    };

    //----------------------------------------------------------------------------------//
    //
    //  find _init and _fini before loading instrumentation library!
    //  These will be used for initialization and finalization if main is not found
    //
    //----------------------------------------------------------------------------------//

    auto* _mutatee_init = find_function(app_image, "_init");
    auto* _mutatee_fini = find_function(app_image, "_fini");

    //----------------------------------------------------------------------------------//
    //
    //  Load the instrumentation libraries
    //
    //----------------------------------------------------------------------------------//

    load_library(get_library_ext(libname));

    if(use_stubs["mpip"] && !is_static_exe)
        load_library({ "libtimemory-mpip.so" });

    if(use_stubs["ompt"])
        load_library(get_library_ext({ "libtimemory-ompt" }));

    for(auto itr : extra_libs)
        load_library(get_library_ext({ itr }));

    //----------------------------------------------------------------------------------//
    //
    //  Find the primary functions that will be used for instrumentation
    //
    //----------------------------------------------------------------------------------//

    verbprintf(0, "Finding functions in image...\n");

    auto* main_func     = find_function(app_image, main_fname.c_str());
    auto* entr_trace    = find_function(app_image, instr_push_func.c_str());
    auto* exit_trace    = find_function(app_image, instr_pop_func.c_str());
    auto* entr_hash     = find_function(app_image, instr_push_hash.c_str());
    auto* exit_hash     = find_function(app_image, instr_pop_hash.c_str());
    auto* init_func     = find_function(app_image, "timemory_trace_init");
    auto* fini_func     = find_function(app_image, "timemory_trace_finalize");
    auto* env_func      = find_function(app_image, "timemory_trace_set_env");
    auto* mpi_func      = find_function(app_image, "timemory_trace_set_mpi");
    auto* hash_func     = find_function(app_image, "timemory_add_hash_id");
    auto* exit_func     = find_function(app_image, "exit", { "_exit" });
    auto* mpi_init_func = find_function(app_image, "MPI_Init", { "MPI_Init_thread" });
    auto* mpi_fini_func = find_function(app_image, "MPI_Finalize");

    if(!main_func && main_fname == "main")
        main_func = find_function(app_image, "_main");

    verbprintf(0, "Instrumenting with '%s' and '%s'...\n", instr_push_func.c_str(),
               instr_pop_func.c_str());

    if(mpi_init_func && mpi_fini_func)
        use_mpi = true;

    //----------------------------------------------------------------------------------//
    //
    //  Handle supplemental instrumentation library functions
    //
    //----------------------------------------------------------------------------------//

    auto add_instr_library = [&](const string_t& _name, const string_t& _beg,
                                 const string_t& _end) {
        verbprintf(3,
                   "Attempting to find instrumentation for '%s' via '%s' and '%s'...\n",
                   _name.c_str(), _beg.c_str(), _end.c_str());
        if(_beg.empty() || _end.empty())
            return false;
        auto* _beg_func = find_function(app_image, _beg);
        auto* _end_func = find_function(app_image, _end);
        if(_beg_func && _end_func)
        {
            use_stubs[_name] = true;
            beg_stubs[_name] = _beg_func;
            end_stubs[_name] = _end_func;
            used_stub_names.insert(_beg);
            used_stub_names.insert(_end);
            verbprintf(0, "Instrumenting '%s' via '%s' and '%s'...\n", _name.c_str(),
                       _beg.c_str(), _end.c_str());
            return true;
        }
        return false;
    };

    if(use_stubs["mpip"])
        add_instr_library("mpip", "timemory_register_mpip", "timemory_deregister_mpip");
    if(use_stubs["ompt"])
        add_instr_library("ompt", "timemory_register_ompt", "timemory_deregister_ompt");

    if(!extra_libs.empty())
    {
        verbprintf(2, "Adding extra libraries...\n");
    }

    for(const auto& itr : extra_libs)
    {
        string_t _name = itr;
        size_t   _pos  = _name.find_last_of('/');
        if(_pos != npos_v)
            _name = _name.substr(_pos + 1);
        _pos = _name.find('.');
        if(_pos != npos_v)
            _name = _name.substr(0, _pos);
        _pos = _name.find("libtimemory-");
        if(_pos != npos_v)
            _name = _name.erase(_pos, std::string("libtimemory-").length());
        _pos = _name.find("lib");
        if(_pos == 0)
            _name = _name.substr(_pos + std::string("lib").length());
        while((_pos = _name.find("-")) != npos_v)
            _name.replace(_pos, 1, "_");

        verbprintf(2,
                   "Supplemental instrumentation library '%s' is named '%s' after "
                   "removing everything before last '/', everything after first '.', and "
                   "'libtimemory-'...\n",
                   itr.c_str(), _name.c_str());

        use_stubs[_name] = false;

        string_t best_init_name = "";
        for(const auto& sitr : init_stub_names)
        {
            if(sitr.find(_name) != npos_v && used_stub_names.count(sitr) == 0)
            {
                verbprintf(
                    3, "Found possible match for '%s' instrumentation init: '%s'...\n",
                    _name.c_str(), sitr.c_str());
                best_init_name = sitr;
                break;
            }
        }

        string_t base_fini_name = "";
        for(const auto& sitr : fini_stub_names)
        {
            if(sitr.find(_name) != npos_v && used_stub_names.count(sitr) == 0)
            {
                verbprintf(
                    3, "Found possible match for '%s' instrumentation fini: '%s'...\n",
                    _name.c_str(), sitr.c_str());
                base_fini_name = sitr;
                break;
            }
        }

        if(add_instr_library(_name, best_init_name, base_fini_name))
            continue;

        // check user-specified signatures first
        for(const auto& bitr : init_stub_names)
        {
            if(used_stub_names.find(bitr) != used_stub_names.end())
                continue;
            for(const auto& fitr : fini_stub_names)
            {
                if(used_stub_names.find(fitr) != used_stub_names.end())
                    continue;
                if(add_instr_library(_name, bitr, fitr))
                    goto found_instr_functions;  // exit loop after match
            }
        }

        // check standard function signature if no user-specified matches
        if(add_instr_library(_name, TIMEMORY_JOIN("", "timemory_register_" + _name),
                             TIMEMORY_JOIN("", "timemory_deregister_" + _name)))
            continue;

    found_instr_functions:
        continue;
    }

    //----------------------------------------------------------------------------------//
    //
    //  Check for any issues finding the required functions
    //
    //----------------------------------------------------------------------------------//

    if(!main_func && is_driver)
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
    else if(!main_func && !is_driver)
    {
        verbprintf(0, "Warning! No main function and is not driver!\n");
    }

    using pair_t = std::pair<procedure_t*, string_t>;

    for(auto itr :
        { pair_t(main_func, main_fname), pair_t(entr_trace, instr_push_func),
          pair_t(exit_trace, instr_pop_func), pair_t(init_func, "timemory_trace_init"),
          pair_t(fini_func, "timemory_trace_finalize"),
          pair_t(env_func, "timemory_trace_set_env") })
    {
        if(itr.first == main_func && !is_driver)
            continue;
        if(!itr.first)
        {
            stringstream_t ss;
            ss << "Error! Couldn't find '" << itr.second.c_str() << "' function";
            fprintf(stderr, "[timemory-run]> %s\n", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }

    if(use_mpi && !(mpi_func || (mpi_init_func && mpi_fini_func)))
    {
        throw std::runtime_error("MPI support was requested but timemory was not built "
                                 "with MPI and GOTCHA support");
    }

    if(use_stubs["mpip"] && !(beg_stubs["mpip"] || end_stubs["mpip"]))
    {
        throw std::runtime_error("MPIP support was requested but could not find "
                                 "timemory_{register,deregister}_mpip functions");
    }

    if(use_stubs["ompt"] && !(beg_stubs["ompt"] || end_stubs["ompt"]))
    {
        throw std::runtime_error("OMPT support was requested but could not find "
                                 "timemory_{register,deregister}_ompt functions");
    }

    auto check_for_debug_info = [](bool& _has_debug_info, auto* _func) {
        // This heuristic guesses that debugging info is available if function
        // is not defined in the DEFAULT_MODULE
        if(_func && !_has_debug_info)
        {
            module_t* _module = _func->getModule();
            if(_module)
            {
                char moduleName[MUTNAMELEN];
                _module->getName(moduleName, MUTNAMELEN);
                if(strcmp(moduleName, "DEFAULT_MODULE") != 0)
                    _has_debug_info = true;
            }
        }
    };

    bool has_debug_info = false;
    check_for_debug_info(has_debug_info, main_func);
    check_for_debug_info(has_debug_info, _mutatee_init);
    check_for_debug_info(has_debug_info, _mutatee_fini);

    //----------------------------------------------------------------------------------//
    //
    //  Find the entry/exit point of either the main (if executable) or the _init
    //  and _fini functions of the library
    //
    //----------------------------------------------------------------------------------//

    bpvector_t<point_t*>* main_entr_points = nullptr;
    bpvector_t<point_t*>* main_exit_points = nullptr;

    if(main_func)
    {
        verbprintf(2, "Finding main function entry/exit... ");
        main_entr_points = main_func->findPoint(BPatch_entry);
        main_exit_points = main_func->findPoint(BPatch_exit);
        verbprintf(2, "Done\n");
    }
    else if(is_driver)
    {
        if(_mutatee_init)
        {
            verbprintf(2, "Finding init entry...\n");
            main_entr_points = _mutatee_init->findPoint(BPatch_entry);
        }
        if(_mutatee_fini)
        {
            verbprintf(2, "Finding fini exit...\n");
            main_exit_points = _mutatee_fini->findPoint(BPatch_exit);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Create the call arguments for the initialization and finalization routines
    //  and the snippets which are inserted using those arguments
    //
    //----------------------------------------------------------------------------------//

    // begin insertion
    if(binary_rewrite)
    {
        verbprintf(2, "Beginning insertion set...\n");
        addr_space->beginInsertionSet();
    }

    function_signature main_sign("int", "main", "", { "int", "char**" });
    if(main_func)
    {
        verbprintf(2, "Getting main function signature...\n");
        main_sign = get_func_file_line_info(main_func->getModule(), main_func);
        if(main_sign.m_params == "()")
            main_sign.m_params = "(int argc, char** argv)";
    }

    verbprintf(2, "Getting call expressions... ");

    auto main_call_args = timemory_call_expr(main_sign.get());
    auto init_call_args = timemory_call_expr(default_components, binary_rewrite, cmdv0);
    auto fini_call_args = timemory_call_expr();
    auto umpi_call_args = timemory_call_expr(use_mpi, is_attached);
    auto mode_call_args = timemory_call_expr("TIMEMORY_INSTRUMENTATION_MODE", instr_mode);
    auto mpie_init_args = timemory_call_expr("TIMEMORY_MPI_INIT", "OFF");
    auto mpie_fini_args = timemory_call_expr("TIMEMORY_MPI_FINALIZE", "OFF");
    auto trace_call_args =
        timemory_call_expr("TIMEMORY_TRACE_COMPONENTS", default_components);
    auto mpip_call_args =
        timemory_call_expr("TIMEMORY_MPIP_COMPONENTS", default_components);
    auto ompt_call_args =
        timemory_call_expr("TIMEMORY_OMPT_COMPONENTS", default_components);
    auto none_call_args = timemory_call_expr();

    verbprintf(2, "Done\n");
    verbprintf(2, "Getting call snippets... ");

    auto init_call = init_call_args.get(init_func);
    auto fini_call = fini_call_args.get(fini_func);
    auto exit_call = fini_call_args.get(exit_func);
    auto umpi_call = umpi_call_args.get(mpi_func);

    auto main_beg_call = main_call_args.get(entr_trace);
    auto main_end_call = main_call_args.get(exit_trace);

    auto trace_env_call = trace_call_args.get(env_func);
    auto mode_env_call  = mode_call_args.get(env_func);
    auto mpip_env_call  = mpip_call_args.get(env_func);
    auto ompt_env_call  = ompt_call_args.get(env_func);
    auto mpii_env_call  = mpie_init_args.get(env_func);
    auto mpif_env_call  = mpie_fini_args.get(env_func);

    verbprintf(2, "Done\n");

    for(const auto& itr : use_stubs)
    {
        if(beg_stubs[itr.first] && end_stubs[itr.first])
        {
            beg_expr[itr.first] = none_call_args.get(beg_stubs[itr.first]);
            end_expr[itr.first] = none_call_args.get(end_stubs[itr.first]);
        }
    }

    for(auto& itr : env_vars)
    {
        auto p = tim::delimit(itr, "=");
        if(p.size() != 2)
        {
            std::cerr << "Error! environment variable: " << itr
                      << " not in form VARIABLE=VALUE\n";
            throw std::runtime_error("Bad format");
        }
        auto _expr = timemory_call_expr(p.at(0), p.at(1));
        env_variables.push_back(_expr.get(env_func));
    }

    //----------------------------------------------------------------------------------//
    //
    //  Configure the initialization and finalization routines
    //
    //----------------------------------------------------------------------------------//

    if(trace_env_call)
        init_names.push_back(trace_env_call.get());
    if(mode_env_call)
        init_names.push_back(mode_env_call.get());
    if(mpii_env_call)
        init_names.push_back(mpii_env_call.get());
    if(mpif_env_call)
        init_names.push_back(mpif_env_call.get());
    if(use_stubs["mpip"] && mpip_env_call)
        init_names.push_back(mpip_env_call.get());
    if(use_stubs["ompt"] && ompt_env_call)
        init_names.push_back(ompt_env_call.get());

    for(auto itr : env_variables)
    {
        if(itr)
            init_names.push_back(itr.get());
    }

    for(const auto& itr : beg_expr)
    {
        if(itr.second)
        {
            verbprintf(1, "+ Adding %s instrumentation...\n", itr.first.c_str());
            init_names.push_back(itr.second.get());
        }
        else
        {
            verbprintf(1, "- Skipping %s instrumentation...\n", itr.first.c_str());
        }
    }

    if(use_mpi && umpi_call)
        init_names.push_back(umpi_call.get());

    if(init_call)
        init_names.push_back(init_call.get());

    if(binary_rewrite)
    {
        if(mpi_init_func && mpi_fini_func)
        {
            verbprintf(2, "Patching MPI init functions\n");
            insert_instr(addr_space, mpi_init_func, init_call, BPatch_exit, nullptr,
                         nullptr);
            insert_instr(addr_space, mpi_fini_func, fini_call, BPatch_entry, nullptr,
                         nullptr);
        }
        else
        {
            verbprintf(2, "Adding main begin and end snippets...\n");
            init_names.push_back(main_beg_call.get());
            fini_names.push_back(main_end_call.get());
        }
    }
    else if(app_thread)
    {
        verbprintf(2, "Patching main function\n");
        insert_instr(addr_space, main_func, init_call, BPatch_entry, nullptr, nullptr);
        if(!use_mpi)
        {
            insert_instr(addr_space, main_func, main_beg_call, BPatch_entry, nullptr,
                         nullptr);
            insert_instr(addr_space, main_func, main_end_call, BPatch_exit, nullptr,
                         nullptr);
        }
        insert_instr(addr_space, main_func, fini_call, BPatch_exit, nullptr, nullptr);
    }
    else
    {
        verbprintf(0, "No binary_rewrite and no app_thread!...\n");
    }

    if(fini_call)
        fini_names.push_back(fini_call.get());

    for(const auto& itr : end_expr)
    {
        if(itr.second)
            fini_names.push_back(itr.second.get());
    }

    if(exit_func && exit_call)
    {
        verbprintf(0, "Inserting exit instrumentation...\n");
        insert_instr(addr_space, exit_func, exit_call, BPatch_entry, nullptr, nullptr);
    }

    //----------------------------------------------------------------------------------//
    //
    //  Lambda for instrumenting procedures. The first pass (usage_pass = true) will
    //  generate the hash_ids for each string so that these can be inserted in bulk
    //  with one operation and do not have to be calculated during runtime.
    //
    //----------------------------------------------------------------------------------//
    std::vector<std::function<void()>> instr_procedure_functions;
    auto instr_procedures = [&](const procedure_vec_t& procedures) {
        verbprintf(2, "Instrumenting %lu procedures...\n",
                   (unsigned long) procedures.size());
        for(auto itr : procedures)
        {
            if(itr == main_func)
                continue;

            char modname[MUTNAMELEN];
            char fname[FUNCNAMELEN];

            module_t* mod = itr->getModule();
            if(mod)
                mod->getName(modname, MUTNAMELEN);
            else
                itr->getModuleName(modname, MUTNAMELEN);

            if(strstr(modname, "libdyninst") != nullptr)
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

            auto name = get_func_file_line_info(mod, itr);

            if(name.get().empty())
            {
                verbprintf(1, "Skipping function [empty name]: %s\n", fname);
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
                verbprintf(1, "Skipping function [excluded]: %s / %s\n",
                           name.m_name.c_str(), name.get().c_str());
                continue;
            }

            if(is_static_exe && has_debug_info && strcmp(fname, "_fini") != 0 &&
               strcmp(modname, "DEFAULT_MODULE") == 0)
            {
                verbprintf(1, "Skipping function [DEFAULT_MODULE]: %s\n", fname);
                continue;
            }

            hash_ids.push_back({ std::hash<string_t>()(name.get()), name.get() });
            available_module_functions.insert(module_function(mod, itr));
            instrumented_module_functions.insert(module_function(mod, itr));

            auto _f = [=]() {
                verbprintf(1, "Instrumenting |> [ %s ] -> [ %s ]\n", modname,
                           name.m_name.c_str());
                auto _name       = name.get();
                auto _hash       = std::hash<string_t>()(_name);
                auto _trace_entr = (entr_hash) ? timemory_call_expr(_hash)
                                               : timemory_call_expr(_name.c_str());
                auto _trace_exit = (exit_hash) ? timemory_call_expr(_hash)
                                               : timemory_call_expr(_name.c_str());
                auto _entr = _trace_entr.get((entr_hash) ? entr_hash : entr_trace);
                auto _exit = _trace_exit.get((exit_hash) ? exit_hash : exit_trace);

                insert_instr(addr_space, itr, _entr, BPatch_entry, nullptr, nullptr);
                insert_instr(addr_space, itr, _exit, BPatch_exit, nullptr, nullptr);
            };

            instr_procedure_functions.push_back(_f);

            if(loop_level_instr)
            {
                verbprintf(1, "Instrumenting at the loop level: %s\n",
                           name.m_name.c_str());

                flow_graph_t*    flow = itr->getCFG();
                basic_loop_vec_t basic_loop;
                if(flow)
                    flow->getOuterLoops(basic_loop);
                for(auto litr : basic_loop)
                {
                    auto lname  = get_loop_file_line_info(mod, itr, flow, litr);
                    auto _lname = lname.get();
                    auto _lhash = std::hash<string_t>()(_lname);
                    hash_ids.push_back({ _lhash, _lname });
                    auto _lf = [=]() {
                        auto _ltrace_entr = (entr_hash)
                                                ? timemory_call_expr(_lhash)
                                                : timemory_call_expr(_lname.c_str());
                        auto _ltrace_exit = (exit_hash)
                                                ? timemory_call_expr(_lhash)
                                                : timemory_call_expr(_lname.c_str());
                        auto _lentr =
                            _ltrace_entr.get((entr_hash) ? entr_hash : entr_trace);
                        auto _lexit =
                            _ltrace_exit.get((exit_hash) ? exit_hash : exit_trace);

                        insert_instr(addr_space, itr, _lentr, BPatch_entry, flow, litr);
                        insert_instr(addr_space, itr, _lexit, BPatch_exit, flow, litr);
                    };
                    instr_procedure_functions.push_back(_lf);
                }
            }
        }
    };

    //----------------------------------------------------------------------------------//
    //
    //  Load the dependent libraries (currently unused)
    //
    //----------------------------------------------------------------------------------//

    if(is_static_exe && false)
    {
        char* bindings   = new char[MUTNAMELEN];
        bool  loadResult = load_dependent_libraries(addr_space, bindings);
        delete[] bindings;
        if(!loadResult)
        {
            fprintf(stderr, "Failed to load dependent libraries\n");
            throw std::runtime_error("Failed to load dependent libraries");
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Do a first pass through all procedures to generate the hash ids
    //
    //----------------------------------------------------------------------------------//

    verbprintf(2, "Beginning loop over modules [hash id generation pass]\n");
    for(auto& m : modules)
    {
        char modname[1024];
        m->getName(modname, 1024);
        if(strstr(modname, "libdyninst") != nullptr)
            continue;

        if(!m->getProcedures())
        {
            verbprintf(1, "Skipping module w/ no procedures: '%s'\n", modname);
            continue;
        }

        verbprintf(1, "Parsing module: %s\n", modname);
        bpvector_t<procedure_t*>* p = m->getProcedures();
        if(!p)
            continue;

        instr_procedures(*p);
    }

    //----------------------------------------------------------------------------------//
    //
    //  Add the snippet that assign the hash ids
    //
    //----------------------------------------------------------------------------------//

    timemory_snippet_vec hash_snippet_vec;
    // generate a call expression for each hash + key
    for(auto& itr : hash_ids)
        hash_snippet_vec.generate(hash_func, itr.first, itr.second.c_str());
    // append all the call expressions to init names
    hash_snippet_vec.append(init_names);

    //----------------------------------------------------------------------------------//
    //
    //  Insert the initialization and finalization routines into the main entry and
    //  exit points
    //
    //----------------------------------------------------------------------------------//

    if(binary_rewrite)
    {
        verbprintf(1, "Adding main entry and exit snippets\n");
        if(main_entr_points)
            addr_space->insertSnippet(BPatch_sequence(init_names), *main_entr_points,
                                      BPatch_callBefore, BPatch_firstSnippet);
        if(main_exit_points)
            addr_space->insertSnippet(BPatch_sequence(fini_names), *main_exit_points,
                                      BPatch_callAfter, BPatch_firstSnippet);
    }
    else if(app_thread)
    {
        verbprintf(1, "Executing init_names...\n");
        for(auto itr : init_names)
        {
            app_thread->oneTimeCode(*itr);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Actually insert the instrumentation into the procedures
    //
    //----------------------------------------------------------------------------------//
    if(app_thread)
    {
        verbprintf(1, "Beginning insertion set...\n");
        addr_space->beginInsertionSet();
    }

    verbprintf(2, "Beginning loop over modules [instrumentation pass]\n");
    for(auto& instr_procedure : instr_procedure_functions)
        instr_procedure();

    if(app_thread)
    {
        verbprintf(1, "Finalizing insertion set...\n");
        bool modified = true;
        bool success  = addr_space->finalizeInsertionSet(true, &modified);
        if(!success)
        {
            verbprintf(
                1,
                "Using insertion set failed. Restarting with individual insertion...\n");
            auto _execute_batch = [&instr_procedure_functions, &addr_space](size_t _beg,
                                                                            size_t _end) {
                verbprintf(1, "Instrumenting batch of functions [%lu, %lu)\n",
                           (unsigned long) _beg, (unsigned long) _end);
                addr_space->beginInsertionSet();
                for(size_t i = _beg; i < _end; ++i)
                {
                    if(i < instr_procedure_functions.size())
                        instr_procedure_functions.at(i)();
                }
                bool _modified = true;
                bool _success  = addr_space->finalizeInsertionSet(true, &_modified);
                return _success;
            };

            auto execute_batch = [&_execute_batch,
                                  &instr_procedure_functions](size_t _beg) {
                if(!_execute_batch(_beg, _beg + batch_size))
                {
                    verbprintf(1,
                               "Batch instrumentation of functions [%lu, %lu) failed. "
                               "Beginning non-batched instrumentation for this set\n",
                               (unsigned long) _beg, (unsigned long) _beg + batch_size);
                    for(size_t i = _beg; i < _beg + batch_size; ++i)
                    {
                        if(i < instr_procedure_functions.size())
                            instr_procedure_functions.at(i)();
                    }
                }
                return _beg + batch_size;
            };

            size_t nidx = 0;
            while(nidx < instr_procedure_functions.size())
            {
                nidx = execute_batch(nidx);
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Dump the available instrumented modules/functions (re-dump available)
    //
    //----------------------------------------------------------------------------------//

    dump_info("available_module_functions.txt", available_module_functions, 0);
    dump_info("instrumented_module_functions.txt", instrumented_module_functions, 0);

    //----------------------------------------------------------------------------------//
    //
    //  Either write the instrumented binary or execute the application
    //
    //----------------------------------------------------------------------------------//
    if(binary_rewrite)
        addr_space->finalizeInsertionSet(false, nullptr);

    int code = -1;
    if(binary_rewrite)
    {
        char cwd[FUNCNAMELEN];
        auto ret = getcwd(cwd, FUNCNAMELEN);
        consume_parameters(ret);

        auto outf = outfile;
        if(outf.find('/') != string_t::npos)
        {
            auto outdir = outf.substr(0, outf.find_last_of('/') - 1);
            tim::makedir(outdir);
        }

        bool success = app_binary->writeFile(outfile.c_str());
        code         = (success) ? EXIT_SUCCESS : EXIT_FAILURE;
        if(success)
            printf("\nThe instrumented executable image is stored in '%s/%s'\n", cwd,
                   outfile.c_str());

        if(main_func)
        {
            printf("[timemory-run]> Getting linked libraries for %s...\n", cmdv0.c_str());
            printf(
                "[timemory-run]> Consider instrumenting the relevant libraries...\n\n");

            using TIMEMORY_PIPE = tim::popen::TIMEMORY_PIPE;

            tim::set_env("LD_TRACE_LOADED_OBJECTS", "1", 1);
            TIMEMORY_PIPE* ldd = tim::popen::popen(cmdv0.c_str());
            tim::set_env("LD_TRACE_LOADED_OBJECTS", "0", 1);

            strvec_t linked_libraries = tim::popen::read_fork(ldd);

            auto perr = tim::popen::pclose(ldd);
            if(perr != 0)
                perror("Error in timemory_fork");

            for(auto itr : linked_libraries)
                printf("\t%s\n", itr.c_str());

            printf("\n");
        }
    }
    else
    {
        printf("Executing...\n");

        // bpatch->setDebugParsing(false);
        // bpatch->setTypeChecking(false);
        // bpatch->setDelayedParsing(true);
        // bpatch->setInstrStackFrames(true);
        // bpatch->setLivenessAnalysis(false);
        // addr_space->beginInsertionSet();

        verbprintf(4, "Registering fork callbacks...\n");
        auto _prefork  = bpatch->registerPreForkCallback(&timemory_fork_callback);
        auto _postfork = bpatch->registerPostForkCallback(&timemory_fork_callback);

        auto _wait_exec = [&]() {
            while(!app_thread->isTerminated())
            {
                verbprintf(3, "Continuing execution...\n");
                app_thread->continueExecution();
                verbprintf(4, "Process is not terminated...\n");
                // std::this_thread::sleep_for(std::chrono::milliseconds(100));
                bpatch->waitForStatusChange();
                verbprintf(4, "Process status change...\n");
                if(app_thread->isStopped())
                {
                    verbprintf(4, "Process is stopped, continuing execution...\n");
                    if(!app_thread->continueExecution())
                    {
                        fprintf(stderr, "continueExecution failed\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        };

        verbprintf(4, "Entering wait for status change mode...\n");
        _wait_exec();

        if(app_thread->terminationStatus() == ExitedNormally)
        {
            if(app_thread->isTerminated())
                printf("\nEnd of timemory-run\n");
            else
                _wait_exec();
        }
        else if(app_thread->terminationStatus() == ExitedViaSignal)
        {
            auto sign = app_thread->getExitSignal();
            fprintf(stderr, "\nApplication exited with signal: %i\n", int(sign));
        }

        // addr_space->finalizeInsertionSet(false, nullptr);

        code = app_thread->getExitCode();
        consume_parameters(_prefork, _postfork);
    }

    // cleanup
    for(int i = 0; i < argc; ++i)
        delete[] _argv[i];
    delete[] _argv;
    for(int i = 0; i < _cmdc; ++i)
        delete[] _cmdv[i];
    delete[] _cmdv;
    return code;
}
//
//======================================================================================//
//
void
read_collection(const string_t& fname, strset_t& collection_set)
{
    string_t searched_paths;
    strset_t prefixes = { ".", get_absolute_path(argv0.c_str()) };
    for(const auto& pitr : prefixes)
    {
        for(auto itr : collection_paths)
        {
            itr = TIMEMORY_JOIN('/', pitr, itr);
            searched_paths += itr;
            searched_paths += ", ";
            auto fpath = TIMEMORY_JOIN('/', itr, fname);

            verbprintf(0, "trying to read collection file @ %s...", fpath.c_str());
            std::ifstream ifs(fpath.c_str());

            if(!ifs)
            {
                verbprintf(0, "trying to read collection file @ %s...", fpath.c_str());
                fpath = TIMEMORY_JOIN('/', itr, to_lower(fname));
                ifs.open(fpath.c_str());
            }

            if(ifs)
            {
                verbprintf(0, ">    reading collection file @ %s...", fpath.c_str());
                string_t tmp;
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
    stringstream_t ss;
    ss << "Unable to find \"" << fname << "\". Searched paths: " << searched_paths;
    if(werror)
        throw std::runtime_error(ss.str());
    else
        std::cerr << ss.str() << "\n";
}

//======================================================================================//

bool
process_file_for_instrumentation(const string_t& file_name)
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

    string_t          ext_str = "\\.S$";
    static std::regex ext_regex(ext_str, regex_opts);
    static std::regex sys_regex("^(s|k|e|w)_[A-Za-z_0-9\\-]+\\.(c|C)$", regex_opts);
    static std::regex userlib_regex(
        "^lib(timemory|caliper|gotcha|papi|cupti|TAU|likwid|"
        "profiler|tcmalloc|dyninst|pfm|nvtx|upcxx|pthread|nvperf)",
        regex_opts);
    static std::regex corelib_regex("^lib(rt-|dl-|util-|python)", regex_opts);
    // these are all due to TAU
    static std::regex prefix_regex(
        "^(RT|Tau|Profiler|Rts|Papi|Py|Comp_xl\\.cpp|Comp_gnu\\.cpp|"
        "UserEvent\\.cpp|FunctionInfo\\.cpp|PthreadLayer\\.cpp|"
        "Comp_intel[0-9]\\.cpp|Tracer\\.cpp|cxx11|locale)",
        regex_opts);

    if(timemory_source_file_constraint(file_name))
    {
        verbprintf(3, "Excluding instrumentation [timemory source] : '%s'...\n",
                   file_name.c_str());
        return false;
    }

    if(!cstd_func_instr && c_stdlib_module_constraint(file_name))
    {
        verbprintf(3, "Excluding instrumentation [c std library] : '%s'...\n",
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
        static strset_t already_reported;
        if(already_reported.count(file_name) == 0)
        {
            verbprintf(2, "%s |> [ %s ]\n", __FUNCTION__, file_name.c_str());
            already_reported.insert(file_name);
        }
    }
    return use;
}

//======================================================================================//

bool
instrument_entity(const string_t& function_name)
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

    static std::regex exclude(
        "(timemory|tim::|cereal|N3tim|MPI_Init|MPI_Finalize|::__[A-Za-z]|"
        "dyninst|tm_clones|malloc$|calloc$|free$|realloc$|std::addressof)",
        regex_opts);
    static std::regex exclude_cxx("(std::_Sp_counted_base|std::use_facet)", regex_opts);
    static std::regex leading(
        "^(_|frame_dummy|\\(|targ|new|delete|operator new|operator delete|std::allocat|"
        "nvtx|gcov|main\\.cold\\.|TAU|tau|Tau|dyn|RT|dl|sys|pthread|posix|clone|thunk)",
        regex_opts);
    static std::regex stlfunc("^std::", regex_opts);
    strset_t          whole = { "init", "fini", "_init", "_fini" };

    if(!stl_func_instr && std::regex_search(function_name, stlfunc))
    {
        verbprintf(3, "Excluding function [stl] : '%s'...\n", function_name.c_str());
        return false;
    }

    if(!cstd_func_instr && c_stdlib_function_constraint(function_name))
    {
        verbprintf(3, "Excluding function [libc] : '%s'...\n", function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude))
    {
        verbprintf(3, "Excluding function [critical, any match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude_cxx))
    {
        verbprintf(3, "Excluding function [critical_cxx, any match] : '%s'...\n",
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
template <typename Tp>
void
insert_instr(address_space_t* mutatee, procedure_t* funcToInstr, Tp traceFunc,
             procedure_loc_t traceLoc, flow_graph_t* cfGraph,
             basic_loop_t* loopToInstrument)
{
    module_t* module = funcToInstr->getModule();
    if(!module || !traceFunc)
        return;

    bpvector_t<point_t*>* _points = nullptr;
    auto                  _trace  = traceFunc.get();

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

    /*if(loop_level_instr)
    {
        flow_graph_t*                     flow = funcToInstr->getCFG();
        bpvector_t<basic_loop_t*> basicLoop;
        flow->getOuterLoops(basicLoop);
        for(auto litr = basicLoop.begin(); litr != basicLoop.end(); ++litr)
        {
            bpvector_t<point_t*>* _tmp;
            if(traceLoc == BPatch_entry)
                _tmp = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, *litr);
            else if(traceLoc == BPatch_exit)
                _tmp = cfGraph->findLoopInstPoints(BPatch_locLoopExit, *litr);
            if(!_tmp)
                continue;
            for(auto& itr : *_tmp)
                _points->push_back(itr);
        }
    }*/

    // verbprintf(0, "Instrumenting |> [ %s ]\n", name.m_name.c_str());

    for(auto& itr : *_points)
    {
        if(!itr)
            continue;
        else if(traceLoc == BPatch_entry)
            mutatee->insertSnippet(*_trace, *itr, BPatch_callBefore, BPatch_firstSnippet);
        // else if(traceLoc == BPatch_exit)
        //    mutatee->insertSnippet(*_trace, *itr, BPatch_callAfter,
        //    BPatch_firstSnippet);
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

    string_t _fname = fname;
    if(_fname.find("timemory") != string_t::npos ||
       _fname.find("tim::") != string_t::npos)
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

    if(process_file_for_instrumentation(string_t(fname)))
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
    string_t _fname = fname;
    if(_fname.find("timemory") != string_t::npos ||
       _fname.find("tim::") != string_t::npos)
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
        if(instrument_entity(string_t(fname)))
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
std::string
get_absolute_exe_filepath(std::string exe_name)
{
    auto file_exists = [](const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    };

    if(!exe_name.empty() && !file_exists(exe_name))
    {
        auto _exe_orig = exe_name;
        auto _paths    = tim::delimit(tim::get_env<std::string>("PATH", ""), ":");
        for(auto& pitr : _paths)
        {
            if(file_exists(TIMEMORY_JOIN('/', pitr, exe_name)))
            {
                exe_name = TIMEMORY_JOIN('/', pitr, exe_name);
                verbprintf(0, "Resolved '%s' to '%s'...\n", _exe_orig.c_str(),
                           exe_name.c_str());
                break;
            }
        }

        if(!file_exists(exe_name))
        {
            verbprintf(0, "Warning! File path to '%s' could not be determined...\n",
                       exe_name.c_str());
        }
    }
    return exe_name;
}

//======================================================================================//
//
inline void
consume()
{
    consume_parameters(initialize_expr, expect_error, error_print);
}
//
namespace
{
static auto _consumed = (consume(), true);
}
