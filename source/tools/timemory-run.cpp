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

int         expectError      = NO_ERROR;
int         debugPrint       = 0;
int         binaryRewrite    = 0;  /* by default, it is turned off */
int         errorPrint       = 0;  // external "dyninst" tracing
bool        loop_level_instr = false;
bool        werror           = false;
bool        stl_func_instr   = false;
std::string main_fname       = "main";
std::string argv0            = "";

template class BPatch_Vector<BPatch_variableExpr*>;

BPatch_function*               name_reg;
BPatch_Vector<BPatch_snippet*> init_names;
BPatch_Vector<BPatch_snippet*> fini_names;
BPatch*                        bpatch;
std::vector<std::regex>        regex_include;
std::vector<std::regex>        regex_exclude;
std::set<std::string>          collection_includes;
std::set<std::string>          collection_excludes;
std::vector<std::string>       collection_paths = { "collections", "tools/collections",
                                              "../share/tools/collections" };

//======================================================================================//
// For selective instrumentation
//
bool
are_file_include_exclude_lists_empty(void)
{
    return false;
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
    std::regex corelib_regex("^lib(rt-|m-|dl-|m-|gcc|c-|stdc++|c++)");
    if(check_project_source_file_for_instrumentation(file_name) ||
       std::regex_search(file_name, ext_regex) ||
       std::regex_search(file_name, corelib_regex) ||
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
    std::regex leading(
        "^(_init|_fini|__|_dl_|_start|_exit|frame_dummy|\\(\\(|\\(__|_"
        "GLOBAL|targ|PMPI_|new|delete|std::allocator|std::_|std::move|nvtx|gcov)");
    std::regex stlfunc("^std::");

    if(!stl_func_instr && std::regex_search(function_name, stlfunc))
        return false;

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
// gets information (line number, filename, and column number) about
// the instrumented loop and formats it properly.
//
function_signature
get_loop_file_line_info(BPatch_image* mutateeImage, BPatch_flowGraph* cfGraph,
                        BPatch_basicBlockLoop* loopToInstrument, BPatch_function* f)
{
    if(!cfGraph || !loopToInstrument || !f)
        return function_signature("", "", "");

    char         fname[1024];
    const char*  typeName = nullptr;
    BPatch_type* returnType;

    BPatch_Vector<BPatch_point*>* loopStartInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopStartIter, loopToInstrument);
    BPatch_Vector<BPatch_point*>* loopExitInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopEndIter, loopToInstrument);

    if(!loopStartInst || !loopExitInst)
        return function_signature("", "", "");

    unsigned long baseAddr = (unsigned long) (*loopStartInst)[0]->getAddress();
    unsigned long lastAddr =
        (unsigned long) (*loopExitInst)[loopExitInst->size() - 1]->getAddress();
    dprintf("Loop: size of lastAddr = %lu: baseAddr = %lu, lastAddr = %lu\n",
            (unsigned long) loopExitInst->size(), (unsigned long) baseAddr,
            (unsigned long) lastAddr);

    f->getName(fname, 1024);

    returnType = f->getReturnType();

    if(returnType)
    {
        typeName = returnType->getName();
    }
    else
        typeName = "void";

    BPatch_Vector<BPatch_statement> lines;
    BPatch_Vector<BPatch_statement> linesEnd;

    bool info1 = mutateeImage->getSourceLines(baseAddr, lines);

    if(info1)
    {
        auto filename = lines[0].fileName();
        auto row1     = lines[0].lineNumber();
        auto col1     = lines[0].lineOffset();
        if(col1 < 0)
            col1 = 0;

        // This following section is attempting to remedy the limitations of
        // getSourceLines for loops. As the program goes through the loop, the resulting
        // lines go from the loop head, through the instructions present in the loop, to
        // the last instruction in the loop, back to the loop head, then to the next
        // instruction outside of the loop. What this section does is starts at the last
        // instruction in the loop, then goes through the addresses until it reaches the
        // next instruction outside of the loop. We then bump back a line. This is not a
        // perfect solution, but we will work with the Dyninst team to find something
        // better.
        bool info2 = mutateeImage->getSourceLines((unsigned long) lastAddr, linesEnd);
        dprintf("size of linesEnd = %lu\n", (unsigned long) linesEnd.size());

        if(info2)
        {
            auto row2 = linesEnd[0].lineNumber();
            auto col2 = linesEnd[0].lineOffset();
            if(col2 < 0)
                col2 = 0;
            if(row2 < row1)
                row1 = row2; /* Fix for wrong line numbers*/

            return function_signature(typeName, fname, filename, { row1, row2 },
                                      { col1, col2 }, true, info1, info2);
        }
        else
        {
            return function_signature(typeName, fname, filename, { row1, 0 }, { col1, 0 },
                                      true, info1, info2);
        }
    }
    else
    {
        return function_signature(typeName, fname, "");
    }
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

    auto ret = new BPatch_retExpr();
    entryTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));
    entryTraceArgs.push_back(ret);
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

    BPatch_Vector<BPatch_point*>* funcEntry = funcToInstr->findPoint(BPatch_entry);
    BPatch_Vector<BPatch_point*>* funcExit  = funcToInstr->findPoint(BPatch_exit);

    BPatch_Vector<BPatch_snippet*> entryTraceArgs;
    BPatch_Vector<BPatch_snippet*> exitTraceArgs;

    auto ret = new BPatch_retExpr();
    entryTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));
    entryTraceArgs.push_back(ret);
    exitTraceArgs.push_back(new BPatch_constExpr(name.get().c_str()));

    BPatch_funcCallExpr entryTrace(*traceEntryFunc, entryTraceArgs);
    BPatch_funcCallExpr exitTrace(*traceExitFunc, exitTraceArgs);

    mutatee->insertSnippet(entryTrace, *funcEntry, BPatch_callBefore,
                           BPatch_firstSnippet);
    mutatee->insertSnippet(exitTrace, *funcExit, BPatch_callAfter, BPatch_lastSnippet);
}

//======================================================================================//
//
// Error callback routine.
//
void
errorFunc(BPatchErrorLevel level, int num, const char** params)
{
    char line[256];

    const char* msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    if(num != expectError)
    {
        printf("Error #%d (level %d): %s\n", num, level, line);
        // We consider some errors fatal.
        if(num == 101)
            exit(-1);
    }
}

//======================================================================================//
//
// For compatibility purposes
//
BPatch_function*
find_function(BPatch_image* appImage, const char* functionName)
{
    // Extract the vector of functions
    BPatch_Vector<BPatch_function*> found_funcs;
    if((nullptr ==
        appImage->findFunction(functionName, found_funcs, false, true, true)) ||
       !found_funcs.size())
    {
        dprintf("timemory-run: Unable to find function %s\n", functionName);
        return nullptr;
    }
    return found_funcs[0];
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
        exit(1);
    }

    // Then find the points using loc (entry/exit) for the given function
    const BPatch_Vector<BPatch_point*>* points = function->findPoint(loc);

    if(points != nullptr)
    {
        // Insert the given snippet at the given point
        if(loc == BPatch_entry)
        {
            appThread->insertSnippet(*snippet, *points, BPatch_callBefore,
                                     BPatch_lastSnippet);
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
    BPatch_function* init_func = find_function(appImage, "timemory_init_trace");
    if(init_func == nullptr)
    {
        fprintf(stderr, "Unable to find function timemory_init_trace\n");
        exit(1);
    }

    // Find the finalization function and call it
    BPatch_function* fini_func = find_function(appImage, "timemory_fini_trace");
    if(fini_func == nullptr)
    {
        fprintf(stderr, "Unable to find function timemory_fini_trace\n");
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
//   check that the cost of a snippet is sane.  Due to differences between
//   platforms, it is impossible to check this exactly in a machine independent
//   manner.
void
check_cost(BPatch_snippet snippet)
{
    float          cost;
    BPatch_snippet copy;

    // test copy constructor too.
    copy = snippet;
    cost = snippet.getCost();
    if(cost < 0.0)
        printf("*Error*: negative snippet cost\n");
    else if(cost == 0.0)
        printf("*Warning*: zero snippet cost\n");
    else if(cost > 0.01)
        printf("*Error*: snippet cost of %f, exceeds max expected of 0.1", cost);
}

//======================================================================================//
void
error_func_real(BPatchErrorLevel level, int num, const char* const* params)
{
    if(num == 0)
    {
        // conditional reporting of warnings and informational messages
        if(errorPrint)
        {
            if(level == BPatchInfo)
            {
                if(errorPrint > 1)
                    printf("%s\n", params[0]);
            }
            else
                printf("%s", params[0]);
        }
    }
    else
    {
        // reporting of actual errors
        char        line[256];
        const char* msg = bpatch->getEnglishErrorString(num);
        bpatch->formatErrorString(line, sizeof(line), msg, params);
        if(num != expectError)
        {
            printf("Error #%d (level %d): %s\n", num, level, line);
            // We consider some errors fatal.
            if(num == 101)
                exit(-1);
        }
    }
}

//======================================================================================//
// We've a null error function when we don't want to display an error
void
error_func_fake(BPatchErrorLevel level, int num, const char* const* params)
{
    consume_parameters(level, num, params);
    // It does nothing.
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

    if(are_file_include_exclude_lists_empty())
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
            /* It is ok to instrument this module. Constraint doesn't exist. */
            return false;
        }
        else
            return true;
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
find_func_or_calls(std::vector<const char*> names, BPatch_Vector<BPatch_point*>& points,
                   BPatch_image* appImage, BPatch_procedureLocation loc = BPatch_locEntry)
{
    BPatch_function* func = nullptr;
    for(std::vector<const char*>::iterator i = names.begin(); i != names.end(); ++i)
    {
        BPatch_function* f = find_function(appImage, *i);
        if(f && f->getModule()->isSharedLib())
        {
            func = f;
            break;
        }
    }
    if(func)
    {
        BPatch_Vector<BPatch_point*>*          fpoints = func->findPoint(loc);
        BPatch_Vector<BPatch_point*>::iterator k;
        if(fpoints && fpoints->size())
        {
            for(k = fpoints->begin(); k != fpoints->end(); k++)
            {
                points.push_back(*k);
            }
            return true;
        }
    }

    // Moderately expensive loop here.  Perhaps we should make a name->point map first
    // and just do lookups through that.
    BPatch_Vector<BPatch_function*>* all_funcs           = appImage->getProcedures();
    auto                             initial_points_size = points.size();
    for(std::vector<const char*>::iterator i = names.begin(); i != names.end(); ++i)
    {
        BPatch_Vector<BPatch_function*>::iterator j;
        for(j = all_funcs->begin(); j != all_funcs->end(); j++)
        {
            BPatch_function* f = *j;
            if(f->getModule()->isSharedLib())
                continue;
            BPatch_Vector<BPatch_point*>* fpoints = f->findPoint(BPatch_locSubroutine);
            if(!fpoints || !fpoints->size())
                continue;
            BPatch_Vector<BPatch_point*>::iterator j;
            for(j = fpoints->begin(); j != fpoints->end(); j++)
            {
                std::string callee = (*j)->getCalledFunctionName();
                if(callee == std::string(*i))
                {
                    points.push_back(*j);
                }
            }
        }
        if(points.size() != initial_points_size)
            return true;
    }

    return false;
}

//======================================================================================//
//
bool
find_func_or_calls(const char* name, BPatch_Vector<BPatch_point*>& points,
                   BPatch_image* image, BPatch_procedureLocation loc = BPatch_locEntry)
{
    std::vector<const char*> v;
    v.push_back(name);
    return find_func_or_calls(v, points, image, loc);
}

//======================================================================================//
//
// check if the application has an MPI library routine MPI_Comm_rank
//
int
check_if_mpi(BPatch_image* appImage, BPatch_Vector<BPatch_point*>& mpiinit,
             BPatch_function*& mpiinitstub, bool binaryRewrite)
{
    consume_parameters(binaryRewrite);

    std::vector<const char*> _init_names;
    _init_names.push_back("MPI_Init");
    _init_names.push_back("PMPI_Init");
    bool ismpi = find_func_or_calls(_init_names, mpiinit, appImage, BPatch_locExit);

    mpiinitstub = find_function(appImage, "timemory_mpi_init_stub");
    if(mpiinitstub == (BPatch_function*) nullptr)
        printf("*** MPIInitStubInt not found! \n");

    if(!ismpi)
    {
        dprintf("*** This is not an MPI Application! \n");
        return 0;
    }
    else
        return 1;
}

//======================================================================================//
// We create a new name that embeds the file and line information in the name
//
function_signature
get_func_file_line_info(BPatch_image* mutatee_addr_space, BPatch_function* f)
{
    bool          info1, info2;
    unsigned long baseAddr, lastAddr;
    char          fname[1024];
    const char*   filename;
    int           row1, col1, row2, col2;
    BPatch_type*  returnType;
    const char*   typeName;

    baseAddr = (unsigned long) (f->getBaseAddr());
    f->getAddressRange(baseAddr, lastAddr);
    BPatch_Vector<BPatch_statement> lines;
    f->getName(fname, 1024);

    returnType = f->getReturnType();

    if(returnType)
    {
        typeName = returnType->getName();
    }
    else
        typeName = "void";

    info1 = mutatee_addr_space->getSourceLines((unsigned long) baseAddr, lines);

    if(info1)
    {
        filename = lines[0].fileName();
        row1     = lines[0].lineNumber();
        col1     = lines[0].lineOffset();
        std::string file(filename);
        if(file.find('/') != std::string::npos)
            file = file.substr(file.find_last_of('/') + 1);

        if(col1 < 0)
            col1 = 0;
        info2 = mutatee_addr_space->getSourceLines((unsigned long) (lastAddr - 1), lines);
        if(info2)
        {
            row2 = lines[1].lineNumber();
            col2 = lines[1].lineOffset();
            if(col2 < 0)
                col2 = 0;
            if(row2 < row1)
                row1 = row2;
            return function_signature(typeName, fname, filename, { row1, 0 }, { 0, 0 },
                                      false, info1, info2);
        }
        else
        {
            return function_signature(typeName, fname, filename, { row1, 0 }, { 0, 0 },
                                      false, info1, info2);
        }
    }
    else
    {
        return function_signature(typeName, fname, "", { 0, 0 }, { 0, 0 }, false, false,
                                  false);
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
    FILE* fp;
    fp = popen(cmd, "r");

    if(fp == nullptr)
    {
        perror("timemory-run: Error launching timemory_show_libs to get list of "
               "dependent static libraries for static binary");
        return 1;
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

//======================================================================================//
//
int
timemory_rewrite_binary(BPatch* bpatch, const char* mutateeName, char* outfile,
                        char* sharedlibname, char* staticlibname, char* bindings)
{
    using namespace std;
    BPatch_Vector<BPatch_point*> mpiinit;
    BPatch_function*             mpiinitstub = nullptr;

    dprintf("Inside timemory_rewrite_binary, name=%s, out=%s\n", mutateeName, outfile);
    BPatch_binaryEdit* mutatee_addr_space = bpatch->openBinary(mutateeName, false);

    if(mutatee_addr_space == nullptr)
    {
        fprintf(stderr, "Failed to open binary %s\n", mutateeName);
        return -1;
    }

    BPatch_image*                    mutateeImage = mutatee_addr_space->getImage();
    BPatch_Vector<BPatch_function*>& allFuncs     = *mutateeImage->getProcedures();
    bool                             isStaticExecutable;

    isStaticExecutable = mutatee_addr_space->isStaticExecutable();

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

    BPatch_function* entryTrace = find_function(mutateeImage, "timemory_register_trace");
    BPatch_function* exitTrace = find_function(mutateeImage, "timemory_deregister_trace");
    BPatch_function* setupFunc = find_function(mutateeImage, "timemory_dyninst_init");
    BPatch_function* cleanupFunc =
        find_function(mutateeImage, "timemory_dyninst_finalize");
    BPatch_function* mainFunc = find_function(mutateeImage, main_fname.c_str());
    name_reg                  = find_function(mutateeImage, "timemory_register_trace");

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

    if(!mainFunc)
    {
        fprintf(stderr, "Couldn't find %s(), aborting\n", main_fname.c_str());
        return -1;
    }

    if(!entryTrace || !exitTrace || !setupFunc || !cleanupFunc)
    {
        fprintf(stderr, "Couldn't find entry/exit/setup/cleanup functions, aborting\n");
        return -1;
    }

    BPatch_Vector<BPatch_point*>* main_init_entry = mainFunc->findPoint(BPatch_entry);
    BPatch_Vector<BPatch_point*>* main_fini_entry = mainFunc->findPoint(BPatch_exit);

    assert(main_init_entry);
    assert(main_init_entry->size());
    assert((*main_init_entry)[0]);

    assert(main_fini_entry);
    assert(main_fini_entry->size());
    assert((*main_fini_entry)[0]);

    mutatee_addr_space->beginInsertionSet();

    int              ismpi = check_if_mpi(mutateeImage, mpiinit, mpiinitstub, true);
    BPatch_constExpr isMPI(ismpi);

    BPatch_Vector<BPatch_snippet*> init_params;
    BPatch_Vector<BPatch_snippet*> fini_params;
    init_params.push_back(&isMPI);

    BPatch_funcCallExpr setup_call(*setupFunc, init_params);
    BPatch_funcCallExpr cleanup_call(*cleanupFunc, fini_params);

    init_names.push_back(&setup_call);
    fini_names.push_back(&cleanup_call);

    if(ismpi && mpiinitstub)
    {
        // Create a snippet that calls timemory_mpi_init_stub with the rank after MPI_Init
        BPatch_function* mpi_rank = find_function(mutateeImage, "timemory_get_rank");
        assert(mpi_rank);
        BPatch_Vector<BPatch_snippet*> rank_args;
        BPatch_funcCallExpr            getrank(*mpi_rank, rank_args);
        BPatch_Vector<BPatch_snippet*> mpiinitargs;
        mpiinitargs.push_back(&getrank);
        BPatch_funcCallExpr initmpi(*mpiinitstub, mpiinitargs);
        mutatee_addr_space->insertSnippet(initmpi, mpiinit, BPatch_callAfter,
                                          BPatch_firstSnippet);
    }

    for(auto it = allFuncs.begin(); it != allFuncs.end(); ++it)
    {
        char fname[FUNCNAMELEN];
        (*it)->getName(fname, FUNCNAMELEN);

        // STATIC EXECUTABLE FUNCTION EXCLUDE
        // Temporarily avoid some functions -- this isn't a solution
        // -- it appears that something like module_constraint would work
        // well here
        if(isStaticExecutable)
        {
            // Always instrument _fini to ensure instrumentation disabled correctly
            if(hasDebuggingInfo && strcmp(fname, "_fini") != 0)
            {
                BPatch_module* funcModule = (*it)->getModule();
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
        (*it)->getModuleName(modname, 1024);
        if(strstr(modname, "libdyninstAPI_RT"))
            continue;

        if(module_constraint(modname))
            continue;

        auto name = get_func_file_line_info(mutatee_addr_space->getImage(), *it);
        if(name.get().empty() || routine_constraint(fname) ||
           routine_constraint(name.m_name.c_str()) || !instrument_entity(name.m_name) ||
           !instrument_entity(name.get()))
            continue;

        insert_trace(*it, mutatee_addr_space, entryTrace, exitTrace, name);

        if(loop_level_instr)
        {
            dprintf("Generating CFG at loop level: %s\n", fname);
            BPatch_flowGraph*                     flow = (*it)->getCFG();
            BPatch_Vector<BPatch_basicBlockLoop*> basicLoop;
            dprintf("Generating outer loop info : %s\n", fname);
            flow->getOuterLoops(basicLoop);
            dprintf("Before instrumenting at loop level: %s\n", fname);

            for(BPatch_Vector<BPatch_basicBlockLoop*>::iterator loopIt =
                    basicLoop.begin();
                loopIt != basicLoop.end(); loopIt++)
            {
                dprintf("Instrumenting at the loop level: %s\n", fname);
                insert_trace(*it, mutatee_addr_space, entryTrace, exitTrace, flow,
                             *loopIt, name);
            }
        }
    }

    BPatch_sequence init_sequence(init_names);
    mutatee_addr_space->insertSnippet(init_sequence, *main_init_entry, BPatch_callBefore,
                                      BPatch_firstSnippet);
    BPatch_sequence fini_sequence(fini_names);
    mutatee_addr_space->insertSnippet(fini_sequence, *main_fini_entry, BPatch_callAfter,
                                      BPatch_firstSnippet);
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
//
// entry point
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
    std::string functions;  // string variable to hold function names
    std::string inputlib = "";

    // bpatch->setTrampRecursive(true); /* enable C++ support */
    // bpatch->setBaseTrampDeletion(true);
    // bpatch->setMergeTramp(false);
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
        std::cout << " [command]: " << cmd_string(_cmdc, _cmdv) << std::endl;

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
        .names({ "-p", "--collection-path" })
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
            "timemory uses the STL internally so this may cause deadlocks. Use the "
            "'-E/--regex-exclude' option to exclude any function causing deadlocks")
        .count(0);

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
    if(parser.exists("p"))
    {
        auto tmp = parser.get<std::vector<std::string>>("p");
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

    // Register a callback function that prints any error messages
    bpatch->registerErrorCallback(error_func_real);

    dprintf("Before createProcess\n");

    if(binaryRewrite)
    {
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

    if(_cmdc == 0)
        return EXIT_FAILURE;

    appThread = bpatch->processCreate(_cmdv[0], (const char**) _cmdv, nullptr);

    dprintf("After createProcess\n");

    if(!appThread)
    {
        fprintf(stderr, "[timemory-run]> createProcess failed\n");
        exit(1);
    }

    dprintf("Before getImage and getModules\n");
    // get image
    BPatch_image*                  appImage = appThread->getImage();
    BPatch_Vector<BPatch_module*>* m        = appImage->getModules();

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
            for(size_t i = 0; i < m->size(); ++i)
            {
                (*m)[i]->getName(name, sizeof(name));
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
                return EXIT_FAILURE;
            }
        }
        else
        {
            fprintf(stderr, "ERROR:%s not loaded properly. \n", sharedlibname);
            fprintf(stderr, "Please make sure that its path is in your LD_LIBRARY_PATH "
                            "environment variable.\n");
            return EXIT_FAILURE;
        }
        dprintf("Load library: %i\n", (int) ret);
    }

    dprintf("Before find_function\n");
    BPatch_function* inFunc;
    BPatch_function* enterstub = find_function(appImage, "timemory_register_trace");
    BPatch_function* exitstub  = find_function(appImage, "timemory_deregister_trace");
    BPatch_function* setupFunc = find_function(appImage, "timemory_dyninst_init");
    // BPatch_function* cleanupFunc = find_function(appImage,
    // "timemory_dyninst_finalize");
    BPatch_function* terminatestub = find_function(appImage, "timemory_dyninst_finalize");
    name_reg                       = find_function(appImage, "timemory_register_trace");

    char                           modulename[512];
    BPatch_Vector<BPatch_snippet*> initArgs;

    dprintf("Before modules loop\n");
    for(size_t j = 0; j < m->size(); j++)
    {
        if(!(*m)[j]->getProcedures())
            continue;
        sprintf(modulename, "Module %s\n", (*m)[j]->getName(fname, FUNCNAMELEN));
        BPatch_Vector<BPatch_function*> p = *((*m)[j]->getProcedures());
        dprintf("%s", modulename);

        if(!module_constraint(fname))
        {  // constraint
            for(size_t i = 0; i < p.size(); ++i)
            {
                // For all procedures within the module, iterate
                p[i]->getName(fname, FUNCNAMELEN);
                auto name = get_func_file_line_info(appImage, p[i]);
                if(!routine_constraint(fname) && !name.get().empty() &&
                   !routine_constraint(name.m_name.c_str()))
                {
                    // routines that are ok to instrument
                    // get full source information
                    functions.append("|");
                    functions.append(name.get().c_str());
                }
            }
        }
    }

    std::cout << "functions: " << functions << std::endl;

    // When we look for MPI calls, we shouldn't display an error message for
    // not find MPI_Comm_rank in the case of a sequential app. So, we turn the
    // Error callback to be Null and turn back the error settings later. This
    // way, it works for both MPI and sequential tasks.
    bpatch->registerErrorCallback(error_func_fake);  // turn off error reporting
    BPatch_constExpr isMPI(check_if_mpi(appImage, mpiinit, mpiinitstub, binaryRewrite));
    bpatch->registerErrorCallback(error_func_real);  // turn it back on

    BPatch_Vector<BPatch_snippet*> setupArgs;
    BPatch_funcCallExpr            setupExpr(*setupFunc, setupArgs);
    appThread->oneTimeCode(setupExpr);

    // initArgs.push_back(&funcName);
    initArgs.push_back(&isMPI);

    initialize(appThread, appImage, initArgs);

    dprintf("Did initialize\n");

    // In our tests, the routines started execution concurrently with the
    // one time code. To avoid this, we first start the one time code and then
    // iterate through the list of routines to select for instrumentation and
    // instrument these. So, we need to iterate twice.

    for(size_t j = 0; j < m->size(); j++)
    {
        if(!(*m)[j]->getProcedures())
            continue;
        sprintf(modulename, "Module %s\n", (*m)[j]->getName(fname, FUNCNAMELEN));
        BPatch_Vector<BPatch_function*> p = *((*m)[j]->getProcedures());
        dprintf("%s", modulename);

        if(!module_constraint(fname))
        {
            // constraint
            for(size_t i = 0; i < p.size(); ++i)
            {
                // For all procedures within the module, iterate
                p[i]->getName(fname, FUNCNAMELEN);
                inFunc    = p[i];
                auto name = get_func_file_line_info(appImage, inFunc);
                dprintf("Name %s\n", fname);
                if(!routine_constraint(fname) && !name.m_name.empty() &&
                   !routine_constraint(name.m_name.c_str()))
                {
                    // routines that are ok to instrument
                    dprintf("Instrumenting |> [ %s ]\n", name.m_name.c_str());

                    auto callee_entry_args = new BPatch_Vector<BPatch_snippet*>();
                    auto callee_exit_args  = new BPatch_Vector<BPatch_snippet*>();

                    auto ret = new BPatch_retExpr();
                    callee_entry_args->push_back(
                        new BPatch_constExpr(name.get().c_str()));
                    callee_entry_args->push_back(ret);
                    callee_exit_args->push_back(new BPatch_constExpr(name.get().c_str()));

                    // dprintf("Instrumenting-> %s Entry\n", fname);
                    invoke_routine_in_func(appThread, appImage, inFunc, BPatch_entry,
                                           enterstub, callee_entry_args);
                    // dprintf("Instrumenting-> %s Exit...", fname);
                    invoke_routine_in_func(appThread, appImage, inFunc, BPatch_exit,
                                           exitstub, callee_exit_args);
                    // dprintf("Done\n");
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
        // exit(1);
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

    printf("Executing...\n");

    auto _continue_exec = [&]() {
        if(!appThread->continueExecution())
            fprintf(stderr, "continueExecution failed\n");
    };

    auto _wait_exec = [&]() {
        uint64_t _ncount = 0;
        while(!appThread->isTerminated())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            bpatch->waitForStatusChange();
            if(appThread->isStopped())
                _continue_exec();
            if(_ncount++ > 50)
            {
                printf("Terminating application after 50 status changes...\n");
                appThread->terminateExecution();
                return;
            }
        }
        if(appThread->isTerminated())
            printf("End of application\n");
    };

    _continue_exec();
    _wait_exec();

    if(!appThread->isTerminated())
        appThread->terminateExecution();

    // cleanup
    for(int i = 0; i < argc; ++i)
        delete[] _argv[i];
    delete[] _argv;
    for(int i = 0; i < _cmdc; ++i)
        delete[] _cmdv[i];
    delete[] _cmdv;
    delete bpatch;
    return 0;
}
