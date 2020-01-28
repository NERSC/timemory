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

#include "BPatch.h"
#include "BPatch_Vector.h"
#include "BPatch_addressSpace.h"
#include "BPatch_basicBlockLoop.h"
#include "BPatch_function.h"
#include "BPatch_point.h"
#include "BPatch_process.h"
#include "BPatch_snippet.h"
#include "BPatch_statement.h"

#include <string.h>
#include <string>
#include <unistd.h>

#include "timemory/library.h"
#include "timemory/timemory.hpp"

#define MUTNAMELEN 64
#define FUNCNAMELEN 32 * 1024
#define NO_ERROR -1

#define TIMEMORY_BIN_DIR "bin"

using std::string;

int expectError   = NO_ERROR;
int debugPrint    = 0;
int binaryRewrite = 0; /* by default, it is turned off */

template class BPatch_Vector<BPatch_variableExpr*>;

BPatch_function*               name_reg;
BPatch_Vector<BPatch_snippet*> funcNames;
BPatch*                        bpatch;

void
check_cost(BPatch_snippet snippet);

//======================================================================================//

// control debug printf statements
#define dprintf                                                                          \
    if(debugPrint)                                                                       \
    printf

//======================================================================================//
// For selective instrumentation
//
bool
are_file_include_exclude_lists_empty(void)
{
    return false;
}

//======================================================================================//

bool
process_file_for_instrumentation(const string& file_name)
{
    if(debugPrint)
        PRINT_HERE("%s", file_name.c_str());
    return true;
}

//======================================================================================//

bool
instrument_entity(const string& function_name)
{
    std::set<std::string> exclude = {
        "tim::",          "timemory",        "cereal::",
        "rapidjson",      "label_array",     "display_unit_array",
        "frame_dummy",    "basic_string",    "~array",
        "~vector",        "~stack",          "~map",
        "~unordered_map", "~deque",          "~pair",
        "~tuple",         "~_Function_base", "_M_fill_insert",
        "_M_erase",       "_M_create_nodes", "_M_initialize",
        "_M_reallocate",  "_M_insert",       "_M_rehash",
        "_M_realloc",     "_M_manager",      "std::_",
        "atexit"
    };

    // don't instrument the functions that are explicitly from timemory
    for(const auto& itr : exclude)
    {
        if(function_name.find(itr) != std::string::npos)
            return false;
    }

    // don't instrument functions with leading underscore
    if(function_name.length() > 0 && function_name[0] == '_')
        return false;

    if(debugPrint)
        PRINT_HERE("%s", function_name.c_str());

    return true;
}

//======================================================================================//

extern bool
match_name(const string& str1, const string& str2);

//======================================================================================//
// prototypes for routines below
//
void
get_func_file_line_info(BPatch_image* mutateeAddressSpace, BPatch_function* f,
                        char* newname);

//======================================================================================//

int
add_name(const char* name)
{
    static int funcID = 0;
    PRINT_HERE("func: %s, id: %i\n", name, funcID);
    BPatch_constExpr*              name_param = new BPatch_constExpr(name);
    BPatch_Vector<BPatch_snippet*> params;
    params.push_back(name_param);
    BPatch_funcCallExpr* call = new BPatch_funcCallExpr(*name_reg, params);
    funcNames.push_back(call);
    return funcID++;
}

//======================================================================================//
// gets information (line number, filename, and column number) about
// the instrumented loop and formats it properly.
//
void
get_loop_file_line_info(BPatch_image* mutateeImage, BPatch_flowGraph* cfGraph,
                        BPatch_basicBlockLoop* loopToInstrument, BPatch_function* f,
                        char* newname)
{
    const char*  filename;
    char         fname[1024];
    const char*  typeName;
    bool         info1, info2;
    int          row1, col1, row2, col2;
    BPatch_type* returnType;

    BPatch_Vector<BPatch_point*>* loopStartInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopStartIter, loopToInstrument);
    BPatch_Vector<BPatch_point*>* loopExitInst =
        cfGraph->findLoopInstPoints(BPatch_locLoopEndIter, loopToInstrument);
    // BPatch_Vector<BPatch_point*>* loopExitInst =
    // cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);

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

    info1 = mutateeImage->getSourceLines(baseAddr, lines);

    if(info1)
    {
        filename = lines[0].fileName();
        row1     = lines[0].lineNumber();
        col1     = lines[0].lineOffset();
        if(col1 < 0)
            col1 = 0;

        //      info2 = mutateeImage->getSourceLines((unsigned long) (lastAddr -1),
        //      lines);

        // This following section is attempting to remedy the limitations of
        // getSourceLines for loops. As the program goes through the loop, the resulting
        // lines go from the loop head, through the instructions present in the loop, to
        // the last instruction in the loop, back to the loop head, then to the next
        // instruction outside of the loop. What this section does is starts at the last
        // instruction in the loop, then goes through the addresses until it reaches the
        // next instruction outside of the loop. We then bump back a line. This is not a
        // perfect solution, but we will work with the Dyninst team to find something
        // better.
        info2 = mutateeImage->getSourceLines((unsigned long) lastAddr, linesEnd);
        dprintf("size of linesEnd = %lu\n", (unsigned long) linesEnd.size());

        if(info2)
        {
            row2 = linesEnd[0].lineNumber();
            col2 = linesEnd[0].lineOffset();
            if(col2 < 0)
                col2 = 0;
            if(row2 < row1)
                row1 = row2; /* Fix for wrong line numbers*/
            sprintf(newname, "Loop: %s %s() [{%s} {%d,%d}-{%d,%d}]", typeName, fname,
                    filename, row1, col1, row2, col2);
        }
        else
        {
            sprintf(newname, "Loop: %s %s() [{%s} {%d,%d}]", typeName, fname, filename,
                    row1, col1);
        }
    }
    else
    {
        strcpy(newname, fname);
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
insert_trace(BPatch_function* functionToInstrument, BPatch_addressSpace* mutatee,
             BPatch_function* traceEntryFunc, BPatch_function* traceExitFunc,
             BPatch_flowGraph* cfGraph, BPatch_basicBlockLoop* loopToInstrument)
{
    char name[1024];
    char modname[1024];

    functionToInstrument->getModuleName(modname, 1024);

    get_loop_file_line_info(mutatee->getImage(), cfGraph, loopToInstrument,
                            functionToInstrument, name);

    BPatch_module* module = functionToInstrument->getModule();
    tim::consume_parameters(module);

    if(strstr(modname, "libdyninstAPI_RT"))
        return;

    //  functionToInstrument->getName(name, 1024);

    int                            id = add_name(name);
    BPatch_Vector<BPatch_snippet*> traceArgs;
    traceArgs.push_back(new BPatch_constExpr(id));

    BPatch_Vector<BPatch_point*>* loopEntr =
        cfGraph->findLoopInstPoints(BPatch_locLoopEntry, loopToInstrument);
    BPatch_Vector<BPatch_point*>* loopExit =
        cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);

    BPatch_Vector<BPatch_snippet*> entryTraceArgs;
    entryTraceArgs.push_back(new BPatch_constExpr(name));
    entryTraceArgs.push_back(new BPatch_constExpr(id));

    BPatch_funcCallExpr entryTrace(*traceEntryFunc, entryTraceArgs);
    BPatch_funcCallExpr exitTrace(*traceExitFunc, traceArgs);

    if(loopEntr->size() == 0)
    {
        printf("Failed to instrument loop entry in %s\n", name);
    }
    else
    {
        for(size_t i = 0; i < loopEntr->size(); i++)
        {
            mutatee->insertSnippet(entryTrace, loopEntr[i], BPatch_callBefore,
                                   BPatch_lastSnippet);
        }
    }

    if(loopExit->size() == 0)
    {
        printf("Failed to instrument loop exit in %s\n", name);
    }
    else
    {
        for(size_t i = 0; i < loopExit->size(); i++)
        {
            mutatee->insertSnippet(exitTrace, loopExit[i], BPatch_callBefore,
                                   BPatch_lastSnippet);
        }
    }
}

//======================================================================================//

void
insert_trace(BPatch_function* functionToInstrument, BPatch_addressSpace* mutatee,
             BPatch_function* traceEntryFunc, BPatch_function* traceExitFunc)
{
    char name[1024];
    char modname[1024];

    functionToInstrument->getModuleName(modname, 1024);
    if(strstr(modname, "libdyninstAPI_RT"))
        return;

    // functionToInstrument->getName(name, 1024);
    get_func_file_line_info(mutatee->getImage(), functionToInstrument, name);

    // int                            id = add_name(name);
    // tim::consume_parameters(id);

    BPatch_Vector<BPatch_point*>* funcEntry =
        functionToInstrument->findPoint(BPatch_entry);
    BPatch_Vector<BPatch_point*>* funcExit = functionToInstrument->findPoint(BPatch_exit);

    BPatch_Vector<BPatch_snippet*> entryTraceArgs;
    BPatch_Vector<BPatch_snippet*> exitTraceArgs;

    auto ret = new BPatch_retExpr();
    entryTraceArgs.push_back(new BPatch_constExpr(name));
    entryTraceArgs.push_back(ret);
    exitTraceArgs.push_back(new BPatch_constExpr(name));
    // exitTraceArgs.push_back(ret);

    BPatch_funcCallExpr entryTrace(*traceEntryFunc, entryTraceArgs);
    BPatch_funcCallExpr exitTrace(*traceExitFunc, exitTraceArgs);

    mutatee->insertSnippet(entryTrace, *funcEntry, BPatch_callBefore, BPatch_lastSnippet);
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
    }  // if
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
    tim::consume_parameters(appImage);
    // First create the snippet using the callee and the args
    const BPatch_snippet* snippet = new BPatch_funcCallExpr(*callee, *callee_args);
    if(callee)
    {
        auto name = callee->getDemangledName();
        PRINT_HERE("name: %s", name.c_str());
    }

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
    tim::consume_parameters(appImage);
    if(callee)
    {
        auto name = callee->getDemangledName();
        PRINT_HERE("name: %s", name.c_str());
    }
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
        appThread->insertSnippet(*snippet, points, BPatch_callAfter);
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
    BPatch_function* call_func = find_function(appImage, "timemory_init_trace");
    if(call_func == nullptr)
    {
        fprintf(stderr, "Unable to find function timemory_init_trace\n");
        exit(1);
    }

    BPatch_funcCallExpr call_Expr(*call_func, initArgs);
    if(binaryRewrite)
    {
        // check_cost(call_Expr);
        // locate the entry point for main
        BPatch_function* main_entry = find_function(appImage, "main");
        if(main_entry == nullptr)
        {
            fprintf(stderr, "timemory-run: Unable to find function main\n");
            exit(1);
        }
        const BPatch_Vector<BPatch_point*>* points = main_entry->findPoint(BPatch_entry);
        const BPatch_snippet* snippet = new BPatch_funcCallExpr(*call_func, initArgs);
        // We invoke the Init snippet before any other call in main!
        if((points != nullptr) && (snippet != nullptr))
        {
            // Insert the given snippet at the given point
            appThread->insertSnippet(*snippet, *points, BPatch_callBefore,
                                     BPatch_firstSnippet);
        }
        else
        {
            fprintf(stderr,
                    "timemory-run: entry points for main or snippet for Init are null\n");
            exit(1);
        }
    }
    else
    {
        appThread->oneTimeCode(call_Expr);
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

int errorPrint = 0;  // external "dyninst" tracing

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
    tim::consume_parameters(level, num, params);
    // It does nothing.
}

//======================================================================================//
// Constraints for instrumentation. Returns true for those modules that
// shouldn't be instrumented.
int
module_constraint(char* fname)
{  // fname is the name of module/file
    int len = strlen(fname);

    if(are_file_include_exclude_lists_empty())
    {  // there are no user sepecified constraints on modules. Use our default
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
            // Wait: first check if it has libtimemory* in the name!
            if(strncmp(fname, "libtimemory", 11) == 0)
            {
                return true; /* constraint applies - do not instrument! */
            }
            else
            {
                return false; /* ok to instrument */
            }
        }  // if
        else
            return true;
    }  // the selective instrumentation file lists are not empty!
    else
    {
        // See if the file should be instrumented
        if(process_file_for_instrumentation(string(fname)))
        {
            // Yes, it should be instrumented. moduleconstraint should return false!
            return false;
        }
        else
        {  // No, the file should not be instrumented. Constraint exists return true
            return true;
        }
    }
}

//======================================================================================//
// Constraint for routines. The constraint returns true for those routines that
// should not be instrumented.
int
routine_constraint(char* fname)
{
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
        if(instrument_entity(string(fname)))
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
    for(std::vector<const char*>::iterator i = names.begin(); i != names.end(); i++)
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
    for(std::vector<const char*>::iterator i = names.begin(); i != names.end(); i++)
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
    tim::consume_parameters(binaryRewrite);

    std::vector<const char*> init_names;
    init_names.push_back("MPI_Init");
    init_names.push_back("PMPI_Init");
    bool ismpi = find_func_or_calls(init_names, mpiinit, appImage, BPatch_locExit);

    mpiinitstub = find_function(appImage, "MPIInitStubInt");
    if(mpiinitstub == (BPatch_function*) nullptr)
        printf("*** MPIInitStubInt not found! \n");

    if(!ismpi)
    {
        dprintf("*** This is not an MPI Application! \n");
        return 0;  // It is not an MPI application
    }
    else
        return 1;  // Yes, it is an MPI application.
}  // check_if_mpi()

//======================================================================================//
// We create a new name that embeds the file and line information in the name
//
void
get_func_file_line_info(BPatch_image* mutateeAddressSpace, BPatch_function* f,
                        char* newname)
{
    bool          info1, info2;
    unsigned long baseAddr, lastAddr;
    char          fname[1024];
    const char*   filename;
    int           row1, col1, row2, col2;
    BPatch_type*  returnType;
    const char*   typeName;

    baseAddr = (unsigned long) (f->getBaseAddr());
    // < dyninst 8+
    // lastAddr = baseAddr + f->getSize();
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

    info1 = mutateeAddressSpace->getSourceLines((unsigned long) baseAddr, lines);

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
        info2 =
            mutateeAddressSpace->getSourceLines((unsigned long) (lastAddr - 1), lines);
        if(info2)
        {
            row2 = lines[1].lineNumber();
            col2 = lines[1].lineOffset();
            if(col2 < 0)
                col2 = 0;
            if(row2 < row1)
                row1 = row2;
            sprintf(newname, "%s %s()/%s:%i", typeName, fname, file.c_str(), row1);
        }
        else
        {
            sprintf(newname, "%s %s()/%s:%i", typeName, fname, file.c_str(), row1);
            // sprintf(newname, "%s %s() [{%s} {%d,%d}]", typeName, fname, filename, row1,
            //        col1);
        }
    }
    else
        strcpy(newname, fname);
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
               "dependent static "
               "libraries for static binary");
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
                        char* libname, char* staticlibname, char* bindings)
{
    using namespace std;
    BPatch_Vector<BPatch_point*> mpiinit;
    BPatch_function*             mpiinitstub = nullptr;

    dprintf("Inside timemory_rewrite_binary, name=%s, out=%s\n", mutateeName, outfile);
    BPatch_binaryEdit* mutateeAddressSpace = bpatch->openBinary(mutateeName, false);

    if(mutateeAddressSpace == nullptr)
    {
        fprintf(stderr, "Failed to open binary %s\n", mutateeName);
        return -1;
    }

    BPatch_image*                    mutateeImage = mutateeAddressSpace->getImage();
    BPatch_Vector<BPatch_function*>* allFuncs     = mutateeImage->getProcedures();
    bool                             isStaticExecutable;

    isStaticExecutable = mutateeAddressSpace->isStaticExecutable();

    if(isStaticExecutable)
    {
        bool result = mutateeAddressSpace->loadLibrary(staticlibname);
        dprintf("staticlibname loaded result = %d\n", result);
        assert(result);
    }
    else
    {
        bool result = mutateeAddressSpace->loadLibrary(libname);
        if(!result)
        {
            printf("Error: loadLibrary(%s) failed. Please ensure that timemory's lib "
                   "directory "
                   "is in your LD_LIBRARY_PATH environment variable and retry.\n",
                   libname);
            printf("You may also want to use timemory-exec while launching the rewritten "
                   "binary. "
                   "If timemory relies on some external libraries (Score-P), these may "
                   "need to "
                   "specified as timemory-exec -loadlib=/path/to/library <mutatee> \n");
        }
        assert(result);
    }

    BPatch_function* entryTrace = find_function(mutateeImage, "timemory_register_trace");
    BPatch_function* exitTrace = find_function(mutateeImage, "timemory_deregister_trace");
    BPatch_function* setupFunc = find_function(mutateeImage, "timemory_dyninst_init");
    BPatch_function* cleanupFunc =
        find_function(mutateeImage, "timemory_dyninst_finalize");
    BPatch_function* mainFunc = find_function(mutateeImage, "main");
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
        fprintf(stderr, "Couldn't find main(), aborting\n");
        return -1;
    }

    if(!entryTrace || !exitTrace || !setupFunc || !cleanupFunc)
    {
        fprintf(stderr, "Couldn't find OTF functions, aborting\n");
        return -1;
    }

    BPatch_Vector<BPatch_point*>* mainEntry = mainFunc->findPoint(BPatch_entry);
    assert(mainEntry);
    assert(mainEntry->size());
    assert((*mainEntry)[0]);

    mutateeAddressSpace->beginInsertionSet();

    int              ismpi = check_if_mpi(mutateeImage, mpiinit, mpiinitstub, true);
    BPatch_constExpr isMPI(ismpi);
    BPatch_Vector<BPatch_snippet*> init_params;
    init_params.push_back(&isMPI);
    BPatch_funcCallExpr setup_call(*setupFunc, init_params);
    funcNames.push_back(&setup_call);

    if(ismpi && mpiinitstub)
    {
        // Create a snippet that calls MPIInitStub with the rank after MPI_Init
        //   BPatch_function *mpi_rank = find_function(mutateeImage, "mpi_getRank");
        BPatch_function* mpi_rank = find_function(mutateeImage, "GetMpiRank");
        assert(mpi_rank);
        BPatch_Vector<BPatch_snippet*> rank_args;
        BPatch_funcCallExpr            getrank(*mpi_rank, rank_args);
        BPatch_Vector<BPatch_snippet*> mpiinitargs;
        mpiinitargs.push_back(&getrank);
        BPatch_funcCallExpr initmpi(*mpiinitstub, mpiinitargs);

        mutateeAddressSpace->insertSnippet(initmpi, mpiinit, BPatch_callAfter,
                                           BPatch_firstSnippet);
    }

    for(auto it = allFuncs->begin(); it != allFuncs->end(); ++it)
    {
        char fname[FUNCNAMELEN];
        (*it)->getName(fname, FUNCNAMELEN);
        // dprintf("Processing %s...\n", fname);

        bool okayToInstr            = true;
        bool instRoutineAtLoopLevel = false;

        // Goes through the vector of timemoryInstrument to check that the
        // current routine is one that has been passed in the selective instrumentation
        // file
        /*
        for(std::vector<timemoryInstrument*>::iterator instIt = instrumentList.begin();
            instIt != instrumentList.end(); instIt++)
        {
            if((*instIt)->getRoutineSpecified())
            {
                const char* instRName = (*instIt)->getRoutineName().c_str();
                dprintf("Examining %s... \n", instRName);

                if(match_name((*instIt)->getRoutineName(), string(fname)))
                {
                    instRoutineAtLoopLevel = true;
                    dprintf("True: instrumenting %s at the loop level\n", instRName);
                }
            }
        }
        */
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
                        okayToInstr = false;
                }
            }
        }

        if(okayToInstr && !routine_constraint(fname))
        {  // ok to instrument

            insert_trace(*it, mutateeAddressSpace, entryTrace, exitTrace);
        }
        else
        {
            // dprintf("Not instrumenting %s\n", fname);
        }

        if(okayToInstr && !routine_constraint(fname) &&
           instRoutineAtLoopLevel)  // Only occurs when we've defined that the selective
                                    // file is for loop instrumentation
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
                insert_trace(*it, mutateeAddressSpace, entryTrace, exitTrace, flow,
                             *loopIt);
            }
        }
    }

    BPatch_sequence sequence(funcNames);
    mutateeAddressSpace->insertSnippet(sequence, *mainEntry, BPatch_callBefore,
                                       BPatch_firstSnippet);
    mutateeAddressSpace->finalizeInsertionSet(false, nullptr);

    if(isStaticExecutable)
    {
        bool loadResult = load_dependent_libraries(mutateeAddressSpace, bindings);
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

    mutateeAddressSpace->writeFile(modifiedFileName.c_str());
    if(!isStaticExecutable)
    {
        unlink(libname);
        /* remove libtimemory.so in the current directory. It interferes */
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
    int  instrumented = 0;      // count of instrumented functions
    int  errflag      = 0;      // determine if error has occured.  default 0
    bool loadlib      = false;  // do we have a library loaded? default false
    char mutname[MUTNAMELEN];   // variable to hold mutator name (ie timemory-run)
    char outfile[MUTNAMELEN];   // variable to hold output file
    char fname[FUNCNAMELEN];
    char libname[FUNCNAMELEN];  // function name and library name variables
    char staticlibname[FUNCNAMELEN];
    BPatch_process*              appThread = nullptr;  // application thread
    BPatch_Vector<BPatch_point*> mpiinit;
    BPatch_function*             mpiinitstub = nullptr;
    bpatch                                   = new BPatch;  // create a new version.
    string functions;  // string variable to hold function names
    // commandline option processing args
    int   vflag  = 0;
    char* xvalue = nullptr;
    char* tvalue = nullptr;
    char* fvalue = nullptr;
    char* ovalue = nullptr;
    int   index;
    int   c;

    // bpatch->setTrampRecursive(true); /* enable C++ support */
    // bpatch->setBaseTrampDeletion(true);
    // bpatch->setMergeTramp(false);
    bpatch->setSaveFPR(true);

    // parse the command line arguments--first, there need to be atleast two arguments,
    // the program name (timemory-run) and the application it is running.  If there are
    // not at least two arguments, set the error flag.
    if(argc < 2)
        errflag = 1;
    // now can loop through the options.  If the first character is '-', then we know we
    // have an option.  Check to see if it is one of our options and process it.  If it is
    // unrecognized, then set the errflag to report an error.  When we come to a non '-'
    // charcter, then we must be at the application name.
    else
    {
        opterr = 0;

        while((c = getopt(argc, argv, "vT:X:o:f:d:")) != -1)
            switch(c)
            {
                case 'v':
                    vflag      = 1;
                    debugPrint = 1; /* Verbose option set */
                    break;
                case 'X':
                    xvalue  = optarg;
                    loadlib = true; /* load an external measurement library */
                    break;
                case 'T':
                    tvalue  = optarg;
                    loadlib = true; /* load an external measurement library */
                    break;
                case 'f':
                    fvalue = optarg; /* choose a selective instrumentation file */
                    // process_instrumentation_requests(fvalue, instrumentList);
                    dprintf("Loading instrumentation requests file %s\n", fvalue);
                    break;
                case 'o':
                    ovalue        = optarg;
                    binaryRewrite = 1; /* binary rewrite is true */
                    strcpy(outfile, ovalue);
                    break;
                case '?':
                    if(optopt == 'X' || optopt == 'f' || optopt == 'o')
                        fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                    else if(isprint(optopt))
                        fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                    else
                        fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                    errflag = 1;
                    break;
                default: errflag = 1;
            }

        dprintf("vflag = %d, xvalue = %s, ovalue = %s, fvalue = %s, tvalue = %s\n", vflag,
                xvalue, ovalue, fvalue, tvalue);

        strncpy(mutname, argv[optind], strlen(argv[optind]) + 1);
        for(index = optind; index < argc; index++)
            dprintf("Non-option argument %s\n", argv[index]);
    }

    char* bindings = (char*) malloc(1024);
    char  cmd[1024];
    char  bindir[] = TIMEMORY_BIN_DIR;
    dprintf("mutatee name = %s\n", mutname);
    if(tvalue != (char*) nullptr)
    {
        dprintf("-T <options> specified\n");
        sprintf(
            cmd,
            "echo %s | sed -e 's@,@ @g' | tr '[A-Z]' '[a-z]' | xargs %s/timemory-config "
            "--binding | sed -e 's@shared@@g'",
            tvalue, bindir);
        FILE* fp;
        fp = popen(cmd, "r");
        if(fp == nullptr)
        {
            perror("timemory-run: Error launching timemory-config to get bindings");
            return 1;
        }

        if((fgets(bindings, 1024, fp)) != nullptr)
        {
            int len = strlen(bindings);
            if(strstr(bindings, "Error") != 0)
            {
                printf("timemory error: %s\n", bindings);
                return 1;
            }
            if(bindings[len - 1] == '\n')
            {
                bindings[len - 1] = '\0';
            }
            /* we have the bindings: print it */
            dprintf("bindings: %s\n", bindings);
        }
        else
        {
            perror("timemory-run: Error reading from pipe to get bindings from "
                   "timemory-config");
            return 1;
        }
        pclose(fp);
    }

    // did we load a library?  if not, load the default
    if(!loadlib)
    {
        sprintf(staticlibname, "libtimemory.a");
        sprintf(libname, "libtimemory.so");
        loadlib = true;
    }

    // has an error occured in the command line arguments?
    if(errflag)
    {
        fprintf(stderr,
                "usage: %s [-Xrun<library> | -T <bindings_options> ] [-v] [-o "
                "outfile] [-f <inst_req> ] <application> [args]\n",
                argv[0]);
        fprintf(
            stderr,
            "%s instruments and executes <application> to generate performance data\n",
            argv[0]);
        fprintf(stderr, "-v is an optional verbose option\n");
        fprintf(stderr, "-o <outfile> is for binary rewriting\n");
        fprintf(stderr, "e.g., \n");
        fprintf(stderr, "%%%s -Xruntimemory -f sel.dat a.out 100 \n", argv[0]);
        fprintf(
            stderr,
            "Loads libtimemory.so from $LD_LIBRARY_PATH, loads selective instrumentation "
            "requests from file sel.dat and executes a.out \n");
        fprintf(stderr, "%%%s -T scorep,papi,pdt -f sel.dat a.out 100 \n", argv[0]);
        fprintf(
            stderr,
            "Loads libtimemory.so from $LD_LIBRARY_PATH, loads "
            "selective instrumentation requests from file sel.dat and executes a.out \n");
        exit(1);
    }

    // Register a callback function that prints any error messages
    bpatch->registerErrorCallback(error_func_real);

    dprintf("Before createProcess\n");
    // Specially added to disable Dyninst 2.0 feature of debug parsing. We were
    // getting an assertion failure under Linux otherwise
    // bpatch->setDebugParsing(false);
    // removed for DyninstAPI 4.0

    if(binaryRewrite)
    {
        timemory_rewrite_binary(bpatch, mutname, outfile, (char*) libname,
                                (char*) staticlibname, bindings);
        // exit from the application
        return 0;
    }

    // DYNINST41PLUS
    appThread =
        bpatch->processCreate(argv[optind], (const char**) &argv[optind], nullptr);
    // appThread          = bpatch->createProcess(argv[optind], &argv[optind], nullptr);

    dprintf("After createProcess\n");

    if(!appThread)
    {
        printf("timemory-run> createProcess failed\n");
        exit(1);
    }

    if(binaryRewrite)
    {
        // enable dumping
        appThread->enableDumpPatchedImage();
    }

    // get image
    BPatch_image*                  appImage = appThread->getImage();
    BPatch_Vector<BPatch_module*>* m        = appImage->getModules();

    // Load the library that has entry and exit routines.
    // Do not load the timemory library if we're rewriting the binary. Use LD_PRELOAD
    // instead. The library may be loaded at a different location.
    if(loadlib == true)
    {
        // try and load the library
        bool ret = appThread->loadLibrary(libname, true);
        if(ret == true)
        {
            // now, check to see if the library is listed as a module in the
            // application image
            char name[FUNCNAMELEN];
            bool found = false;
            for(size_t i = 0; i < m->size(); i++)
            {
                (*m)[i]->getName(name, sizeof(name));
                if(strcmp(name, libname) == 0)
                {
                    found = true;
                    break;
                }
            }
            if(found)
            {
                dprintf("%s loaded properly\n", libname);
            }
            else
            {
                printf("Error in loading library %s\n", libname);
                // exit(1);
            }
        }
        else
        {
            printf("ERROR:%s not loaded properly. \n", libname);
            printf("Please make sure that its path is in your LD_LIBRARY_PATH "
                   "environment variable.\n");
            exit(1);
        }
    }

    BPatch_function* inFunc;
    BPatch_function* enterstub = find_function(appImage, "timemory_register_trace");
    BPatch_function* exitstub  = find_function(appImage, "timemory_deregister_trace");
    BPatch_function* terminationstub =
        find_function(appImage, "timemory_finalize_library");
    BPatch_Vector<BPatch_snippet*> initArgs;

    char modulename[256];
    for(size_t j = 0; j < m->size(); j++)
    {
        // sprintf(modulename, "Module %s\n", (*m)[j]->getName(fname, FUNCNAMELEN));
        BPatch_Vector<BPatch_function*>* p = (*m)[j]->getProcedures();
        // dprintf("%s", modulename);

        if(!module_constraint(fname))
        {  // constraint
            for(size_t i = 0; i < p->size(); i++)
            {
                // For all procedures within the module, iterate
                (*p)[i]->getName(fname, FUNCNAMELEN);
                // dprintf("Name %s\n", fname);
                if(routine_constraint(fname))
                {
                    // The above procedures shouldn't be instrumented
                    // dprintf("don't instrument %s\n", fname);
                }
                else
                {
                    // routines that are ok to instrument
                    // get full source information
                    dprintf("Instrumenting %s\n", fname);
                    get_func_file_line_info(appImage, (*p)[i], fname);
                    functions.append("|");
                    functions.append(fname);
                }
            }
        }
    }

    std::cout << "functions: " << functions << std::endl;
    // form the args to InitCode
    BPatch_constExpr funcName(functions.c_str());

    // When we look for MPI calls, we shouldn't display an error message for
    // not find MPI_Comm_rank in the case of a sequential app. So, we turn the
    // Error callback to be Null and turn back the error settings later. This
    // way, it works for both MPI and sequential tasks.

    bpatch->registerErrorCallback(error_func_fake);  // turn off error reporting
    BPatch_constExpr isMPI(check_if_mpi(appImage, mpiinit, mpiinitstub, binaryRewrite));
    bpatch->registerErrorCallback(error_func_real);  // turn it back on

    initArgs.push_back(&funcName);
    initArgs.push_back(&isMPI);

    initialize(appThread, appImage, initArgs);
    dprintf("Did initialize\n");

    // In our tests, the routines started execution concurrently with the
    // one time code. To avoid this, we first start the one time code and then
    // iterate through the list of routines to select for instrumentation and
    // instrument these. So, we need to iterate twice.

    for(size_t j = 0; j < m->size(); j++)
    {
        // sprintf(modulename, "Module %s\n", (*m)[j]->getName(fname, FUNCNAMELEN));
        BPatch_Vector<BPatch_function*>* p = (*m)[j]->getProcedures();
        dprintf("%s", modulename);

        if(!module_constraint(fname))
        {
            // constraint
            for(size_t i = 0; i < p->size(); i++)
            {
                // For all procedures within the module, iterate
                (*p)[i]->getName(fname, FUNCNAMELEN);
                // dprintf("Name %s\n", fname);
                if(routine_constraint(fname))
                {
                    // The above procedures shouldn't be instrumented
                    // dprintf("don't instrument %s\n", fname);
                }
                else
                {  // routines that are ok to instrument
                    dprintf("Assigning id %d to %s\n", instrumented, fname);
                    instrumented++;
                    BPatch_Vector<BPatch_snippet*>* callee_entry_args =
                        new BPatch_Vector<BPatch_snippet*>();
                    BPatch_Vector<BPatch_snippet*>* callee_exit_args =
                        new BPatch_Vector<BPatch_snippet*>();
                    BPatch_constExpr* constExprFunc = new BPatch_constExpr(instrumented);
                    BPatch_constExpr* constExprId   = new BPatch_constExpr(instrumented);

                    callee_entry_args->push_back(constExprFunc);
                    callee_entry_args->push_back(constExprId);
                    callee_exit_args->push_back(constExprId);

                    inFunc = (*p)[i];
                    dprintf("Instrumenting-> %s Entry\n", fname);
                    invoke_routine_in_func(appThread, appImage, inFunc, BPatch_entry,
                                           enterstub, callee_entry_args);
                    dprintf("Instrumenting-> %s Exit...", fname);
                    invoke_routine_in_func(appThread, appImage, inFunc, BPatch_exit,
                                           exitstub, callee_exit_args);
                    dprintf("Done\n");
                    delete callee_entry_args;
                    delete callee_exit_args;
                    delete constExprFunc;
                    delete constExprId;
                }  // else -- routines that are ok to instrument
            }      // for -- procedures
        }          // if -- module constraint
    }              // for -- modules

    BPatch_function* exitpoint = find_function(appImage, "exit");
    if(exitpoint == nullptr)
        exitpoint = find_function(appImage, "_exit");

    if(exitpoint == nullptr)
    {
        fprintf(stderr, "UNABLE TO FIND exit() \n");
        // exit(1);
    }
    else
    {
        // When _exit is invoked, call ProgramTermination routine
        BPatch_Vector<BPatch_snippet*>* exitargs = new BPatch_Vector<BPatch_snippet*>();
        BPatch_constExpr*               Name     = new BPatch_constExpr("_exit");
        exitargs->push_back(Name);
        invoke_routine_in_func(appThread, appImage, exitpoint, BPatch_entry,
                               terminationstub, exitargs);
        delete exitargs;
        delete Name;
    }  // else

    if(!mpiinit.size())
    {
        dprintf("*** MPI_Init not found. This is not an MPI Application! \n");
    }
    else
    {  // we've found either MPI_Comm_rank or PMPI_Comm_rank!
        dprintf("FOUND MPI_Comm_rank or PMPI_Comm_rank! \n");
        BPatch_Vector<BPatch_snippet*>* mpistubargs =
            new BPatch_Vector<BPatch_snippet*>();
        BPatch_paramExpr paramRank(1);

        mpistubargs->push_back(&paramRank);
        invoke_routine_in_func(appThread, appImage, mpiinit, mpiinitstub, mpistubargs);
        delete mpistubargs;
    }

    /* check to see if we have to rewrite the binary image */
    if(binaryRewrite)
    {
        // DYNINSTAPI_8_PLUS
        const char* directory = ".";  // appThread->dumpImage("a.inst");
        // char* directory = appThread->dumpPatchedImage(outfile);
        /* see if it was rewritten properly */
        if(directory)
        {
            printf("The instrumented executable image is stored in %s directory\n",
                   directory);
        }
        else
        {
            fprintf(stderr, "Error: Binary rewriting did not work: No directory name \
returned\n\nIf you are using Dyninst 5.2 this feature is no longer \
supported and \
timemory-run will run the application using dynamic instrumentation....\n");
        }
        delete bpatch;
        return 0;
    }

    dprintf("Executing...\n");
    appThread->continueExecution();

    while(!appThread->isTerminated())
    {
        bpatch->waitForStatusChange();
        sleep(1);
    }  // while

    if(appThread->isTerminated())
    {
        dprintf("End of application\n");
    }  // if

    // cleanup
    delete bpatch;
    return 0;
}
