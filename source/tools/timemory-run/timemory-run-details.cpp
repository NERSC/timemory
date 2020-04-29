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

#include "timemory-run.hpp"

//======================================================================================//
//
//  For selective instrumentation (unused)
//
bool
are_file_include_exclude_lists_empty()
{
    return collection_includes.empty() && collection_excludes.empty();
}

//======================================================================================//
//
//  Gets information (line number, filename, and column number) about
//  the instrumented loop and formats it properly.
//
function_signature
get_loop_file_line_info(BPatch_image* mutateeImage, BPatch_function* f,
                        BPatch_flowGraph*      cfGraph,
                        BPatch_basicBlockLoop* loopToInstrument)
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
    verbprintf(0, "Loop: size of lastAddr = %lu: baseAddr = %lu, lastAddr = %lu\n",
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
        verbprintf(0, "size of linesEnd = %lu\n", (unsigned long) linesEnd.size());

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
//
//  We create a new name that embeds the file and line information in the name
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
//  Error callback routine.
//
void
errorFunc(BPatchErrorLevel level, int num, const char** params)
{
    char line[256];

    const char* msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    if(num != expect_error)
    {
        printf("Error #%d (level %d): %s\n", num, level, line);
        // We consider some errors fatal.
        if(num == 101)
            exit(-1);
    }
}

//======================================================================================//
//
//  For compatibility purposes
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
        verbprintf(0, "timemory-run: Unable to find function %s\n", functionName);
        return nullptr;
    }
    return found_funcs[0];
}

//======================================================================================//
//
//   check that the cost of a snippet is sane.  Due to differences between
//   platforms, it is impossible to check this exactly in a machine independent
//   manner.
//
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
//
void
error_func_real(BPatchErrorLevel level, int num, const char* const* params)
{
    if(num == 0)
    {
        // conditional reporting of warnings and informational messages
        if(error_print)
        {
            if(level == BPatchInfo)
            {
                if(error_print > 1)
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
        if(num != expect_error)
        {
            printf("Error #%d (level %d): %s\n", num, level, line);
            // We consider some errors fatal.
            if(num == 101)
                exit(-1);
        }
    }
}

//======================================================================================//
//
//  We've a null error function when we don't want to display an error
//
void
error_func_fake(BPatchErrorLevel level, int num, const char* const* params)
{
    consume_parameters(level, num, params);
    // It does nothing.
}

//======================================================================================//
//
bool
find_func_or_calls(std::vector<const char*> names, BPatch_Vector<BPatch_point*>& points,
                   BPatch_image* appImage, BPatch_procedureLocation loc)
{
    using function_t     = BPatch_function;
    using point_t        = BPatch_point;
    using function_vec_t = BPatch_Vector<function_t*>;
    using point_vec_t    = BPatch_Vector<point_t*>;

    function_t* func = nullptr;
    for(auto nitr = names.begin(); nitr != names.end(); ++nitr)
    {
        function_t* f = find_function(appImage, *nitr);
        if(f && f->getModule()->isSharedLib())
        {
            func = f;
            break;
        }
    }

    if(func)
    {
        point_vec_t* fpoints = func->findPoint(loc);
        if(fpoints && fpoints->size())
        {
            for(auto pitr = fpoints->begin(); pitr != fpoints->end(); ++pitr)
                points.push_back(*pitr);
            return true;
        }
    }

    // Moderately expensive loop here.  Perhaps we should make a name->point map first
    // and just do lookups through that.
    function_vec_t* all_funcs           = appImage->getProcedures();
    auto            initial_points_size = points.size();
    for(auto nitr = names.begin(); nitr != names.end(); ++nitr)
    {
        for(auto fitr = all_funcs->begin(); fitr != all_funcs->end(); ++fitr)
        {
            function_t* f = *fitr;
            if(f->getModule()->isSharedLib())
                continue;
            point_vec_t* fpoints = f->findPoint(BPatch_locSubroutine);
            if(!fpoints || fpoints->empty())
                continue;
            for(auto pitr = fpoints->begin(); pitr != fpoints->end(); pitr++)
            {
                std::string callee = (*pitr)->getCalledFunctionName();
                if(callee == std::string(*nitr))
                    points.push_back(*pitr);
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
                   BPatch_image* image, BPatch_procedureLocation loc)
{
    std::vector<const char*> v;
    v.push_back(name);
    return find_func_or_calls(v, points, image, loc);
}

//======================================================================================//
//
static inline void
consume()
{
    consume_parameters(initialize_expr, bpatch, use_ompt, use_mpi, use_mpip,
                       stl_func_instr, werror, loop_level_instr, error_print,
                       binary_rewrite, debug_print, expect_error, is_static_exe,
                       available_modules, available_procedures, instrumented_modules,
                       instrumented_procedures);
    if(false)
    {
        timemory_thread_exit(nullptr, ExitedNormally);
        timemory_fork_callback(nullptr, nullptr);
    }
}
