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

#include <regex>
#include <string>

bool
check_if_timemory_source_file(const std::string& fname)
{
    // clang-format off
    //
    std::regex file_regex("(elf_ops.c|gotcha.c|gotcha_auxv.c|gotcha_dl.c|gotcha_utils.c|hash.c|libc_wrappers.c|library_filters.c|tool.c|translations.c|Annotation.cpp|AnnotationBinding.cpp|Blackboard.cpp|Caliper.cpp|ChannelController.cpp|ConfigManager.cpp|MemoryPool.cpp|MetadataTree.cpp|RegionProfile.cpp|SnapshotRecord.cpp|api.cpp|cali.cpp|cali_datatracker.cpp|config_sanity_check.cpp|CudaActivityController.cpp|EventTraceController.cpp|HatchetRegionProfileController.cpp|HatchetSampleProfileController.cpp|NvProfController.cpp|RuntimeReportController.cpp|controllers.cpp|Attribute.cpp|CaliperMetadataAccessInterface.cpp|CompressedSnapshotRecord.cpp|Entry.cpp|Log.cpp|Node.cpp|NodeBuffer.cpp|OutputStream.cpp|RuntimeConfig.cpp|SnapshotBuffer.cpp|SnapshotTextFormatter.cpp|StringConverter.cpp|Variant.cpp|cali_types.c|cali_variant.c|format_util.cpp|parse_util.cpp|unitfmt.c|vlenc.c|aggregate_over_mpi.cpp|SpotController.cpp|SpotV1Controller.cpp|MpiReport.cpp|MpitServiceMPI.cpp|MpiTracing.cpp|MpiWrap.cpp|tau.cpp|setup_mpi.cpp|Aggregator.cpp|CalQLParser.cpp|CaliReader.cpp|CaliWriter.cpp|CaliperMetadataDB.cpp|Expand.cpp|FlatExclusiveRegionProfile.cpp|FlatInclusiveRegionProfile.cpp|FormatProcessor.cpp|JsonFormatter.cpp|JsonSplitFormatter.cpp|NestedExclusiveRegionProfile.cpp|NestedInclusiveRegionProfile.cpp|QueryProcessor.cpp|QuerySpec.cpp|RecordSelector.cpp|SnapshotTree.cpp|TableFormatter.cpp|TreeFormatter.cpp|UserFormatter.cpp|Services.cpp|AdiakExport.cpp|AdiakImport.cpp|Aggregate.cpp|AggregationDB.cpp|AllocService.cpp|Callpath.cpp|CpuInfo.cpp|Cupti.cpp|CuptiEventSampling.cpp|CuptiTrace.cpp|Debug.cpp|EnvironmentInfo.cpp|EventTrigger.cpp|InstLookup.cpp|callbacks.c|curious.c|dynamic_array.c|file_registry.c|mount_tree.c|wrappers.c|IOService.cpp|KokkosLookup.cpp|KokkosProfilingSymbols.cpp|KokkosTime.cpp|Libpfm.cpp|perf_postprocessing.cpp|perf_util.c|MemUsageService.cpp|NVProf.cpp|OmptService.cpp|Papi.cpp|PthreadService.cpp|Recorder.cpp|Report.cpp|Sampler.cpp|Sos.cpp|Spot.cpp|Statistics.cpp|SymbolLookup.cpp|SysAllocService.cpp|TextLog.cpp|Timestamp.cpp|IntelTopdown.cpp|Trace.cpp|TraceBufferChunk.cpp|validator.cpp|VTuneBindings.cpp|AttributeExtract.cpp|cali-query.cpp|query_common.cpp|cali-stat.cpp|mpi-caliquery.cpp|Args.cpp|performance.cpp|sandbox.cpp|sandbox_json.cpp|sandbox_rtti.cpp|base.cpp|derived.cpp|sandbox_vs.cpp|cancel.cu|critical.cu|data_sharing.cu|libcall.cu|loop.cu|omp_data.cu|omptarget-nvptx.cu|parallel.cu|reduction.cu|sync.cu|task.cu|elf_common.c|rtl.cpp|device.cpp|interface.cpp|omptarget.cpp|extractExternal.cpp|kmp_affinity.cpp|kmp_alloc.cpp|kmp_atomic.cpp|kmp_barrier.cpp|kmp_cancel.cpp|kmp_csupport.cpp|kmp_debug.cpp|kmp_debugger.cpp|kmp_dispatch.cpp|kmp_environment.cpp|kmp_error.cpp|kmp_ftn_cdecl.cpp|kmp_ftn_extra.cpp|kmp_ftn_stdcall.cpp|kmp_global.cpp|kmp_gsupport.cpp|kmp_i18n.cpp|kmp_import.cpp|kmp_io.cpp|kmp_itt.cpp|kmp_lock.cpp|kmp_runtime.cpp|kmp_sched.cpp|kmp_settings.cpp|kmp_stats.cpp|kmp_stats_timing.cpp|kmp_str.cpp|kmp_stub.cpp|kmp_taskdeps.cpp|kmp_tasking.cpp|kmp_taskq.cpp|kmp_threadprivate.cpp|kmp_utility.cpp|kmp_version.cpp|kmp_wait_release.cpp|ompt-general.cpp|ompt-specific.cpp|test-touch.c|ittnotify_static.c|tsan_annotations.cpp|z_Linux_util.cpp|z_Windows_NT-586_util.cpp|z_Windows_NT_util.cpp|library.cpp|libpytimemory.cpp|extern.cpp|factory.cpp|current_peak_rss.cpp|data_rss.cpp|kernel_mode_time.cpp|num_io_in.cpp|num_io_out.cpp|num_major_page_faults.cpp|num_minor_page_faults.cpp|num_msg_recv.cpp|num_msg_sent.cpp|num_signals.cpp|num_swap.cpp|page_rss.cpp|peak_rss.cpp|priority_context_switch.cpp|read_bytes.cpp|stack_rss.cpp|user_mode_time.cpp|virtual_memory.cpp|voluntary_context_switch.cpp|written_bytes.cpp|complete_list.cpp|full_auto_timer.cpp|generic_bundle.cpp|hybrid_bundle.cpp|minimal_auto_timer.cpp|extern.cu|settings.cpp|timemory_c.c|timemory_c.cpp|kp_timemory.cpp|kp_timemory_filter.cpp|sample.cpp|timem.cpp|timemory-avail.cpp|timemory-yaml.cpp|timemory-mpip.cpp|timemory-ompt.cpp|timemory-run-regex.cpp|timemory-run-details.cpp|timemory-run.cpp|trace.cpp)");
    return std::regex_search(fname, file_regex);
    //
    // clang-format on
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
