#!/usr/bin/env python

#
# the list of components in timemory
#   see <timemory/components/types.hpp>
#
components = [
    "real_clock",
    "system_clock",
    "user_clock",
    "cpu_clock",
    "monotonic_clock",
    "monotonic_raw_clock",
    "thread_cpu_clock",
    "process_cpu_clock",
    "cpu_util",
    "process_cpu_util",
    "thread_cpu_util",
    "peak_rss",
    "current_rss",
    "stack_rss",
    "data_rss",
    "num_swap",
    "num_io_in",
    "num_io_out",
    "num_minor_page_faults",
    "num_major_page_faults",
    "num_msg_sent",
    "num_msg_recv",
    "num_signals",
    "voluntary_context_switch",
    "priority_context_switch",
    "cuda_event",
    "papi_array_t",
    "caliper",
    "trip_count",
    "read_bytes",
    "written_bytes",
    "cupti_counters",
    "cupti_activity",
    "nvtx_marker",
    "cpu_roofline_sp_flops",
    "cpu_roofline_dp_flops",
    "cpu_roofline_flops",
    "gpu_roofline_hp_flops",
    "gpu_roofline_sp_flops",
    "gpu_roofline_dp_flops",
    "gpu_roofline_flops",
]

#
# dictionary of components that have a different component
# name than the enumeration
# e.g. "component_name" : "enumeration_name"
#
mangled_enums = {
    "system_clock": "sys_clock",
    "real_clock": "wall_clock",
    "papi_array_t": "papi_array",
}

#
# dictionary of components that have a different string
# identifier than than enumeration
# e.g. "component_name" : "string_identifier"
#
mangled_strings = {
    "system_clock": ["sys_clock"],
    "papi_array_t": ["papi_array", "papi"],
    "cpu_roofline_sp_flops": ["cpu_roofline_sp", "cpu_roofline_single"],
    "cpu_roofline_dp_flops": ["cpu_roofline_dp", "cpu_roofline_double"],
    "caliper": ["cali"],
    "written_bytes": ["write_bytes"],
    "nvtx_marker": ["nvtx"],
}

recommended_types = {
    "tuple": ["real_clock", "system_clock", "user_clock", "cpu_util",
              "current_rss", "peak_rss", "read_bytes", "written_bytes",
              "num_minor_page_faults", "num_major_page_faults",
              "voluntary_context_switch", "priority_context_switch"],
    "list": ["caliper", "papi_array_t",
             "cuda_event", "nvtx_marker",
             "cupti_counters", "cupti_activity",
             "cpu_roofline_flops", "gpu_roofline_flops"],
}

traits = {
    "is_timing_category": ("std::true_type",
    [
        "real_clock",
        "system_clock",
        "user_clock",
        "cpu_clock",
        "monotonic_clock",
        "monotonic_raw_clock",
        "thread_cpu_clock",
        "process_cpu_clock",
        "cuda_event",
        "cupti_activity",
    ]),
    "is_memory_category": ("std::true_type",
    [
        "peak_rss",
        "current_rss",
        "stack_rss",
        "data_rss",
        "num_swap",
        "num_io_in",
        "num_io_out",
        "num_minor_page_faults",
        "num_major_page_faults",
        "num_msg_sent",
        "num_msg_recv",
        "num_signals",
        "voluntary_context_switch",
        "priority_context_switch",
        "read_bytes",
        "written_bytes",
    ]),
    "uses_timing_units": ("std::true_type",
    [
        "real_clock",
        "system_clock",
        "user_clock",
        "cpu_clock",
        "monotonic_clock",
        "monotonic_raw_clock",
        "thread_cpu_clock",
        "process_cpu_clock",
        "cuda_event",
        "cupti_activity",
    ]),
    "uses_memory_units": ("std::true_type",
    [
        "peak_rss",
        "current_rss",
        "stack_rss",
        "data_rss",
        "read_bytes",
        "written_bytes",
    ]),
}
