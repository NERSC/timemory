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
    "cpu_roofline_sp_flops",
    "cpu_roofline_dp_flops",
    "caliper",
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
    "caliper": ["cali"]
}


conditional_types = {
    "papi": ["papi_array_t", "cpu_roofline_sp_flops", "cpu_roofline_dp_flops"],
    "cuda": ["cuda_event"],
}
