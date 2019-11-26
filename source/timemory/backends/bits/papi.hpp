//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file backends/bits/papi.hpp
 * \headerfile backends/bits/papi.hpp "timemory/backends/bits/papi.hpp"
 * This provides a fake PAPI interface to simplify compilation when PAPI is not
 * available, similar to mpi.hpp
 *
 */

#pragma once

#include <string>

// adding these definitions make things easier
#if !defined(TIMEMORY_USE_PAPI)

#    define PAPI_OK 0       /* No error */
#    define PAPI_EINVAL -1  /* Invalid argument */
#    define PAPI_ENOMEM -2  /* Insufficient memory */
#    define PAPI_ESYS -3    /* A System/C library call failed */
#    define PAPI_ECMP -4    /* Not supported by component */
#    define PAPI_ESBSTR -4  /* Backwards compatibility */
#    define PAPI_ECLOST -5  /* Access to the counters was lost or interrupted */
#    define PAPI_EBUG -6    /* Internal error, please send mail to the developers */
#    define PAPI_ENOEVNT -7 /* Event does not exist */
#    define PAPI_ECNFLCT                                                                 \
        -8 /* Event exists, but cannot be counted due to counter resource limitations \  \
            * \                                                                          \
            */
#    define PAPI_ENOTRUN -9     /* EventSet is currently not running */
#    define PAPI_EISRUN -10     /* EventSet is currently counting */
#    define PAPI_ENOEVST -11    /* No such EventSet Available */
#    define PAPI_ENOTPRESET -12 /* Event in argument is not a valid preset */
#    define PAPI_ENOCNTR -13    /* Hardware does not support performance counters */
#    define PAPI_EMISC -14      /* Unknown error code */
#    define PAPI_EPERM -15      /* Permission level does not permit operation */
#    define PAPI_ENOINIT -16    /* PAPI hasn't been initialized yet */
#    define PAPI_ENOCMP -17     /* Component Index isn't set */
#    define PAPI_ENOSUPP -18    /* Not supported */
#    define PAPI_ENOIMPL -19    /* Not implemented */
#    define PAPI_EBUF -20       /* Buffer size exceeded */
#    define PAPI_EINVAL_DOM                                                              \
        -21 /* EventSet domain is not supported for the operation \      \               \
             */
#    define PAPI_EATTR -22         /* Invalid or missing event attributes */
#    define PAPI_ECOUNT -23        /* Too many events or attributes */
#    define PAPI_ECOMBO -24        /* Bad combination of features */
#    define PAPI_ECMP_DISABLED -25 /* Component containing event is disabled */
#    define PAPI_NUM_ERRORS 26     /* Number of error messages specified in this API */

#    define PAPI_NOT_INITED 0
#    define PAPI_LOW_LEVEL_INITED 1    /* Low level has called library init */
#    define PAPI_HIGH_LEVEL_INITED 2   /* High level has called library init */
#    define PAPI_THREAD_LEVEL_INITED 4 /* Threads have been inited */

#    define PAPI_PRESET_MASK ((int) 0x80000000)
#    define PAPI_NATIVE_MASK ((int) 0x40000000)
#    define PAPI_UE_MASK ((int) 0xC0000000)
#    define PAPI_PRESET_AND_MASK 0x7FFFFFFF
#    define PAPI_NATIVE_AND_MASK 0xBFFFFFFF /* this masks just the native bit */
#    define PAPI_UE_AND_MASK 0x3FFFFFFF

#    define PAPI_MAX_PRESET_EVENTS 128 /*The maxmimum number of preset events */
#    define PAPI_MAX_USER_EVENTS 50    /*The maxmimum number of user defined events */
#    define USER_EVENT_OPERATION_LEN                                                     \
        512 /*The maximum length of the operation string for user defined events */

enum
{
    PAPI_L1_DCM_idx = 0, /*Level 1 data cache misses */
    PAPI_L1_ICM_idx,     /*Level 1 instruction cache misses */
    PAPI_L2_DCM_idx,     /*Level 2 data cache misses */
    PAPI_L2_ICM_idx,     /*Level 2 instruction cache misses */
    PAPI_L3_DCM_idx,     /*Level 3 data cache misses */
    PAPI_L3_ICM_idx,     /*Level 3 instruction cache misses */
    PAPI_L1_TCM_idx,     /*Level 1 total cache misses */
    PAPI_L2_TCM_idx,     /*Level 2 total cache misses */
    PAPI_L3_TCM_idx,     /*Level 3 total cache misses */
    PAPI_CA_SNP_idx,     /*Snoops */
    PAPI_CA_SHR_idx,     /*Request for shared cache line (SMP) */
    PAPI_CA_CLN_idx,     /*Request for clean cache line (SMP) */
    PAPI_CA_INV_idx,     /*Request for cache line Invalidation (SMP) */
    PAPI_CA_ITV_idx,     /*Request for cache line Intervention (SMP) */
    PAPI_L3_LDM_idx,     /*Level 3 load misses */
    PAPI_L3_STM_idx,     /*Level 3 store misses */
                         /* 0x10 */
    PAPI_BRU_IDL_idx,    /*Cycles branch units are idle */
    PAPI_FXU_IDL_idx,    /*Cycles integer units are idle */
    PAPI_FPU_IDL_idx,    /*Cycles floating point units are idle */
    PAPI_LSU_IDL_idx,    /*Cycles load/store units are idle */
    PAPI_TLB_DM_idx,     /*Data translation lookaside buffer misses */
    PAPI_TLB_IM_idx,     /*Instr translation lookaside buffer misses */
    PAPI_TLB_TL_idx,     /*Total translation lookaside buffer misses */
    PAPI_L1_LDM_idx,     /*Level 1 load misses */
    PAPI_L1_STM_idx,     /*Level 1 store misses */
    PAPI_L2_LDM_idx,     /*Level 2 load misses */
    PAPI_L2_STM_idx,     /*Level 2 store misses */
    PAPI_BTAC_M_idx,     /*BTAC miss */
    PAPI_PRF_DM_idx,     /*Prefetch data instruction caused a miss */
    PAPI_L3_DCH_idx,     /*Level 3 Data Cache Hit */
    PAPI_TLB_SD_idx,     /*Xlation lookaside buffer shootdowns (SMP) */
    PAPI_CSR_FAL_idx,    /*Failed store conditional instructions */
                         /* 0x20 */
    PAPI_CSR_SUC_idx,    /*Successful store conditional instructions */
    PAPI_CSR_TOT_idx,    /*Total store conditional instructions */
    PAPI_MEM_SCY_idx,    /*Cycles Stalled Waiting for Memory Access */
    PAPI_MEM_RCY_idx,    /*Cycles Stalled Waiting for Memory Read */
    PAPI_MEM_WCY_idx,    /*Cycles Stalled Waiting for Memory Write */
    PAPI_STL_ICY_idx,    /*Cycles with No Instruction Issue */
    PAPI_FUL_ICY_idx,    /*Cycles with Maximum Instruction Issue */
    PAPI_STL_CCY_idx,    /*Cycles with No Instruction Completion */
    PAPI_FUL_CCY_idx,    /*Cycles with Maximum Instruction Completion */
    PAPI_HW_INT_idx,     /*Hardware interrupts */
    PAPI_BR_UCN_idx,     /*Unconditional branch instructions executed */
    PAPI_BR_CN_idx,      /*Conditional branch instructions executed */
    PAPI_BR_TKN_idx,     /*Conditional branch instructions taken */
    PAPI_BR_NTK_idx,     /*Conditional branch instructions not taken */
    PAPI_BR_MSP_idx,     /*Conditional branch instructions mispred */
    PAPI_BR_PRC_idx,     /*Conditional branch instructions corr. pred */
                         /* 0x30 */
    PAPI_FMA_INS_idx,    /*FMA instructions completed */
    PAPI_TOT_IIS_idx,    /*Total instructions issued */
    PAPI_TOT_INS_idx,    /*Total instructions executed */
    PAPI_INT_INS_idx,    /*Integer instructions executed */
    PAPI_FP_INS_idx,     /*Floating point instructions executed */
    PAPI_LD_INS_idx,     /*Load instructions executed */
    PAPI_SR_INS_idx,     /*Store instructions executed */
    PAPI_BR_INS_idx,     /*Total branch instructions executed */
    PAPI_VEC_INS_idx,    /*Vector/SIMD instructions executed (could include integer) */
    PAPI_RES_STL_idx,    /*Cycles processor is stalled on resource */
    PAPI_FP_STAL_idx,    /*Cycles any FP units are stalled */
    PAPI_TOT_CYC_idx,    /*Total cycles executed */
    PAPI_LST_INS_idx,    /*Total load/store inst. executed */
    PAPI_SYC_INS_idx,    /*Sync. inst. executed */
    PAPI_L1_DCH_idx,     /*L1 D Cache Hit */
    PAPI_L2_DCH_idx,     /*L2 D Cache Hit */
    /* 0x40 */
    PAPI_L1_DCA_idx, /*L1 D Cache Access */
    PAPI_L2_DCA_idx, /*L2 D Cache Access */
    PAPI_L3_DCA_idx, /*L3 D Cache Access */
    PAPI_L1_DCR_idx, /*L1 D Cache Read */
    PAPI_L2_DCR_idx, /*L2 D Cache Read */
    PAPI_L3_DCR_idx, /*L3 D Cache Read */
    PAPI_L1_DCW_idx, /*L1 D Cache Write */
    PAPI_L2_DCW_idx, /*L2 D Cache Write */
    PAPI_L3_DCW_idx, /*L3 D Cache Write */
    PAPI_L1_ICH_idx, /*L1 instruction cache hits */
    PAPI_L2_ICH_idx, /*L2 instruction cache hits */
    PAPI_L3_ICH_idx, /*L3 instruction cache hits */
    PAPI_L1_ICA_idx, /*L1 instruction cache accesses */
    PAPI_L2_ICA_idx, /*L2 instruction cache accesses */
    PAPI_L3_ICA_idx, /*L3 instruction cache accesses */
    PAPI_L1_ICR_idx, /*L1 instruction cache reads */
    /* 0x50 */
    PAPI_L2_ICR_idx, /*L2 instruction cache reads */
    PAPI_L3_ICR_idx, /*L3 instruction cache reads */
    PAPI_L1_ICW_idx, /*L1 instruction cache writes */
    PAPI_L2_ICW_idx, /*L2 instruction cache writes */
    PAPI_L3_ICW_idx, /*L3 instruction cache writes */
    PAPI_L1_TCH_idx, /*L1 total cache hits */
    PAPI_L2_TCH_idx, /*L2 total cache hits */
    PAPI_L3_TCH_idx, /*L3 total cache hits */
    PAPI_L1_TCA_idx, /*L1 total cache accesses */
    PAPI_L2_TCA_idx, /*L2 total cache accesses */
    PAPI_L3_TCA_idx, /*L3 total cache accesses */
    PAPI_L1_TCR_idx, /*L1 total cache reads */
    PAPI_L2_TCR_idx, /*L2 total cache reads */
    PAPI_L3_TCR_idx, /*L3 total cache reads */
    PAPI_L1_TCW_idx, /*L1 total cache writes */
    PAPI_L2_TCW_idx, /*L2 total cache writes */
    /* 0x60 */
    PAPI_L3_TCW_idx,  /*L3 total cache writes */
    PAPI_FML_INS_idx, /*FM ins */
    PAPI_FAD_INS_idx, /*FA ins */
    PAPI_FDV_INS_idx, /*FD ins */
    PAPI_FSQ_INS_idx, /*FSq ins */
    PAPI_FNV_INS_idx, /*Finv ins */
    PAPI_FP_OPS_idx,  /*Floating point operations executed */
    PAPI_SP_OPS_idx,  /* Floating point operations executed; optimized to count scaled
                         single precision vector operations */
    PAPI_DP_OPS_idx,  /* Floating point operations executed; optimized to count scaled
                         double precision vector operations */
    PAPI_VEC_SP_idx,  /* Single precision vector/SIMD instructions */
    PAPI_VEC_DP_idx,  /* Double precision vector/SIMD instructions */
    PAPI_REF_CYC_idx, /* Reference clock cycles */
    PAPI_END_idx      /*This should always be last! */
};

#    define PAPI_NULL -1 /**<A nonexistent hardware event used as a placeholder */
#    define PAPI_L1_DCM                                                                  \
        (PAPI_L1_DCM_idx | PAPI_PRESET_MASK) /*Level 1 data cache misses */
#    define PAPI_L1_ICM                                                                  \
        (PAPI_L1_ICM_idx | PAPI_PRESET_MASK) /*Level 1 instruction cache misses */
#    define PAPI_L2_DCM                                                                  \
        (PAPI_L2_DCM_idx | PAPI_PRESET_MASK) /*Level 2 data cache misses */
#    define PAPI_L2_ICM                                                                  \
        (PAPI_L2_ICM_idx | PAPI_PRESET_MASK) /*Level 2 instruction cache misses */
#    define PAPI_L3_DCM                                                                  \
        (PAPI_L3_DCM_idx | PAPI_PRESET_MASK) /*Level 3 data cache misses */
#    define PAPI_L3_ICM                                                                  \
        (PAPI_L3_ICM_idx | PAPI_PRESET_MASK) /*Level 3 instruction cache misses */
#    define PAPI_L1_TCM                                                                  \
        (PAPI_L1_TCM_idx | PAPI_PRESET_MASK) /*Level 1 total cache misses */
#    define PAPI_L2_TCM                                                                  \
        (PAPI_L2_TCM_idx | PAPI_PRESET_MASK) /*Level 2 total cache misses */
#    define PAPI_L3_TCM                                                                  \
        (PAPI_L3_TCM_idx | PAPI_PRESET_MASK) /*Level 3 total cache misses */
#    define PAPI_CA_SNP (PAPI_CA_SNP_idx | PAPI_PRESET_MASK) /*Snoops */
#    define PAPI_CA_SHR                                                                  \
        (PAPI_CA_SHR_idx | PAPI_PRESET_MASK) /*Request for shared cache line (SMP) */
#    define PAPI_CA_CLN                                                                  \
        (PAPI_CA_CLN_idx | PAPI_PRESET_MASK) /*Request for clean cache line (SMP) */
#    define PAPI_CA_INV                                                                  \
        (PAPI_CA_INV_idx |                                                               \
         PAPI_PRESET_MASK) /*Request for cache line Invalidation (SMP) */
#    define PAPI_CA_ITV                                                                  \
        (PAPI_CA_ITV_idx |                                                               \
         PAPI_PRESET_MASK) /*Request for cache line Intervention (SMP) */
#    define PAPI_L3_LDM (PAPI_L3_LDM_idx | PAPI_PRESET_MASK) /*Level 3 load misses */
#    define PAPI_L3_STM (PAPI_L3_STM_idx | PAPI_PRESET_MASK) /*Level 3 store misses */
#    define PAPI_BRU_IDL                                                                 \
        (PAPI_BRU_IDL_idx | PAPI_PRESET_MASK) /*Cycles branch units are idle */
#    define PAPI_FXU_IDL                                                                 \
        (PAPI_FXU_IDL_idx | PAPI_PRESET_MASK) /*Cycles integer units are idle */
#    define PAPI_FPU_IDL                                                                 \
        (PAPI_FPU_IDL_idx | PAPI_PRESET_MASK) /*Cycles floating point units are idle */
#    define PAPI_LSU_IDL                                                                 \
        (PAPI_LSU_IDL_idx | PAPI_PRESET_MASK) /*Cycles load/store units are idle */
#    define PAPI_TLB_DM                                                                  \
        (PAPI_TLB_DM_idx | PAPI_PRESET_MASK) /*Data translation lookaside buffer misses  \
                                              */
#    define PAPI_TLB_IM                                                                  \
        (PAPI_TLB_IM_idx |                                                               \
         PAPI_PRESET_MASK) /*Instr translation lookaside buffer misses */
#    define PAPI_TLB_TL                                                                  \
        (PAPI_TLB_TL_idx |                                                               \
         PAPI_PRESET_MASK) /*Total translation lookaside buffer misses */
#    define PAPI_L1_LDM (PAPI_L1_LDM_idx | PAPI_PRESET_MASK) /*Level 1 load misses */
#    define PAPI_L1_STM (PAPI_L1_STM_idx | PAPI_PRESET_MASK) /*Level 1 store misses */
#    define PAPI_L2_LDM (PAPI_L2_LDM_idx | PAPI_PRESET_MASK) /*Level 2 load misses */
#    define PAPI_L2_STM (PAPI_L2_STM_idx | PAPI_PRESET_MASK) /*Level 2 store misses */
#    define PAPI_BTAC_M (PAPI_BTAC_M_idx | PAPI_PRESET_MASK) /*BTAC miss */
#    define PAPI_PRF_DM                                                                  \
        (PAPI_PRF_DM_idx | PAPI_PRESET_MASK) /*Prefetch data instruction caused a miss   \
                                              */
#    define PAPI_L3_DCH (PAPI_L3_DCH_idx | PAPI_PRESET_MASK) /*Level 3 Data Cache Hit */
#    define PAPI_TLB_SD                                                                  \
        (PAPI_TLB_SD_idx |                                                               \
         PAPI_PRESET_MASK) /*Xlation lookaside buffer shootdowns (SMP) */
#    define PAPI_CSR_FAL                                                                 \
        (PAPI_CSR_FAL_idx | PAPI_PRESET_MASK) /*Failed store conditional instructions */
#    define PAPI_CSR_SUC                                                                 \
        (PAPI_CSR_SUC_idx |                                                              \
         PAPI_PRESET_MASK) /*Successful store conditional instructions */
#    define PAPI_CSR_TOT                                                                 \
        (PAPI_CSR_TOT_idx | PAPI_PRESET_MASK) /*Total store conditional instructions */
#    define PAPI_MEM_SCY                                                                 \
        (PAPI_MEM_SCY_idx |                                                              \
         PAPI_PRESET_MASK) /*Cycles Stalled Waiting for Memory Access */
#    define PAPI_MEM_RCY                                                                 \
        (PAPI_MEM_RCY_idx | PAPI_PRESET_MASK) /*Cycles Stalled Waiting for Memory Read   \
                                               */
#    define PAPI_MEM_WCY                                                                 \
        (PAPI_MEM_WCY_idx | PAPI_PRESET_MASK) /*Cycles Stalled Waiting for Memory Write  \
                                               */
#    define PAPI_STL_ICY                                                                 \
        (PAPI_STL_ICY_idx | PAPI_PRESET_MASK) /*Cycles with No Instruction Issue */
#    define PAPI_FUL_ICY                                                                 \
        (PAPI_FUL_ICY_idx | PAPI_PRESET_MASK) /*Cycles with Maximum Instruction Issue */
#    define PAPI_STL_CCY                                                                 \
        (PAPI_STL_CCY_idx | PAPI_PRESET_MASK) /*Cycles with No Instruction Completion */
#    define PAPI_FUL_CCY                                                                 \
        (PAPI_FUL_CCY_idx |                                                              \
         PAPI_PRESET_MASK) /*Cycles with Maximum Instruction Completion */
#    define PAPI_HW_INT (PAPI_HW_INT_idx | PAPI_PRESET_MASK) /*Hardware interrupts */
#    define PAPI_BR_UCN                                                                  \
        (PAPI_BR_UCN_idx |                                                               \
         PAPI_PRESET_MASK) /*Unconditional branch instructions executed */
#    define PAPI_BR_CN                                                                   \
        (PAPI_BR_CN_idx | PAPI_PRESET_MASK) /*Conditional branch instructions executed   \
                                             */
#    define PAPI_BR_TKN                                                                  \
        (PAPI_BR_TKN_idx | PAPI_PRESET_MASK) /*Conditional branch instructions taken */
#    define PAPI_BR_NTK                                                                  \
        (PAPI_BR_NTK_idx |                                                               \
         PAPI_PRESET_MASK) /*Conditional branch instructions not taken */
#    define PAPI_BR_MSP                                                                  \
        (PAPI_BR_MSP_idx | PAPI_PRESET_MASK) /*Conditional branch instructions mispred   \
                                              */
#    define PAPI_BR_PRC                                                                  \
        (PAPI_BR_PRC_idx |                                                               \
         PAPI_PRESET_MASK) /*Conditional branch instructions corr. pred */
#    define PAPI_FMA_INS                                                                 \
        (PAPI_FMA_INS_idx | PAPI_PRESET_MASK) /*FMA instructions completed */
#    define PAPI_TOT_IIS                                                                 \
        (PAPI_TOT_IIS_idx | PAPI_PRESET_MASK) /*Total instructions issued */
#    define PAPI_TOT_INS                                                                 \
        (PAPI_TOT_INS_idx | PAPI_PRESET_MASK) /*Total instructions executed */
#    define PAPI_INT_INS                                                                 \
        (PAPI_INT_INS_idx | PAPI_PRESET_MASK) /*Integer instructions executed */
#    define PAPI_FP_INS                                                                  \
        (PAPI_FP_INS_idx | PAPI_PRESET_MASK) /*Floating point instructions executed */
#    define PAPI_LD_INS                                                                  \
        (PAPI_LD_INS_idx | PAPI_PRESET_MASK) /*Load instructions executed */
#    define PAPI_SR_INS                                                                  \
        (PAPI_SR_INS_idx | PAPI_PRESET_MASK) /*Store instructions executed */
#    define PAPI_BR_INS                                                                  \
        (PAPI_BR_INS_idx | PAPI_PRESET_MASK) /*Total branch instructions executed */
#    define PAPI_VEC_INS (PAPI_VEC_INS_idx | PAPI_PRESET_MASK)
/*Vector/SIMD instructions executed (could include integer) */
#    define PAPI_RES_STL                                                                 \
        (PAPI_RES_STL_idx | PAPI_PRESET_MASK) /*Cycles processor is stalled on resource  \
                                               */
#    define PAPI_FP_STAL                                                                 \
        (PAPI_FP_STAL_idx | PAPI_PRESET_MASK) /*Cycles any FP units are stalled */
#    define PAPI_TOT_CYC                                                                 \
        (PAPI_TOT_CYC_idx | PAPI_PRESET_MASK) /*Total cycles executed                    \
                                               */
#    define PAPI_LST_INS                                                                 \
        (PAPI_LST_INS_idx | PAPI_PRESET_MASK) /*Total load/store inst. executed */
#    define PAPI_SYC_INS (PAPI_SYC_INS_idx | PAPI_PRESET_MASK) /*Sync. inst. executed */
#    define PAPI_L1_DCH (PAPI_L1_DCH_idx | PAPI_PRESET_MASK)   /*L1 D Cache Hit */
#    define PAPI_L2_DCH (PAPI_L2_DCH_idx | PAPI_PRESET_MASK)   /*L2 D Cache Hit */
#    define PAPI_L1_DCA (PAPI_L1_DCA_idx | PAPI_PRESET_MASK)   /*L1 D Cache Access */
#    define PAPI_L2_DCA (PAPI_L2_DCA_idx | PAPI_PRESET_MASK)   /*L2 D Cache Access */
#    define PAPI_L3_DCA (PAPI_L3_DCA_idx | PAPI_PRESET_MASK)   /*L3 D Cache Access */
#    define PAPI_L1_DCR (PAPI_L1_DCR_idx | PAPI_PRESET_MASK)   /*L1 D Cache Read */
#    define PAPI_L2_DCR (PAPI_L2_DCR_idx | PAPI_PRESET_MASK)   /*L2 D Cache Read */
#    define PAPI_L3_DCR (PAPI_L3_DCR_idx | PAPI_PRESET_MASK)   /*L3 D Cache Read */
#    define PAPI_L1_DCW (PAPI_L1_DCW_idx | PAPI_PRESET_MASK)   /*L1 D Cache Write */
#    define PAPI_L2_DCW (PAPI_L2_DCW_idx | PAPI_PRESET_MASK)   /*L2 D Cache Write */
#    define PAPI_L3_DCW (PAPI_L3_DCW_idx | PAPI_PRESET_MASK)   /*L3 D Cache Write */
#    define PAPI_L1_ICH                                                                  \
        (PAPI_L1_ICH_idx | PAPI_PRESET_MASK) /*L1 instruction cache hits */
#    define PAPI_L2_ICH                                                                  \
        (PAPI_L2_ICH_idx | PAPI_PRESET_MASK) /*L2 instruction cache hits */
#    define PAPI_L3_ICH                                                                  \
        (PAPI_L3_ICH_idx | PAPI_PRESET_MASK) /*L3 instruction cache hits */
#    define PAPI_L1_ICA                                                                  \
        (PAPI_L1_ICA_idx | PAPI_PRESET_MASK) /*L1 instruction cache accesses */
#    define PAPI_L2_ICA                                                                  \
        (PAPI_L2_ICA_idx | PAPI_PRESET_MASK) /*L2 instruction cache accesses */
#    define PAPI_L3_ICA                                                                  \
        (PAPI_L3_ICA_idx | PAPI_PRESET_MASK) /*L3 instruction cache accesses */
#    define PAPI_L1_ICR                                                                  \
        (PAPI_L1_ICR_idx | PAPI_PRESET_MASK) /*L1 instruction cache reads */
#    define PAPI_L2_ICR                                                                  \
        (PAPI_L2_ICR_idx | PAPI_PRESET_MASK) /*L2 instruction cache reads */
#    define PAPI_L3_ICR                                                                  \
        (PAPI_L3_ICR_idx | PAPI_PRESET_MASK) /*L3 instruction cache reads */
#    define PAPI_L1_ICW                                                                  \
        (PAPI_L1_ICW_idx | PAPI_PRESET_MASK) /*L1 instruction cache writes */
#    define PAPI_L2_ICW                                                                  \
        (PAPI_L2_ICW_idx | PAPI_PRESET_MASK) /*L2 instruction cache writes */
#    define PAPI_L3_ICW                                                                  \
        (PAPI_L3_ICW_idx | PAPI_PRESET_MASK) /*L3 instruction cache writes */
#    define PAPI_L1_TCH (PAPI_L1_TCH_idx | PAPI_PRESET_MASK) /*L1 total cache hits */
#    define PAPI_L2_TCH (PAPI_L2_TCH_idx | PAPI_PRESET_MASK) /*L2 total cache hits */
#    define PAPI_L3_TCH (PAPI_L3_TCH_idx | PAPI_PRESET_MASK) /*L3 total cache hits */
#    define PAPI_L1_TCA                                                                  \
        (PAPI_L1_TCA_idx | PAPI_PRESET_MASK) /*L1 total cache accesses                   \
                                              */
#    define PAPI_L2_TCA                                                                  \
        (PAPI_L2_TCA_idx | PAPI_PRESET_MASK) /*L2 total cache accesses                   \
                                              */
#    define PAPI_L3_TCA                                                                  \
        (PAPI_L3_TCA_idx | PAPI_PRESET_MASK)                   /*L3 total cache accesses \
                                                                */
#    define PAPI_L1_TCR (PAPI_L1_TCR_idx | PAPI_PRESET_MASK)   /*L1 total cache reads */
#    define PAPI_L2_TCR (PAPI_L2_TCR_idx | PAPI_PRESET_MASK)   /*L2 total cache reads */
#    define PAPI_L3_TCR (PAPI_L3_TCR_idx | PAPI_PRESET_MASK)   /*L3 total cache reads */
#    define PAPI_L1_TCW (PAPI_L1_TCW_idx | PAPI_PRESET_MASK)   /*L1 total cache writes */
#    define PAPI_L2_TCW (PAPI_L2_TCW_idx | PAPI_PRESET_MASK)   /*L2 total cache writes */
#    define PAPI_L3_TCW (PAPI_L3_TCW_idx | PAPI_PRESET_MASK)   /*L3 total cache writes */
#    define PAPI_FML_INS (PAPI_FML_INS_idx | PAPI_PRESET_MASK) /*FM ins */
#    define PAPI_FAD_INS (PAPI_FAD_INS_idx | PAPI_PRESET_MASK) /*FA ins */
#    define PAPI_FDV_INS (PAPI_FDV_INS_idx | PAPI_PRESET_MASK) /*FD ins */
#    define PAPI_FSQ_INS (PAPI_FSQ_INS_idx | PAPI_PRESET_MASK) /*FSq ins */
#    define PAPI_FNV_INS (PAPI_FNV_INS_idx | PAPI_PRESET_MASK) /*Finv ins */
#    define PAPI_FP_OPS                                                                  \
        (PAPI_FP_OPS_idx | PAPI_PRESET_MASK) /*Floating point operations executed */
#    define PAPI_SP_OPS                                                                  \
        (PAPI_SP_OPS_idx |                                                               \
         PAPI_PRESET_MASK) /* Floating point operations executed; optimized to count */
                           /* scaled single precision vector operations */
#    define PAPI_DP_OPS                                                                  \
        (PAPI_DP_OPS_idx |                                                               \
         PAPI_PRESET_MASK) /* Floating point operations executed; optimized to count */
                           /* scaled double precision vector operations */
#    define PAPI_VEC_SP                                                                  \
        (PAPI_VEC_SP_idx |                                                               \
         PAPI_PRESET_MASK) /* Single precision vector/SIMD instructions */
#    define PAPI_VEC_DP                                                                  \
        (PAPI_VEC_DP_idx |                                                               \
         PAPI_PRESET_MASK) /* Double precision vector/SIMD instructions */
#    define PAPI_REF_CYC                                                                 \
        (PAPI_REF_CYC_idx | PAPI_PRESET_MASK) /* Reference clock cycles */

#    define PAPI_END (PAPI_END_idx | PAPI_PRESET_MASK) /*This should always be last! */

#    define PAPI_DETACH 1 /* Detach */
#    define PAPI_DEBUG 2  /* Option to turn on  debugging features of the PAPI library */
#    define PAPI_MULTIPLEX 3 /* Turn on/off or multiplexing for an eventset */
#    define PAPI_DEFDOM                                                                  \
        4 /* Domain for all new eventsets. Takes non-NULL option pointer. */
#    define PAPI_DOMAIN 5 /* Domain for an eventset */
#    define PAPI_DEFGRN 6 /* Granularity for all new eventsets */
#    define PAPI_GRANUL 7 /* Granularity for an eventset */
#    define PAPI_DEF_MPX_NS                                                              \
        8 /* Multiplexing/overflowing interval in ns, same as PAPI_DEF_ITIMER_NS */
//#define PAPI_EDGE_DETECT    9       /* Count cycles of events if supported [not
// implemented] */ #define PAPI_INVERT         10              /* Invert count detect if
// supported [not implemented] */
#    define PAPI_MAX_MPX_CTRS 11 /* Maximum number of counters we can multiplex */
#    define PAPI_PROFIL                                                                  \
        12 /* Option to turn on the overflow/profil reporting software [not impled]*/
#    define PAPI_PRELOAD                                                                 \
        13 /* Option to find out the environment variable that can preload libraries */
#    define PAPI_CLOCKRATE 14  /* Clock rate in MHz */
#    define PAPI_MAX_HWCTRS 15 /* Number of physical hardware counters */
#    define PAPI_HWINFO 16     /* Hardware information */
#    define PAPI_EXEINFO 17    /* Executable information */
#    define PAPI_MAX_CPUS 18   /* Number of ncpus we can talk to from here */
#    define PAPI_ATTACH 19     /* Attach to a another tid/pid instead of ourself */
#    define PAPI_SHLIBINFO 20  /* Shared Library information */
#    define PAPI_LIB_VERSION                                                             \
        21 /* Option to find out the complete version number of the PAPI library */
#    define PAPI_COMPONENTINFO 22 /* Find out what the component supports */
/* Currently the following options are only available on Itanium; they may be supported
 * elsewhere in the future */
#    define PAPI_DATA_ADDRESS 23 /* Option to set data address range restriction */
#    define PAPI_INSTR_ADDRESS                                                           \
        24 /* Option to set instruction address range restriction */
#    define PAPI_DEF_ITIMER                                                              \
        25 /* Option to set the type of itimer used in both software multiplexing, */
           /* overflowing and profiling */
#    define PAPI_DEF_ITIMER_NS                                                           \
        26 /* Multiplexing/overflowing interval in ns, same as PAPI_DEF_MPX_NS */
/* Currently the following options are only available on systems using the perf_events
 * component within papi */
#    define PAPI_CPU_ATTACH                                                              \
        27                  /* Specify a cpu number the event set should be tied to      \
                             */
#    define PAPI_INHERIT 28 /* Option to set counter inheritance flag */
#    define PAPI_USER_EVENTS_FILE                                                        \
        29 /* Option to set file from where to parse user defined events */

#    define PAPI_INIT_SLOTS                                                              \
        64 /*Number of initialized slots in \ DynamicArray of EventSets */

#    define PAPI_MIN_STR_LEN 64   /* For small strings, like names & stuff */
#    define PAPI_MAX_STR_LEN 128  /* For average run-of-the-mill strings */
#    define PAPI_2MAX_STR_LEN 256 /* For somewhat longer run-of-the-mill strings */
#    define PAPI_HUGE_STR_LEN                                                            \
        1024 /* This should be defined in terms of a system parameter */

#    define PAPI_PMU_MAX 40  /* maximum number of pmu's supported by one component */
#    define PAPI_DERIVED 0x1 /* Flag to indicate that the event is derived */

#    define PAPI_MAX_INFO_TERMS                                                          \
        12 /* should match PAPI_EVENTS_IN_DERIVED_EVENT defined in papi_internal.h */

typedef struct __PAPI_event_info
{
    unsigned int event_code = 0;  /* preset (0x8xxxxxxx) or
                                 native (0x4xxxxxxx) event code */
    std::string symbol      = ""; /* name of the event */
    std::string short_descr = ""; /* a short description suitable for
                                          use as a label */
    std::string long_descr = "";  /* a longer description:
                                      typically a sentence for presets,
                                      possibly a paragraph from vendor
                                      docs for native events */

    int         component_index = 0;  /* component this event belongs to */
    std::string units           = ""; /* units event is measured in */
    int         location        = 0;  /* location event applies to */
    int         data_type       = 0;  /* data type returned by PAPI */
    int         value_type      = 0;  /* sum or absolute */
    int         timescope       = 0;  /* from start, etc. */
    int         update_type     = 0;  /* how event is updated */
    int         update_freq     = 0;  /* how frequently event is updated */

    /* PRESET SPECIFIC FIELDS FOLLOW */

    unsigned int count = 0; /* number of terms (usually 1)
                           in the code and name fields
                           - presets: these are native events
                           - native: these are unused */

    unsigned int event_type = 0; /* event type or category
                                for preset events only */

    std::string derived = ""; /* name of the derived type
                                       - presets: usually NOT_DERIVED
                                       - native: empty string */
    std::string postfix = ""; /* string containing postfix
                                        operations; only defined for
                                        preset events of derived type
                                        DERIVED_POSTFIX */

    unsigned int code[PAPI_MAX_INFO_TERMS]; /* array of values that further
                                          describe the event:
                                          - presets: native event_code values
                                          - native:, register values(?) */

    std::string name         /* names of code terms: */
        [PAPI_2MAX_STR_LEN]; /* - presets: native event names,
                                - native: descriptive strings
                                for each register value(?) */

    std::string note = ""; /* an optional developer note
                                   supplied with a preset event
                                   to delineate platform specific
                                   anomalies or restrictions */

} PAPI_event_info_t;

#    if 0
/* These are used both by PAPI and by the genpapifdef utility */
/* They are in their own include to allow genpapifdef to be built */
/* without having to link against libpapi.a                       */

hwi_presets_t _papi_hwi_presets[PAPI_MAX_PRESET_EVENTS] = {
/*  0 */ {"PAPI_L1_DCM",
          "L1D cache misses",
          "Level 1 data cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/*  1 */ {"PAPI_L1_ICM",
          "L1I cache misses",
          "Level 1 instruction cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/*  2 */ {"PAPI_L2_DCM",
          "L2D cache misses",
          "Level 2 data cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/*  3 */ {"PAPI_L2_ICM",
          "L2I cache misses",
          "Level 2 instruction cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/*  4 */ {"PAPI_L3_DCM",
          "L3D cache misses",
          "Level 3 data cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/*  5 */ {"PAPI_L3_ICM",
          "L3I cache misses",
          "Level 3 instruction cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/*  6 */ {"PAPI_L1_TCM",
          "L1 cache misses",
          "Level 1 cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/*  7 */ {"PAPI_L2_TCM",
          "L2 cache misses",
          "Level 2 cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/*  8 */ {"PAPI_L3_TCM",
          "L3 cache misses",
          "Level 3 cache misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/*  9 */ {"PAPI_CA_SNP",
          "Snoop Requests",
          "Requests for a snoop", 0,
          0, PAPI_PRESET_BIT_CACH,
          NULL, {0},{NULL}, NULL},
/* 10 */ {"PAPI_CA_SHR",
          "Ex Acces shared CL",
          "Requests for exclusive access to shared cache line", 0,
          0, PAPI_PRESET_BIT_CACH,
          NULL, {0},{NULL}, NULL},
/* 11 */ {"PAPI_CA_CLN",
          "Ex Access clean CL",
          "Requests for exclusive access to clean cache line", 0,
          0, PAPI_PRESET_BIT_CACH,
          NULL, {0},{NULL}, NULL},
/* 12 */ {"PAPI_CA_INV",
          "Cache ln invalid",
          "Requests for cache line invalidation", 0,
          0, PAPI_PRESET_BIT_CACH,
          NULL, {0},{NULL}, NULL},
/* 13 */ {"PAPI_CA_ITV",
          "Cache ln intervene",
          "Requests for cache line intervention", 0,
          0, PAPI_PRESET_BIT_CACH,
          NULL, {0},{NULL}, NULL},
/* 14 */ {"PAPI_L3_LDM",
          "L3 load misses",
          "Level 3 load misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 15 */ {"PAPI_L3_STM",
          "L3 store misses",
          "Level 3 store misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 16 */ {"PAPI_BRU_IDL",
          "Branch idle cycles",
          "Cycles branch units are idle", 0,
          0, PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_BR,
          NULL, {0},{NULL}, NULL},
/* 17 */ {"PAPI_FXU_IDL",
          "IU idle cycles",
          "Cycles integer units are idle", 0,
          0, PAPI_PRESET_BIT_IDL,
          NULL, {0},{NULL}, NULL},
/* 18 */ {"PAPI_FPU_IDL",
          "FPU idle cycles",
          "Cycles floating point units are idle", 0,
          0, PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 19 */ {"PAPI_LSU_IDL",
          "L/SU idle cycles",
          "Cycles load/store units are idle", 0,
          0, PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 20 */ {"PAPI_TLB_DM",
          "Data TLB misses",
          "Data translation lookaside buffer misses", 0,
          0, PAPI_PRESET_BIT_TLB,
          NULL, {0},{NULL}, NULL},
/* 21 */ {"PAPI_TLB_IM",
          "Instr TLB misses",
          "Instruction translation lookaside buffer misses", 0,
          0, PAPI_PRESET_BIT_TLB + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 22 */ {"PAPI_TLB_TL",
          "Total TLB misses",
          "Total translation lookaside buffer misses", 0,
          0, PAPI_PRESET_BIT_TLB,
          NULL, {0},{NULL}, NULL},
/* 23 */ {"PAPI_L1_LDM",
          "L1 load misses",
          "Level 1 load misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 24 */ {"PAPI_L1_STM",
          "L1 store misses",
          "Level 1 store misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 25 */ {"PAPI_L2_LDM",
          "L2 load misses",
          "Level 2 load misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 26 */ {"PAPI_L2_STM",
          "L2 store misses",
          "Level 2 store misses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 27 */ {"PAPI_BTAC_M",
          "Br targt addr miss",
          "Branch target address cache misses", 0,
          0, PAPI_PRESET_BIT_BR,
          NULL, {0},{NULL}, NULL},
/* 28 */ {"PAPI_PRF_DM",
          "Data prefetch miss",
          "Data prefetch cache misses", 0,
          0, PAPI_PRESET_BIT_CACH,
          NULL, {0},{NULL}, NULL},
/* 29 */ {"PAPI_L3_DCH",
          "L3D cache hits",
          "Level 3 data cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 30 */ {"PAPI_TLB_SD",
          "TLB shootdowns",
          "Translation lookaside buffer shootdowns", 0,
          0, PAPI_PRESET_BIT_TLB,
          NULL, {0},{NULL}, NULL},
/* 31 */ {"PAPI_CSR_FAL",
          "Failed store cond",
          "Failed store conditional instructions", 0,
          0, PAPI_PRESET_BIT_CND + PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 32 */ {"PAPI_CSR_SUC",
          "Good store cond",
          "Successful store conditional instructions", 0,
          0, PAPI_PRESET_BIT_CND + PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 33 */ {"PAPI_CSR_TOT",
          "Total store cond",
          "Total store conditional instructions", 0,
          0, PAPI_PRESET_BIT_CND + PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 34 */ {"PAPI_MEM_SCY",
          "Stalled mem cycles",
          "Cycles Stalled Waiting for memory accesses", 0,
          0, PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 35 */ {"PAPI_MEM_RCY",
          "Stalled rd cycles",
          "Cycles Stalled Waiting for memory Reads", 0,
          0, PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 36 */ {"PAPI_MEM_WCY",
          "Stalled wr cycles",
          "Cycles Stalled Waiting for memory writes", 0,
          0, PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 37 */ {"PAPI_STL_ICY",
          "No instr issue",
          "Cycles with no instruction issue", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 38 */ {"PAPI_FUL_ICY",
          "Max instr issue",
          "Cycles with maximum instruction issue", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 39 */ {"PAPI_STL_CCY",
          "No instr done",
          "Cycles with no instructions completed", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 40 */ {"PAPI_FUL_CCY",
          "Max instr done",
          "Cycles with maximum instructions completed", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 41 */ {"PAPI_HW_INT",
          "Hdw interrupts",
          "Hardware interrupts", 0,
          0, PAPI_PRESET_BIT_MSC,
          NULL, {0},{NULL}, NULL},
/* 42 */ {"PAPI_BR_UCN",
          "Uncond branch",
          "Unconditional branch instructions", 0,
          0, PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
          NULL, {0},{NULL}, NULL},
/* 43 */ {"PAPI_BR_CN",
          "Cond branch",
          "Conditional branch instructions", 0,
          0, PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
          NULL, {0},{NULL}, NULL},
/* 44 */ {"PAPI_BR_TKN",
          "Cond branch taken",
          "Conditional branch instructions taken", 0,
          0, PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
          NULL, {0},{NULL}, NULL},
/* 45 */ {"PAPI_BR_NTK",
          "Cond br not taken",
          "Conditional branch instructions not taken", 0,
          0, PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
          NULL, {0},{NULL}, NULL},
/* 46 */ {"PAPI_BR_MSP",
          "Cond br mspredictd",
          "Conditional branch instructions mispredicted", 0,
          0, PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
          NULL, {0},{NULL}, NULL},
/* 47 */ {"PAPI_BR_PRC",
          "Cond br predicted",
          "Conditional branch instructions correctly predicted", 0,
          0, PAPI_PRESET_BIT_BR + PAPI_PRESET_BIT_CND,
          NULL, {0},{NULL}, NULL},
/* 48 */ {"PAPI_FMA_INS",
          "FMAs completed",
          "FMA instructions completed", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 49 */ {"PAPI_TOT_IIS",
          "Instr issued",
          "Instructions issued", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 50 */ {"PAPI_TOT_INS",
          "Instr completed",
          "Instructions completed", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 51 */ {"PAPI_INT_INS",
          "Int instructions",
          "Integer instructions", 0,
          0, PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 52 */ {"PAPI_FP_INS",
          "FP instructions",
          "Floating point instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 53 */ {"PAPI_LD_INS",
          "Loads",
          "Load instructions", 0,
          0, PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 54 */ {"PAPI_SR_INS",
          "Stores",
          "Store instructions", 0,
          0, PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 55 */ {"PAPI_BR_INS",
          "Branches",
          "Branch instructions", 0,
          0, PAPI_PRESET_BIT_BR,
          NULL, {0},{NULL}, NULL},
/* 56 */ {"PAPI_VEC_INS",
          "Vector/SIMD instr",
          "Vector/SIMD instructions (could include integer)", 0,
          0, PAPI_PRESET_BIT_MSC,
          NULL, {0},{NULL}, NULL},
/* 57 */ {"PAPI_RES_STL",
          "Stalled res cycles",
          "Cycles stalled on any resource", 0,
          0, PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_MSC,
          NULL, {0},{NULL}, NULL},
/* 58 */ {"PAPI_FP_STAL",
          "Stalled FPU cycles",
          "Cycles the FP unit(s) are stalled", 0,
          0, PAPI_PRESET_BIT_IDL + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 59 */ {"PAPI_TOT_CYC",
          "Total cycles",
          "Total cycles", 0,
          0, PAPI_PRESET_BIT_MSC,
          NULL, {0},{NULL}, NULL},
/* 60 */ {"PAPI_LST_INS",
          "L/S completed",
          "Load/store instructions completed", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_MEM,
          NULL, {0},{NULL}, NULL},
/* 61 */ {"PAPI_SYC_INS",
          "Syncs completed",
          "Synchronization instructions completed", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_MSC,
          NULL, {0},{NULL}, NULL},
/* 62 */ {"PAPI_L1_DCH",
          "L1D cache hits",
          "Level 1 data cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 63 */ {"PAPI_L2_DCH",
          "L2D cache hits",
          "Level 2 data cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 64 */ {"PAPI_L1_DCA",
          "L1D cache accesses",
          "Level 1 data cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 65 */ {"PAPI_L2_DCA",
          "L2D cache accesses",
          "Level 2 data cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 66 */ {"PAPI_L3_DCA",
          "L3D cache accesses",
          "Level 3 data cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 67 */ {"PAPI_L1_DCR",
          "L1D cache reads",
          "Level 1 data cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 68 */ {"PAPI_L2_DCR",
          "L2D cache reads",
          "Level 2 data cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 69 */ {"PAPI_L3_DCR",
          "L3D cache reads",
          "Level 3 data cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 70 */ {"PAPI_L1_DCW",
          "L1D cache writes",
          "Level 1 data cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 71 */ {"PAPI_L2_DCW",
          "L2D cache writes",
          "Level 2 data cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 72 */ {"PAPI_L3_DCW",
          "L3D cache writes",
          "Level 3 data cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 73 */ {"PAPI_L1_ICH",
          "L1I cache hits",
          "Level 1 instruction cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 74 */ {"PAPI_L2_ICH",
          "L2I cache hits",
          "Level 2 instruction cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 75 */ {"PAPI_L3_ICH",
          "L3I cache hits",
          "Level 3 instruction cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 76 */ {"PAPI_L1_ICA",
          "L1I cache accesses",
          "Level 1 instruction cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 77 */ {"PAPI_L2_ICA",
          "L2I cache accesses",
          "Level 2 instruction cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 78 */ {"PAPI_L3_ICA",
          "L3I cache accesses",
          "Level 3 instruction cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 79 */ {"PAPI_L1_ICR",
          "L1I cache reads",
          "Level 1 instruction cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 80 */ {"PAPI_L2_ICR",
          "L2I cache reads",
          "Level 2 instruction cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 81 */ {"PAPI_L3_ICR",
          "L3I cache reads",
          "Level 3 instruction cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 82 */ {"PAPI_L1_ICW",
          "L1I cache writes",
          "Level 1 instruction cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 83 */ {"PAPI_L2_ICW",
          "L2I cache writes",
          "Level 2 instruction cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 84 */ {"PAPI_L3_ICW",
          "L3I cache writes",
          "Level 3 instruction cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3 + PAPI_PRESET_BIT_INS,
          NULL, {0},{NULL}, NULL},
/* 85 */ {"PAPI_L1_TCH",
          "L1 cache hits",
          "Level 1 total cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 86 */ {"PAPI_L2_TCH",
          "L2 cache hits",
          "Level 2 total cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 87 */ {"PAPI_L3_TCH",
          "L3 cache hits",
          "Level 3 total cache hits", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 88 */ {"PAPI_L1_TCA",
          "L1 cache accesses",
          "Level 1 total cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 89 */ {"PAPI_L2_TCA",
          "L2 cache accesses",
          "Level 2 total cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 90 */ {"PAPI_L3_TCA",
          "L3 cache accesses",
          "Level 3 total cache accesses", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 91 */ {"PAPI_L1_TCR",
          "L1 cache reads",
          "Level 1 total cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 92 */ {"PAPI_L2_TCR",
          "L2 cache reads",
          "Level 2 total cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 93 */ {"PAPI_L3_TCR",
          "L3 cache reads",
          "Level 3 total cache reads", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 94 */ {"PAPI_L1_TCW",
          "L1 cache writes",
          "Level 1 total cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L1,
          NULL, {0},{NULL}, NULL},
/* 95 */ {"PAPI_L2_TCW",
          "L2 cache writes",
          "Level 2 total cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L2,
          NULL, {0},{NULL}, NULL},
/* 96 */ {"PAPI_L3_TCW",
          "L3 cache writes",
          "Level 3 total cache writes", 0,
          0, PAPI_PRESET_BIT_CACH + PAPI_PRESET_BIT_L3,
          NULL, {0},{NULL}, NULL},
/* 97 */ {"PAPI_FML_INS",
          "FPU multiply",
          "Floating point multiply instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 98 */ {"PAPI_FAD_INS",
          "FPU add",
          "Floating point add instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 99 */ {"PAPI_FDV_INS",
          "FPU divide",
          "Floating point divide instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*100 */ {"PAPI_FSQ_INS",
          "FPU square root",
          "Floating point square root instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*101 */ {"PAPI_FNV_INS",
          "FPU inverse",
          "Floating point inverse instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*102 */ {"PAPI_FP_OPS",
          "FP operations",
          "Floating point operations", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*103 */ {"PAPI_SP_OPS",
          "SP operations",
          "Floating point operations; optimized to count scaled single precision vector operations", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*104 */ {"PAPI_DP_OPS",
          "DP operations",
          "Floating point operations; optimized to count scaled double precision vector operations", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*105 */ {"PAPI_VEC_SP",
          "SP Vector/SIMD instr",
          "Single precision vector/SIMD instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/*106 */ {"PAPI_VEC_DP",
          "DP Vector/SIMD instr",
          "Double precision vector/SIMD instructions", 0,
          0, PAPI_PRESET_BIT_INS + PAPI_PRESET_BIT_FP,
          NULL, {0},{NULL}, NULL},
/* 107 */ {"PAPI_REF_CYC",
          "Reference cycles",
          "Reference clock cycles", 0,
          0, PAPI_PRESET_BIT_MSC,
          NULL, {0},{NULL}, NULL},
/*108 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*109 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*110 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*111 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*112 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*113 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*114 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*115 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*116 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*117 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*118 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*119 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*120 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*121 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*122 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*123 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*124 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*125 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*126 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
/*127 */ {NULL, NULL, NULL,0,0,0,NULL,{0},{NULL}, NULL},
};

const hwi_describe_t _papi_hwi_err[PAPI_NUM_ERRORS] = {
    /* 0 */ {PAPI_OK, "PAPI_OK", "No error"},
    /* 1 */ {PAPI_EINVAL, "PAPI_EINVAL", "Invalid argument"},
    /* 2 */ {PAPI_ENOMEM, "PAPI_ENOMEM", "Insufficient memory"},
    /* 3 */ {PAPI_ESYS, "PAPI_ESYS", "A System/C library call failed"},
    /* 4 */ {PAPI_ECMP, "PAPI_ECMP", "Not supported by component"},
    /* 5 */ {PAPI_ECLOST, "PAPI_ECLOST", "Access to the counters was lost or interrupted"},
    /* 6 */ {PAPI_EBUG, "PAPI_EBUG", "Internal error, please send mail to the developers"},
    /* 7 */ {PAPI_ENOEVNT, "PAPI_ENOEVNT", "Event does not exist"},
    /* 8 */ {PAPI_ECNFLCT, "PAPI_ECNFLCT", "Event exists, but cannot be counted due to hardware resource limits"},
    /* 9 */ {PAPI_ENOTRUN, "PAPI_ENOTRUN", "EventSet is currently not running"},
    /*10 */ {PAPI_EISRUN, "PAPI_EISRUN", "EventSet is currently counting"},
    /*11 */ {PAPI_ENOEVST, "PAPI_ENOEVST", "No such EventSet available"},
    /*12 */ {PAPI_ENOTPRESET, "PAPI_ENOTPRESET", "Event in argument is not a valid preset"},
    /*13 */ {PAPI_ENOCNTR, "PAPI_ENOCNTR", "Hardware does not support performance counters"},
    /*14 */ {PAPI_EMISC, "PAPI_EMISC", "Unknown error code"},
    /*15 */ {PAPI_EPERM, "PAPI_EPERM", "Permission level does not permit operation"},
    /*16 */ {PAPI_ENOINIT, "PAPI_ENOINIT", "PAPI hasn't been initialized yet"},
    /*17 */ {PAPI_ENOCMP, "PAPI_ENOCMP", "Component Index isn't set"},
    /*18 */ {PAPI_ENOSUPP, "PAPI_ENOSUPP", "Not supported"},
    /*19 */ {PAPI_ENOIMPL, "PAPI_ENOIMPL", "Not implemented"},
    /*20 */ {PAPI_EBUF, "PAPI_EBUF", "Buffer size exceeded"},
    /*21 */ {PAPI_EINVAL_DOM, "PAPI_EINVAL_DOM", "EventSet domain is not supported for the operation"},
    /*22 */ {PAPI_EATTR, "PAPI_EATTR", "Invalid or missing event attributes"},
    /*23 */ {PAPI_ECOUNT, "PAPI_ECOUNT", "Too many events or attributes"},
    /*24 */ {PAPI_ECOMBO, "PAPI_ECOMBO", "Bad combination of features"}
    /*25 */ {PAPI_ECMP_DISABLED, "PAPI_ECMP_DISABLED", "Component containing event is disabled"}
};
#    endif

#endif  // !defined(TIMEMORY_USE_PAPI)
