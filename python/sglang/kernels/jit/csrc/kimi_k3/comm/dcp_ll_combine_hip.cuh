// DCP8 MLA decode merge on gfx950 as one kernel (ROCm, wave64).
//
// Every rank holds the attention partials of all 96 heads over its own KV shard
// ([B, 96, 128] bf16 after the V projection, with the per-head fp32 LSE) and owns 12 heads of
// the result. One workgroup per batch row, two threads per 16-byte vector of a local head
// slot: one pushes that vector of each peer's heads into the peer's scratch, the other
// combines the 8 partials of this rank's head by LSE weight,
//     out = bf16(sum_i partial_i * exp(lse_i - max_i lse) / max(sum_i exp(lse_i - max), 1))
// with a NaN or +inf LSE (a shard with no key) read as -inf: dcp_pack_a2a_send + all_to_all +
// _dcp_lse_combine_kernel (ops/attention/dcp_kernels.py), bit for bit.
//
// The exchange has no barrier: the receiver polls its own scratch. An LSE travels as an LL
// line (ll_hip.cuh: 8 payload bytes + the epoch flag). A payload vector travels as one
// 16-byte line that reads all-zero until it lands and that its reader zeroes again, so the
// exchange moves no more bytes than the all-to-all.
//
// Scratch (receiver-owned, a region of communication/k3_ll_comm_hip's):
// [2 epochs][kWorld src ranks][kMaxRows rows][kLinesPerRow lines]; a row's lines are its 192
// vectors, then its 12 LSEs. Row b is always handled by workgroup b, whose flag
// epochs[b] + 1 is rank-local; every rank launches the same row counts in the same order, so
// the flags agree. A line is rewritten two epochs later, by which point its receiver has
// consumed and zeroed it (the writer has since received the receiver's next-epoch row).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "../attn_res/common_hip.cuh"
#include "ll_hip.cuh"
#include <cmath>
#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, Signal

#ifndef USE_ROCM
#error "kimi_k3/comm/dcp_ll_combine_hip.cuh is a wave64 (ROCm) kernel"
#endif

namespace sglang {
namespace k3_dcp_ll_combine {

using namespace k3_ll;
using k3_attn_res_read::kVec;
using k3_attn_res_read::pack8;
using k3_attn_res_read::unpack8;

inline constexpr int kWorld = 8;
// Must match communication/k3_ll_comm_hip.py: MAX_ROWS and the region's lines per row.
inline constexpr int kMaxRows = 384;
inline constexpr int kHeads = 12;  // heads a rank owns
inline constexpr int kHeadDim = 128;
inline constexpr int kHeadVecs = kHeadDim / kVec;
inline constexpr int kThreads = kHeads * kHeadVecs;
inline constexpr int kLseLine = kThreads;
inline constexpr int kLinesPerRow = kLseLine + kHeads;
inline constexpr int64_t kSrcLines = int64_t{kMaxRows} * kLinesPerRow;
inline constexpr int64_t kEpochLines = kWorld * kSrcLines;
static_assert(kThreads % 64 == 0, "whole waves");

struct Params {
  const bf16_t* __restrict__ v;  // [B, kWorld * kHeads, kHeadDim] bf16, last dim contiguous
  int64_t v_stride_b;            // elements
  int64_t v_stride_h;
  const float* __restrict__ lse;  // [B, kWorld * kHeads] fp32
  int64_t lse_stride_b;
  int64_t lse_stride_h;
  uint4* __restrict__ out;  // [B, kHeads, kHeadDim] bf16, contiguous
  // Every rank's region base. Index it with constants only: a runtime index turns the
  // kernarg read into a vector load, whose wait then also waits for in-flight remote stores.
  uint4* scratch[kWorld];
  uint4* self_scratch;           // scratch[rank]
  int32_t* __restrict__ epochs;  // [kMaxRows], rank-local
  int32_t rank;
};

// The DCP combine reads a NaN or +inf LSE (no local key) as -inf.
SGL_DEVICE float sanitize_lse(float lse) {
  return (lse != lse || lse == INFINITY) ? -INFINITY : lse;
}

// A payload line holds one 16-byte vector and reads all-zero until it arrives: a word of
// two +0.0 travels with its low half -0.0, which the combine's sum cannot tell from +0.0
// (the weights are >= 0 and the accumulator starts at +0.0).
SGL_DEVICE uint4 mark_payload(uint4 x) {
  x.x = x.x ? x.x : 0x8000u, x.y = x.y ? x.y : 0x8000u;
  x.z = x.z ? x.z : 0x8000u, x.w = x.w ? x.w : 0x8000u;
  return x;
}

SGL_DEVICE bool arrived(const uint4& x) {
  return x.x != 0u && x.y != 0u && x.z != 0u && x.w != 0u;
}

SGL_DEVICE void store_line(uint4* dst, const uint4& x) {
  llx_v4u v = {x.x, x.y, x.z, x.w};
  __builtin_nontemporal_store(v, reinterpret_cast<llx_v4u*>(dst));
  asm volatile("" ::: "memory");
}

// One half of a workgroup pushes, the other polls and combines: a poll issued after a
// wave's own pushes would wait for their acks (vmcnt retires in order).
inline constexpr int kBlock = 2 * kThreads;

SGL_DEVICE void
push_row(const Params& p, const bf16_t* v, const float* lse, int64_t push_line, int tid, uint32_t flag) {
  // Every load comes before the first push: one after it would wait for the push's ack.
  uint4 part[kWorld];
#pragma unroll
  for (int peer = 0; peer < kWorld; ++peer)
    part[peer] = *reinterpret_cast<const uint4*>(v + peer * kHeads * p.v_stride_h);
  if (tid < kHeads) {
    uint32_t bits[kWorld];
#pragma unroll
    for (int peer = 0; peer < kWorld; ++peer)
      bits[peer] = __float_as_uint(lse[(peer * kHeads + tid) * p.lse_stride_h]);
#pragma unroll
    for (int peer = 0; peer < kWorld; ++peer) {
      if (peer != p.rank) ll_store(p.scratch[peer] + push_line + kLseLine + tid, bits[peer], 0u, flag);
    }
  }
#pragma unroll
  for (int peer = 0; peer < kWorld; ++peer) {
    if (peer != p.rank) store_line(p.scratch[peer] + push_line + tid, mark_payload(part[peer]));
  }
}

// Triton does not contract the combine's multiply-add.
#pragma clang fp contract(off)
template <bool kBaseE>
SGL_DEVICE uint4
combine_row(const Params& p, const bf16_t* v, const float* lse, int64_t row_line, int tid, uint32_t flag) {
  const int head = tid / kHeadVecs;
  const uint4 own = *reinterpret_cast<const uint4*>(v + p.rank * kHeads * p.v_stride_h);
  const float own_lse = lse[(p.rank * kHeads + head) * p.lse_stride_h];
  // This rank's own slot of the region is never written; its lines are loaded and ignored
  // so that the loads of a round stay branch-free.
  uint4* const lines = p.self_scratch + row_line;

  // The loads of a round are issued together: after the peers' lines land, the combine is
  // one round away, not one dependent load per source.
  uint4 l[kWorld];
  bool ready;
  do {
#pragma unroll
    for (int src = 0; src < kWorld; ++src)
      l[src] = ll_load(lines + src * kSrcLines + kLseLine + head);
    ready = true;
#pragma unroll
    for (int src = 0; src < kWorld; ++src)
      ready &= src == p.rank || (l[src].y == flag && l[src].w == flag);
  } while (!ready);
  uint4 x[kWorld];
  do {
#pragma unroll
    for (int src = 0; src < kWorld; ++src)
      x[src] = ll_load(lines + src * kSrcLines + tid);
    ready = true;
#pragma unroll
    for (int src = 0; src < kWorld; ++src)
      ready &= src == p.rank || arrived(x[src]);
  } while (!ready);

  float centered[kWorld];
  float lse_max = -INFINITY;
#pragma unroll
  for (int src = 0; src < kWorld; ++src) {
    centered[src] = sanitize_lse(src == p.rank ? own_lse : __uint_as_float(l[src].x));
    lse_max = centered[src] > lse_max ? centered[src] : lse_max;
  }
  if (lse_max == -INFINITY) lse_max = 0.f;

  float acc[kVec] = {};
  float weight_sum = 0.f;
#pragma unroll
  for (int src = 0; src < kWorld; ++src) {
    // exp as Triton's AMD backend lowers tl.exp (exp2 of the product), so the weights
    // match _dcp_lse_combine_kernel's bit for bit.
    const float c = centered[src] - lse_max;
    const float w = kBaseE ? __builtin_exp2f(c * 1.4426950408889634f) : __builtin_exp2f(c);
    weight_sum += w;
    float f[kVec];
    unpack8(src == p.rank ? own : x[src], f);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      acc[e] += f[e] * w;
  }
  const float denom = fmaxf(weight_sum, 1.f);
#pragma unroll
  for (int e = 0; e < kVec; ++e)
    acc[e] = acc[e] / denom;

  // Each payload line has this one reader, which hands it back all-zero.
#pragma unroll
  for (int src = 0; src < kWorld; ++src) {
    if (src != p.rank) store_line(lines + src * kSrcLines + tid, make_uint4(0u, 0u, 0u, 0u));
  }
  return pack8(acc);
}

template <bool kBaseE>
__global__ __launch_bounds__(kBlock, 1) void dcp_ll_combine_kernel(const Params p) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x % kThreads;
  const uint32_t flag = next_flag(p.epochs, b);
  const int64_t row_line = static_cast<int64_t>(flag & 1u) * kEpochLines + static_cast<int64_t>(b) * kLinesPerRow;
  // This thread's vector of local head slot tid / kHeadVecs, in rank 0's heads.
  const bf16_t* v = p.v + b * p.v_stride_b + (tid / kHeadVecs) * p.v_stride_h + (tid % kHeadVecs) * kVec;
  const float* lse = p.lse + b * p.lse_stride_b;

  if (threadIdx.x < kThreads) {
    push_row(p, v, lse, row_line + p.rank * kSrcLines, tid, flag);
  } else {
    p.out[static_cast<int64_t>(b) * kThreads + tid] = combine_row<kBaseE>(p, v, lse, row_line, tid, flag);
  }

  // Every wave has read epochs[b] by the time all of them are here.
  lds_barrier();
  if (threadIdx.x == 0) p.epochs[b] = static_cast<int32_t>(flag);
}

struct K3DcpLLCombineKernel {
  // The merge over `comm`, the dedicated aiter CustomAllreduce* whose meta buffers hold the
  // LL scratch; the region starts region_offset bytes into the scratch.
  static void
  run(int64_t comm,
      int64_t region_offset,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView lse,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView epochs,
      bool lse_base_e) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto device = SymbolicDevice{};
    device.set_options<kDLROCM>();
    TensorMatcher({B, kWorld * kHeads, kHeadDim})
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(v);
    TensorMatcher({B, kWorld * kHeads}).with_strides({-1, -1}).with_dtype<float>().with_device(device).verify(lse);
    TensorMatcher({B, kHeads, kHeadDim}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({kMaxRows}).with_dtype<int32_t>().with_device(device).verify(epochs);
    const int64_t rows = B.unwrap();
    RuntimeCheck(rows >= 1 && rows <= kMaxRows, "the DCP LL combine takes 1-", kMaxRows, " rows");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 && v.stride(0) % kVec == 0 && v.stride(1) % kVec == 0,
        "partials must be 16-byte aligned");
    RuntimeCheck(region_offset >= 0 && region_offset % 16 == 0, "bad scratch region offset");

    Params params{
        .v = static_cast<const bf16_t*>(v.data_ptr()),
        .v_stride_b = v.stride(0),
        .v_stride_h = v.stride(1),
        .lse = static_cast<const float*>(lse.data_ptr()),
        .lse_stride_b = lse.stride(0),
        .lse_stride_h = lse.stride(1),
        .out = static_cast<uint4*>(out.data_ptr()),
        .scratch = {},
        .self_scratch = nullptr,
        .epochs = static_cast<int32_t*>(epochs.data_ptr()),
        .rank = 0,
    };
    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorld, "the DCP LL combine requires TP", kWorld);
    params.rank = communicator->rank_;
    for (int i = 0; i < kWorld; ++i) {
      params.scratch[i] = reinterpret_cast<uint4*>(
          reinterpret_cast<char*>(communicator->sg_.signals[i]) + sizeof(aiter::Signal) + region_offset);
    }
    params.self_scratch = params.scratch[communicator->rank_];
    const DLDevice dev = device.unwrap();
    if (lse_base_e) {
      LaunchKernel(static_cast<int>(rows), kBlock, dev)(dcp_ll_combine_kernel<true>, params);
    } else {
      LaunchKernel(static_cast<int>(rows), kBlock, dev)(dcp_ll_combine_kernel<false>, params);
    }
  }
};

}  // namespace k3_dcp_ll_combine
}  // namespace sglang
