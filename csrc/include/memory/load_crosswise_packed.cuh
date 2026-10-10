#pragma once
/* Packed k-pair staging for 8-bit crosswise operands. */
#include <utils/define.cuh>
#include <utils/tensor.cuh>

namespace astrai {
namespace gemm {

/*
 * One operand element's raw byte. The 8-bit storages are class types (fp8) as
 * often as integers, and a numeric conversion would round the bit pattern away
 * — every packed-grid elementwise path reads bytes through here.
 */
template <typename ElemT> DEVICE_FORCEINLINE unsigned raw_byte(const ElemT& e) {
    return *reinterpret_cast<const unsigned char*>(&e);
}

/*
 * k-pair packed staging for the 8-bit crosswise operand: a packed unit is a
 * 16-bit (row, k-pair) cell, so the two ADJACENT runs (2j, 2j+1) suffice
 * and one thread owns its unit outright — no lane pairing, no shfl_xor, one
 * PRMT per output word (against the canonical carry's 4 LDG + 48 PRMT +
 * 16 STS + 16 shfl per 64B). The staged tile is the transposed [kK/2][rows]
 * grid the 16-bit crosswise path already reads: ldmatrix.trans turns 16
 * packed rows (one mma k-segment = 32 contract bytes) into m16n8k32
 * fragments, so that reader's lane offsets and steps carry over unchanged.
 * The element-granular sibling covers the row tail, a misaligned base and a
 * thinner bus, so the fast carry stays predication-free.
 */
template <typename SmemLayout, typename ElemT, int kThreads>
DEVICE_FORCEINLINE void pack_crosswise_general(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                                               const ElemT* __restrict__ operand,
                                               int64_t rows,
                                               int64_t contract,
                                               int64_t ld,
                                               int tid,
                                               int64_t k_base,
                                               int64_t block_row) {
    static_assert(sizeof(ElemT) == 1, "the packed grid stages 8-bit elements");
    constexpr int kRowsTile = SmemLayout::kChunks * 8; // 8 units per 16B chunk
    const int units = SmemLayout::kRows * kRowsTile;
    for (int u = tid; u < units; u += kThreads) {
        const int j = u / kRowsTile;
        const int row = u % kRowsTile;
        const int64_t r = block_row + row;
        const bool row_ok = r < rows;
        const int64_t k0 = k_base + 2 * (int64_t)j;
        /*
         * One unit = the two contract elements (k0, k0+1) of one row; a
         * contract tail or row tail contributes zero bytes. The bytes are
         * RAW (fp8 has no integer conversion — a cast through float rounds
         * the bit pattern away).
         */
        const unsigned lo = row_ok && k0 < contract ? raw_byte(operand[k0 * ld + r]) : 0u;
        const unsigned hi = row_ok && k0 + 1 < contract ? raw_byte(operand[(k0 + 1) * ld + r]) : 0u;
        *reinterpret_cast<unsigned short*>(tile(j, row * 2)) = (unsigned short)(lo | (hi << 8));
    }
}

template <typename SmemLayout, typename ElemT, int kThreads, bool kOn> struct PairPackCarry {
    static_assert(!kOn || sizeof(ElemT) == 1, "the packed carry stages the 8-bit crosswise path");
    static constexpr int kRowsTile = kOn ? SmemLayout::kChunks * 8 : 0;  // 8 units/chunk
    static constexpr int kGroups = kOn ? kRowsTile / 16 : 0;             // 16-row groups per tile
    static constexpr int kUnits = kOn ? SmemLayout::kRows * kGroups : 0; // 32B units
    static constexpr bool kFits = kOn && kUnits <= kThreads;

    uint4 v[2];   // this unit's two runs: k = 2j and 2j+1, 16 rows each
    int jrow = 0; // the packed row this thread owns
    int grp = 0;  // its 16-row group
    bool active = false;
    bool fast = false;

    DEVICE_FORCEINLINE void issue(const ElemT* __restrict__ operand,
                                  int64_t rows,
                                  int64_t contract,
                                  int64_t ld,
                                  int tid,
                                  int64_t k_base,
                                  int64_t block_row) {
        if constexpr (!kFits)
            return;
        active = tid < kUnits;
        if (!active)
            return;
        jrow = tid / kGroups;
        grp = tid % kGroups;
        const int64_t r0 = block_row + grp * 16;
        /*
         * r0 is a multiple of 16 and p*ld preserves alignment whenever ld has
         * it, so both runs of a unit share one verdict (same rule as the
         * general loader). No shuffle pairs this thread with another, so a
         * boundary unit may simply take the elementwise arm below.
         */
        const bool run_aligned =
            ((reinterpret_cast<uintptr_t>(operand) | (ld * (int64_t)sizeof(ElemT))) & 15) == 0;
        fast = (r0 + 15 < rows) && run_aligned;
        const int64_t k0 = k_base + 2 * (int64_t)jrow;
        if (fast) {
#pragma unroll
            for (int i = 0; i < 2; ++i)
                v[i] = k0 + i < contract
                           ? __ldg(reinterpret_cast<const uint4*>(operand + (k0 + i) * ld + r0))
                           : make_uint4(0u, 0u, 0u, 0u);
        } else {
            v[0] = make_uint4(0u, 0u, 0u, 0u);
            v[1] = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    /*
     * PRMT + STS for the unit issue() fetched. Each output word packs two
     * adjacent rows' k-pair bytes — [a.b(2p), b.b(2p), a.b(2p+1), b.b(2p+1)]
     * out of run words a = 2j and b = 2j+1 — so one byte-perm per word turns
     * the two runs into the packed row's 32 bytes, no exchange needed.
     */
    template <typename TileT>
    DEVICE_FORCEINLINE void commit(TileT tile,
                                   const ElemT* __restrict__ operand,
                                   int64_t rows,
                                   int64_t contract,
                                   int64_t ld,
                                   int tid,
                                   int64_t k_base,
                                   int64_t block_row) const {
        if constexpr (!kFits) {
            /*
             * Off (kOn false): nothing to stage. Thin bus (kOn true but not
             * kFits): issue() staged nothing, the round trip stays synchronous
             * here (the packed grid, still one thread per unit).
             */
            if constexpr (kOn)
                pack_crosswise_general<SmemLayout, ElemT, kThreads>(tile, operand, rows, contract,
                                                                    ld, tid, k_base, block_row);
            return;
        }
        if (!active)
            return;
        if (fast) {
            const unsigned* a = reinterpret_cast<const unsigned*>(v);
            const unsigned* b = a + 4;
            unsigned w[8];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                w[2 * q] = __byte_perm(a[q], b[q], 0x5140u);
                w[2 * q + 1] = __byte_perm(a[q], b[q], 0x7362u);
            }
            /*
             * The 32B unit spans exactly two 16B chunks of the packed row;
             * each is swizzled on its own, so they are two stores.
             */
            *reinterpret_cast<uint4*>(tile(jrow, grp * 32)) = make_uint4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<uint4*>(tile(jrow, grp * 32 + 16)) =
                make_uint4(w[4], w[5], w[6], w[7]);
            return;
        }
        // Boundary unit (row tail / misaligned base): elementwise pack.
        const int64_t r0 = block_row + grp * 16;
        const int64_t k0 = k_base + 2 * (int64_t)jrow;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int64_t r = r0 + i;
            const bool row_ok = r < rows;
            const unsigned lo = (row_ok && k0 < contract) ? raw_byte(operand[k0 * ld + r]) : 0u;
            const unsigned hi =
                (row_ok && k0 + 1 < contract) ? raw_byte(operand[(k0 + 1) * ld + r]) : 0u;
            *reinterpret_cast<unsigned short*>(tile(jrow, grp * 32 + i * 2)) =
                (unsigned short)(lo | (hi << 8));
        }
    }
};

/*
 * The packed-grid route: the two-phase carry when the bus fits it, the
 * elementwise packed grid otherwise.
 */
template <typename SmemLayout, typename ElemT, int kThreads>
DEVICE_FORCEINLINE void load_crosswise_paired(Tensor<PtrEngine<ElemT>, SmemLayout> tile,
                                              const ElemT* __restrict__ operand,
                                              int64_t rows,
                                              int64_t contract,
                                              int64_t ld,
                                              int tid,
                                              int64_t k_base,
                                              int64_t block_row) {
    PairPackCarry<SmemLayout, ElemT, kThreads, true> carry;
    carry.issue(operand, rows, contract, ld, tid, k_base, block_row);
    carry.commit(tile, operand, rows, contract, ld, tid, k_base, block_row);
}

} // namespace gemm
} // namespace astrai
