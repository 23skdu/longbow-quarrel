package vector

import "math/bits"

// turboquant_adc.go implements P0-03: asymmetric distance computation
// for TurboQuant.
//
// The old HNSW traversal re-quantized the float32 query for *every
// traversed edge* (an O(d^2) rotation + packing, re-created qjl matrix
// included), then walked the QJL bytes one at a time with
// bits.OnesCount8. This file removes both costs:
//
//   - the query side is prepared once per Search (TurboPrep), and
//   - each stored vector carries its QJL bits pre-packed as 64-bit
//     words (TurboQuantVector.QJLWords, filled by packQJLWords at
//     preparation; rebuilds of the index call RefreshTurboQuantWords),
//     so the per-edge work is one int8 dot plus a popcount over words,
//   - flat/IVF scans score batches over TurboPackedColumns, the packed
//     column reflection of a set of quantized vectors, which the native
//     kernels (AVX2; AVX-512 with VPOPCNTDQ under the avx512 tag)
//     score contiguously.
//
// The scoring formula is byte-for-byte the one DotTurboQuant computes,
// so the scalar reference stays the parity oracle:
//
//	score = float32(polarSum) * (scaleA*scaleB)
//	      + (totalBits - 2*hamming)/totalBits * (qjlScaleA*qjlScaleB)

// TurboPrep is the prepared query side for asymmetric distance
// computation. Vec holds the (cached) quantized query exactly as
// QuantizeTurboQuant produced it; Words is the packed form of
// Vec.QJLBits.
type TurboPrep struct {
	Vec       TurboQuantVector
	Words     []uint64
	TotalBits int
}

// PrepareTurboQuantQuery packs the QJL bytes of a quantized query (or
// stored vector) into 64-bit words for popcount scoring.
func PrepareTurboQuantQuery(v *TurboQuantVector) *TurboPrep {
	return &TurboPrep{
		Vec:       *v,
		Words:     packQJLWords(v.QJLBits),
		TotalBits: len(v.QJLBits) * 8,
	}
}

// packQJLWords expands packed sign bytes into 64-bit words. QJL bit r
// lives at Bits[r/8] & (1 << (r%8)); in the words it lands at
// Words[r/64] & (1 << (r%64)). All padding bits above len(Bits)*8 stay
// zero, so popcounting the XOR of words counts exactly the bits the
// byte loop of DotTurboQuant counts.
func packQJLWords(bitsBytes []byte) []uint64 {
	if len(bitsBytes) == 0 {
		return nil
	}
	words := make([]uint64, (len(bitsBytes)*8+63)/64)
	for b, x := range bitsBytes {
		if x == 0 {
			continue
		}
		for k := 0; k < 8; k++ {
			if x&(1<<uint(k)) != 0 {
				r := b*8 + k
				words[r/64] |= 1 << uint(r%64)
			}
		}
	}
	return words
}

// RefreshTurboQuantWords fills the QJLWords field of v in place (idempotent).
func RefreshTurboQuantWords(v *TurboQuantVector) {
	if len(v.QJLBits) == 0 {
		if v.QJLWords != nil {
			v.QJLWords = nil
		}
		return
	}
	if len(v.QJLWords) != (len(v.QJLBits)*8+63)/64 {
		v.QJLWords = packQJLWords(v.QJLBits)
	}
}

// dotTurboQuantPrepared is the scalar asymmetric distance of a prepared
// query against a stored vector. It must reproduce DotTurboQuant
// exactly: the codes accumulate in one int32, the correction factor is
// built with the same float32 operation order, and the QJL bytes are
// the shorter of the two sides.
func dotTurboQuantPrepared(q *TurboPrep, v *TurboQuantVector) float32 {
	total := q.TotalBits
	if vb := len(v.QJLBits) * 8; vb < total {
		total = vb
	}
	polarSum := turboCodesDotScalar(q.Vec.Codes, v.Codes)
	score := float32(polarSum) * (q.Vec.Scale * v.Scale)
	if total > 0 && q.Vec.QJLScale > 0 && v.QJLScale > 0 {
		qa := q.Vec.QJLBits
		qb := v.QJLBits
		qjlBytes := total / 8
		diff := 0
		for b := 0; b < qjlBytes; b++ {
			diff += bits.OnesCount8(qa[b] ^ qb[b])
		}
		corr := float32(total-2*diff) / float32(total)
		score += corr * (q.Vec.QJLScale * v.QJLScale)
	}
	return score
}

// scoreTurboPair scores one traversed edge (HNSW): the native kernel
// when built and the query was prepared with words, else the scalar.
func scoreTurboPair(ks *distanceKernelSet, q *TurboPrep, v *TurboQuantVector) float32 {
	if ks != nil && ks.turboDot != nil && len(q.Words) > 0 && len(v.QJLWords) == len(q.Words) {
		return ks.turboDot(q, v)
	}
	return dotTurboQuantPrepared(q, v)
}

// turboCodesDotScalar is the reference int8 dot product with 8-way
// unrolling (matches DotTurboQuant bit for bit).
func turboCodesDotScalar(a, b []int8) int32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var polarSum int32
	i := 0
	for ; i <= n-8; i += 8 {
		polarSum += int32(a[i+0])*int32(b[i+0]) +
			int32(a[i+1])*int32(b[i+1]) +
			int32(a[i+2])*int32(b[i+2]) +
			int32(a[i+3])*int32(b[i+3]) +
			int32(a[i+4])*int32(b[i+4]) +
			int32(a[i+5])*int32(b[i+5]) +
			int32(a[i+6])*int32(b[i+6]) +
			int32(a[i+7])*int32(b[i+7])
	}
	for ; i < n; i++ {
		polarSum += int32(a[i]) * int32(b[i])
	}
	return polarSum
}

// TurboPackedColumns is the packed column reflection of a set of
// TurboQuant vectors: codes, QJL words and the per-vector scales laid
// out contiguously so batch kernels and the mmap store can stream them.
// It is what "precomputed lookup tables" means for this codec: the
// per-vector QJL popcount tables (Words) and the scales are resolved at
// index-build time, not per query.
//
// Rows are packed only when every stored vector matches the same
// Dim / TotalBits contract; otherwise the builder returns nil and the
// index falls back to per-vector scoring.
type TurboPackedColumns struct {
	Count       int
	Dim         int
	WordsPerRow int
	TotalBits   int

	Codes    []int8    // Count*Dim
	Words    []uint64  // Count*WordsPerRow
	Scale    []float32 // Count
	QJLScale []float32 // Count
	Norm     []float32 // Count
}

// NewTurboPackedColumns packs vs (whose codes each carry dim elements)
// into contiguous columns. Rows with unexpected lengths or mixed QJL
// sizes make the pack impossible; callers then use the per-vector path.
func NewTurboPackedColumns(vs []TurboQuantVector, dim int) *TurboPackedColumns {
	if len(vs) == 0 || dim <= 0 {
		return nil
	}
	var totalBits int
	var wordsPerRow int
	for i := range vs {
		if len(vs[i].Codes) != dim {
			return nil
		}
		rowBits := len(vs[i].QJLBits) * 8
		if i == 0 {
			totalBits = rowBits
		} else if rowBits != totalBits {
			return nil
		}
	}
	if totalBits == 0 {
		wordsPerRow = 0
	} else {
		wordsPerRow = (totalBits + 63) / 64
	}

	packed := &TurboPackedColumns{
		Count:       len(vs),
		Dim:         dim,
		WordsPerRow: wordsPerRow,
		TotalBits:   totalBits,
		Codes:       make([]int8, len(vs)*dim),
		Scale:       make([]float32, len(vs)),
		QJLScale:    make([]float32, len(vs)),
		Norm:        make([]float32, len(vs)),
	}
	if wordsPerRow > 0 {
		packed.Words = make([]uint64, len(vs)*wordsPerRow)
	}

	for i := range vs {
		copy(packed.Codes[i*dim:(i+1)*dim], vs[i].Codes)
		RefreshTurboQuantWords(&vs[i])
		if wordsPerRow > 0 {
			if len(vs[i].QJLWords) != wordsPerRow {
				// Inconsistent prepacked words (built outside the
				// codec): repack from bytes.
				words := packQJLWords(vs[i].QJLBits)
				copy(packed.Words[i*wordsPerRow:(i+1)*wordsPerRow], words)
			} else {
				copy(packed.Words[i*wordsPerRow:(i+1)*wordsPerRow], vs[i].QJLWords)
			}
		}
		packed.Scale[i] = vs[i].Scale
		packed.QJLScale[i] = vs[i].QJLScale
		packed.Norm[i] = vs[i].Norm
	}
	return packed
}

// Row returns the TurboQuant vector at slot i as slices into the packed
// arrays (zero-copy views; no allocation).
func (p *TurboPackedColumns) Row(i int) *TurboQuantVector {
	v := &TurboQuantVector{
		ID:       i,
		Codes:    p.Codes[i*p.Dim : (i+1)*p.Dim],
		Scale:    p.Scale[i],
		QJLScale: p.QJLScale[i],
		Norm:     p.Norm[i],
	}
	if p.WordsPerRow > 0 {
		v.QJLWords = p.Words[i*p.WordsPerRow : (i+1)*p.WordsPerRow]
	}
	return v
}

// QueryPrepFor prepares the packed-count query side of the index.
func (p *TurboPackedColumns) QueryPrepFor(v *TurboQuantVector) *TurboPrep {
	return PrepareTurboQuantQuery(v)
}
