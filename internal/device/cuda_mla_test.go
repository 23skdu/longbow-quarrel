//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

func TestCUDA_MLADecompressKV(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 4
	kvLoraRank := 16
	heads := 4
	qkNopeDim := 8
	vHeadDim := 8

	compKVData := make([]float32, numTokens*kvLoraRank)
	for i := range compKVData {
		compKVData[i] = float32(i%7+1) * 0.1
	}
	compressedKV := ctx.NewTensorFP32(numTokens, kvLoraRank)
	_ = compressedKV.LoadFrom(compKVData)
	defer compressedKV.Free()

	totalKRows := heads * qkNopeDim
	totalVRows := heads * vHeadDim
	totalRows := totalKRows + totalVRows
	wData := make([]float32, totalRows*kvLoraRank)
	for i := range wData {
		wData[i] = float32(i%11+1) * 0.05
	}
	wUKV := ctx.NewTensorFP32(totalRows, kvLoraRank)
	_ = wUKV.LoadFrom(wData)
	defer wUKV.Free()

	kNope, v := ctx.MLADecompressKV(compressedKV, wUKV, numTokens, kvLoraRank, heads, qkNopeDim, vHeadDim)
	if kNope == nil || v == nil {
		t.Fatalf("expected non-nil tensors, got kNope=%v, v=%v", kNope, v)
	}
	defer kNope.Free()
	defer v.Free()

	kHost := kNope.ToHostF32()
	vHost := v.ToHostF32()

	// Compute expected reference purely in Go
	refKHost := make([]float32, numTokens*totalKRows)
	refVHost := make([]float32, numTokens*totalVRows)
	for tok := 0; tok < numTokens; tok++ {
		tokKV := compKVData[tok*kvLoraRank : (tok+1)*kvLoraRank]
		for r := 0; r < totalKRows; r++ {
			wRow := wData[r*kvLoraRank : (r+1)*kvLoraRank]
			var sum float32
			for j := 0; j < kvLoraRank; j++ {
				sum += tokKV[j] * wRow[j]
			}
			refKHost[tok*totalKRows+r] = sum
		}
		for r := 0; r < totalVRows; r++ {
			wRow := wData[(totalKRows+r)*kvLoraRank : (totalKRows+r+1)*kvLoraRank]
			var sum float32
			for j := 0; j < kvLoraRank; j++ {
				sum += tokKV[j] * wRow[j]
			}
			refVHost[tok*totalVRows+r] = sum
		}
	}

	for i := range kHost {
		diff := math.Abs(float64(kHost[i] - refKHost[i]))
		if diff > 1e-4 {
			t.Fatalf("kNope[%d] mismatch: CUDA=%f, Ref=%f, diff=%f", i, kHost[i], refKHost[i], diff)
		}
	}
	for i := range vHost {
		diff := math.Abs(float64(vHost[i] - refVHost[i]))
		if diff > 1e-4 {
			t.Fatalf("v[%d] mismatch: CUDA=%f, Ref=%f, diff=%f", i, vHost[i], refVHost[i], diff)
		}
	}
}

func TestCUDA_MLAProjectQuerySplitRoPE(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 4
	heads := 4
	qkNopeDim := 8
	qkRopeDim := 8
	theta := float32(10000.0)

	inHeadDim := qkNopeDim + qkRopeDim
	qAllData := make([]float32, numTokens*heads*inHeadDim)
	for i := range qAllData {
		qAllData[i] = float32(i%13+1) * 0.2
	}
	qAll := ctx.NewTensorFP32(numTokens, heads*inHeadDim)
	_ = qAll.LoadFrom(qAllData)
	defer qAll.Free()

	posIdsData := []int32{0, 1, 2, 3}
	posIds := ctx.NewTensorI32(1, numTokens)
	_ = posIds.LoadFrom(posIdsData)
	defer posIds.Free()

	qNope, qRope := ctx.MLAProjectQuerySplitRoPE(qAll, posIds, numTokens, heads, qkNopeDim, qkRopeDim, theta)
	if qNope == nil || qRope == nil {
		t.Fatalf("expected non-nil tensors")
	}
	defer qNope.Free()
	defer qRope.Free()

	nopeHost := qNope.ToHostF32()
	ropeHost := qRope.ToHostF32()

	// Compute expected reference purely in Go
	refNopeHost := make([]float32, numTokens*heads*qkNopeDim)
	refRopeHost := make([]float32, numTokens*heads*qkRopeDim)
	halfRope := qkRopeDim / 2

	for tok := 0; tok < numTokens; tok++ {
		pos := int(posIdsData[tok])
		for h := 0; h < heads; h++ {
			baseIn := (tok*heads + h) * inHeadDim
			baseNope := (tok*heads + h) * qkNopeDim
			baseRope := (tok*heads + h) * qkRopeDim

			copy(refNopeHost[baseNope:baseNope+qkNopeDim], qAllData[baseIn:baseIn+qkNopeDim])

			for i := 0; i < halfRope; i++ {
				freq := float32(pos) * float32(math.Pow(float64(theta), float64(-2*i)/float64(qkRopeDim)))
				cosVal := float32(math.Cos(float64(freq)))
				sinVal := float32(math.Sin(float64(freq)))

				x0 := qAllData[baseIn+qkNopeDim+i]
				x1 := qAllData[baseIn+qkNopeDim+i+halfRope]

				refRopeHost[baseRope+i] = x0*cosVal - x1*sinVal
				refRopeHost[baseRope+i+halfRope] = x0*sinVal + x1*cosVal
			}
		}
	}

	for i := range nopeHost {
		diff := math.Abs(float64(nopeHost[i] - refNopeHost[i]))
		if diff > 1e-4 {
			t.Fatalf("qNope[%d] mismatch: CUDA=%f, Ref=%f, diff=%f", i, nopeHost[i], refNopeHost[i], diff)
		}
	}
	for i := range ropeHost {
		diff := math.Abs(float64(ropeHost[i] - refRopeHost[i]))
		if diff > 1e-4 {
			t.Fatalf("qRope[%d] mismatch: CUDA=%f, Ref=%f, diff=%f", i, ropeHost[i], refRopeHost[i], diff)
		}
	}
}

func TestCUDA_MLAAbsorbedQuery(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 2
	heads := 4
	qkNopeDim := 8
	kvLoraRank := 16

	qNopeData := make([]float32, numTokens*heads*qkNopeDim)
	for i := range qNopeData {
		qNopeData[i] = float32(i%9+1) * 0.1
	}
	qNope := ctx.NewTensorFP32(numTokens, heads*qkNopeDim)
	_ = qNope.LoadFrom(qNopeData)
	defer qNope.Free()

	wUKData := make([]float32, heads*qkNopeDim*kvLoraRank)
	for i := range wUKData {
		wUKData[i] = float32(i%17+1) * 0.05
	}
	wUK := ctx.NewTensorFP32(heads*qkNopeDim, kvLoraRank)
	_ = wUK.LoadFrom(wUKData)
	defer wUK.Free()

	qAbs := ctx.MLAAbsorbedQuery(qNope, wUK, numTokens, heads, qkNopeDim, kvLoraRank)
	if qAbs == nil {
		t.Fatalf("expected non-nil qAbs")
	}
	defer qAbs.Free()

	absHost := qAbs.ToHostF32()

	// Compute expected reference purely in Go
	refHost := make([]float32, numTokens*heads*kvLoraRank)
	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < heads; h++ {
			qHead := qNopeData[(tok*heads+h)*qkNopeDim : (tok*heads+h+1)*qkNopeDim]
			for j := 0; j < kvLoraRank; j++ {
				var sum float32
				for k := 0; k < qkNopeDim; k++ {
					sum += qHead[k] * wUKData[(h*qkNopeDim+k)*kvLoraRank+j]
				}
				refHost[(tok*heads+h)*kvLoraRank+j] = sum
			}
		}
	}

	for i := range absHost {
		diff := math.Abs(float64(absHost[i] - refHost[i]))
		if diff > 1e-4 {
			t.Fatalf("qAbs[%d] mismatch: CUDA=%f, Ref=%f, diff=%f", i, absHost[i], refHost[i], diff)
		}
	}
}

func TestCUDA_MLAAbsorbedDecodeAttention(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	numTokens := 1
	seqLen := 8
	heads := 4
	kvLoraRank := 16
	qkRopeDim := 8
	vHeadDim := 8
	scale := float32(1.0 / math.Sqrt(float64(qkRopeDim+kvLoraRank)))

	qAbsData := make([]float32, numTokens*heads*kvLoraRank)
	for i := range qAbsData {
		qAbsData[i] = float32(i%5+1) * 0.1
	}
	qAbs := ctx.NewTensorFP32(numTokens, heads*kvLoraRank)
	_ = qAbs.LoadFrom(qAbsData)
	defer qAbs.Free()

	qRopeData := make([]float32, numTokens*heads*qkRopeDim)
	for i := range qRopeData {
		qRopeData[i] = float32(i%3+1) * 0.15
	}
	qRope := ctx.NewTensorFP32(numTokens, heads*qkRopeDim)
	_ = qRope.LoadFrom(qRopeData)
	defer qRope.Free()

	kCacheData := make([]float32, seqLen*kvLoraRank)
	for i := range kCacheData {
		kCacheData[i] = float32(i%7+1) * 0.05
	}
	kCache := ctx.NewTensorFP32(seqLen, kvLoraRank)
	_ = kCache.LoadFrom(kCacheData)
	defer kCache.Free()

	kRopeData := make([]float32, seqLen*qkRopeDim)
	for i := range kRopeData {
		kRopeData[i] = float32(i%4+1) * 0.08
	}
	kRope := ctx.NewTensorFP32(seqLen, qkRopeDim)
	_ = kRope.LoadFrom(kRopeData)
	defer kRope.Free()

	wUVData := make([]float32, heads*vHeadDim*kvLoraRank)
	for i := range wUVData {
		wUVData[i] = float32(i%11+1) * 0.02
	}
	wUV := ctx.NewTensorFP32(heads*vHeadDim, kvLoraRank)
	_ = wUV.LoadFrom(wUVData)
	defer wUV.Free()

	output := ctx.MLAAbsorbedDecodeAttention(qAbs, qRope, kCache, kRope, wUV, numTokens, seqLen, heads, kvLoraRank, qkRopeDim, vHeadDim, scale)
	if output == nil {
		t.Fatalf("expected non-nil output")
	}
	defer output.Free()

	outHost := output.ToHostF32()

	// Compute expected reference purely in Go
	refHost := make([]float32, numTokens*heads*vHeadDim)
	scores := make([]float32, seqLen)
	latent := make([]float32, kvLoraRank)

	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < heads; h++ {
			curQAbs := qAbsData[(tok*heads+h)*kvLoraRank : (tok*heads+h+1)*kvLoraRank]
			curQRope := qRopeData[(tok*heads+h)*qkRopeDim : (tok*heads+h+1)*qkRopeDim]

			maxScore := float32(-1e30)
			for s := 0; s < seqLen; s++ {
				curK := kCacheData[s*kvLoraRank : (s+1)*kvLoraRank]
				curKRope := kRopeData[s*qkRopeDim : (s+1)*qkRopeDim]

				var dotC, dotR float32
				for j := 0; j < kvLoraRank; j++ {
					dotC += curQAbs[j] * curK[j]
				}
				for j := 0; j < qkRopeDim; j++ {
					dotR += curQRope[j] * curKRope[j]
				}
				sc := (dotC + dotR) * scale
				scores[s] = sc
				if sc > maxScore {
					maxScore = sc
				}
			}

			var sumExp float32
			for s := 0; s < seqLen; s++ {
				expVal := float32(math.Exp(float64(scores[s] - maxScore)))
				scores[s] = expVal
				sumExp += expVal
			}
			invSum := float32(1.0 / (float64(sumExp) + 1e-9))
			for s := 0; s < seqLen; s++ {
				scores[s] *= invSum
			}

			for j := 0; j < kvLoraRank; j++ {
				var sumLatent float32
				for s := 0; s < seqLen; s++ {
					sumLatent += scores[s] * kCacheData[s*kvLoraRank+j]
				}
				latent[j] = sumLatent
			}

			wHead := wUVData[(h*vHeadDim)*kvLoraRank : ((h+1)*vHeadDim)*kvLoraRank]
			for i := 0; i < vHeadDim; i++ {
				wRow := wHead[i*kvLoraRank : (i+1)*kvLoraRank]
				var outVal float32
				for j := 0; j < kvLoraRank; j++ {
					outVal += latent[j] * wRow[j]
				}
				refHost[(tok*heads+h)*vHeadDim+i] = outVal
			}
		}
	}

	for i := range outHost {
		diff := math.Abs(float64(outHost[i] - refHost[i]))
		if diff > 1e-3 {
			t.Fatalf("output[%d] mismatch: CUDA=%f, Ref=%f, diff=%f", i, outHost[i], refHost[i], diff)
		}
	}
}
