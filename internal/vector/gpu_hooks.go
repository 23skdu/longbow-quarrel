package vector

// gpu_hooks.go declares the CUDA hooks the indexes consult. The
// untagged (default, avx512) builds leave every variable nil and every
// call site falls back to the CPU path; the cuda-tagged build
// (vector_cuda.go) fills them when the device initialises.
//
// The interface{} store parameters are *gpuFloat32Store / *gpuComplexStore
// under the cuda tag. A non-cuda build can never hold such a value:
// only vector_cuda.go creates them, and it is the only build that reads
// them back, so the hooks type-check without importing CUDA types here.

var (
	// gpuStageFloat32 uploads a uniform float32 row set (row-major,
	// dim stride) to the device. ok=false on any failure (no device,
	// out of memory, ragged rows); callers must not retry after a
	// failure.
	gpuStageFloat32 func(rows [][]float32, dim int) (store interface{}, ok bool)

	// gpuStageComplex uploads the split real/imag planes (P0-02) as
	// float32 on the device (half the bandwidth of float64; the
	// GeForce cards running this repo have no usable fp64 rate). The
	// cuda-tagged parity test asserts 1e-4 agreement with the fp64
	// CPU reference.
	gpuStageComplex func(rows [][]complex128, dim int) (store interface{}, ok bool)

	// gpuFreeFloat32 / gpuFreeComplex release a staged store (no-op on
	// nil is never needed: callers only hold stores after successful
	// staging, and each hook has its matching free).
	gpuFreeFloat32 func(store interface{})
	gpuFreeComplex func(store interface{})

	// gpuQPackedFloat32 packs a query block [numQ][dim] row-major.
	gpuQPackedFloat32 func(queries [][]float32) (qs interface{}, ok bool)

	// gpuQPackedComplex packs query blocks as split float32 planes.
	gpuQPackedComplex func(queries [][]complex128) (qs interface{}, ok bool)

	// gpuSearchFloat32 scores every staged row against every staged
	// query with one kernel launch and resolves top-k on the GPU.
	// Returns the per-query results in query order, or ok=false when
	// the shape does not fit the kernel contract (the caller then
	// uses the CPU path; nothing is freed - the store stays staged).
	gpuSearchFloat32 func(store interface{}, qs interface{}, numQ int, k int, l2 bool) (results [][]SearchResult, ok bool)

	// gpuSearchComplex mirrors gpuSearchFloat32 for the SoA planes.
	gpuSearchComplex func(store interface{}, qs interface{}, numQ int, k int, l2 bool) (results [][]SearchResult, ok bool)

	// gpuStageLists stages the inverted-file id arrays of an IVF index
	// (concatenated int32 ids plus per-list lengths) so the scan hook
	// can address rows without per-list uploads. listsStore is opaque
	// and consumed by gpuScanListsFloat32 only.
	gpuStageLists func(store interface{}, lists [][]int) (listsStore interface{}, ok bool)

	// gpuFreeLists releases the staged lists.
	gpuFreeLists func(listsStore interface{})

	// gpuScanListsFloat32 scores the rows of the given IVF lists
	// (listIDs index the staged lists) against one query with a
	// single launch and downloads the scores. The flat return is the
	// concatenation of per-list score blocks and counts[i] is the
	// number of scores for list i, so the caller can slice per list:
	// scores[offset : offset+counts[i]].
	gpuScanListsFloat32 func(store interface{}, listsStore interface{}, q interface{}, listIDs []int32) (scores []float32, counts []int32, ok bool)

	// gpuKmeansAssign runs one mini-batch k-means assignment pass on
	// the device: for every staged sample listed in sampleIdx it
	// finds the nearest centroid and adds (sample, best) into sums /
	// counts, which the caller zeroed. The centroid update stays on
	// the CPU so both builds share the update arithmetic exactly.
	gpuKmeansAssign func(store interface{}, sampleIdx []int32, dim int, centroids [][]float32, sums []float32, counts []int32) bool

	// gpuFullAssign assigns every staged sample to its nearest
	// centroid; the returned slice indexes the samples in sample
	// order (position, not sampleIdx content).
	gpuFullAssign func(store interface{}, dim int, centroids [][]float32) (best []int32, ok bool)
)
