//go:build cuda && cgo && linux && amd64

package vector

// vector_cuda.go fills the gpu_hooks.go variables with CUDA-backed
// implementations (P0-01 batch inverted-list scoring, P0-02 complex SoA
// batch path, P0-04 GPU mini-batch k-means).
//
// Layout contracts (device side, see cuda_vector_kernels.cu):
//
//	- float32 rows staged once per index, row-major, stride == dim;
//	- complex SoA staged as two float32 planes (half the fp64
//	  bandwidth; the fp64 CPU reference agrees within the 1e-4 parity
//	  budget for the dimensions in use, tested by vector_cuda_test.go);
//	- queries packed [numQ*dim] (or split planes for the SoA kernel);
//	- one launch scores all probed lists (concatenated ids + exclusive
//	  offsets) and top-k resolves on the device, best score first with
//	  heapAdd's smaller-id tie-break.
//
// Every hook degrades to ok=false on any shape or device error so the
// CPU paths always remain available. Staged stores belong to the index
// lifetime (no per-search frees; the GPU holds one copy of the rows).

/*
#include <stdint.h>
#include <stddef.h>

#cgo cuda LDFLAGS: -L${SRCDIR} -lvector_cuda_kernels -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,/usr/local/cuda/lib64

int  vectorCudaInit(void);
void* vectorMalloc(size_t bytes);
void  vectorFree(void* p);
int   vectorUpload(void* dst, const void* src, size_t bytes);
int   vectorUploadChunked(void* dst, const void* src, size_t bytes, int chunks);
int   vectorDownload(void* dst, const void* src, size_t bytes);
int   vectorMemset32(void* dst, size_t bytes);
int   vectorSync(void);
int   vectorMQScanF32(const float* qs, int numQ, const float* rows, long stride, int dim, int count, int l2, float* out);
int   vectorMQScanCSoA(const float* qsRe, const float* qsIm, int numQ, const float* re, const float* im, int dim, int count, int l2, float* out);
int   vectorScanListsF32(const float* rows, long stride, int dim, const int* ids, const long* offsets, const int* probeLists, const long* probeBase, int numProbes, const float* q, float* scores, int maxListLen);
int   vectorTopKF32(const float* scores, int count, int numQ, int k, int l2, int* ids, float* vals);
int   vectorKmeansAssignF32(const float* rows, long stride, int dim, const int* sampleIdx, int numSamples, const float* centroids, int numCentroids, float* sums, int* counts);
int   vectorFullAssignF32(const float* rows, long stride, int dim, const float* centroids, int numCentroids, int count, int* best);
*/
import "C"

// #cgo cuda LDFLAGS are declared in the preamble above.

import (
	"unsafe"
)

// gpuFloat32Store is the staged device mirror of a uniform float32
// index (FlatIndex or IVF rows).
type gpuFloat32Store struct {
	dev   unsafe.Pointer
	count int
	dim   int
}

// gpuComplexStore stages the split float32 planes of a complex index.
type gpuComplexStore struct {
	reDev unsafe.Pointer
	imDev unsafe.Pointer
	count int
	dim   int
}

// gpuQueryF32 is a packed [numQ*dim] query block on the device.
type gpuQueryF32 struct {
	dev  unsafe.Pointer
	numQ int
	dim  int
}

// gpuQueryComplex is a packed split query block.
type gpuQueryComplex struct {
	reDev unsafe.Pointer
	imDev unsafe.Pointer
	numQ  int
	dim   int
}

// gpuListsStore stages the concatenated inverted-list ids and their
// exclusive prefix offsets (numLists+1 entries).
type gpuListsStore struct {
	idsDev     unsafe.Pointer
	offsetsDev unsafe.Pointer
	numLists   int
	total      int
}

func init() {
	if C.vectorCudaInit() != 1 {
		return
	}
	gpuStageFloat32 = gpuStageFloat32CUDA
	gpuStageComplex = gpuStageComplexCUDA
	gpuFreeFloat32 = gpuFreeFloat32CUDA
	gpuFreeComplex = gpuFreeComplexCUDA
	gpuQPackedFloat32 = gpuQPackedFloat32CUDA
	gpuQPackedComplex = gpuQPackedComplexCUDA
	gpuSearchFloat32 = gpuSearchFloat32CUDA
	gpuSearchComplex = gpuSearchComplexCUDA
	gpuStageLists = gpuStageListsCUDA
	gpuFreeLists = gpuFreeListsCUDA
	gpuScanListsFloat32 = gpuScanListsFloat32CUDA
	gpuKmeansAssign = gpuKmeansAssignCUDA
	gpuFullAssign = gpuFullAssignCUDA
}

// ---- staging --------------------------------------------------------

// gpuStageFloat32CUDA copies the rows into one contiguous block and
// uploads it.
//
// A copy out of page-locked memory runs at about three times the rate
// of a copy out of pageable memory, but the page-locked memory has to
// be allocated first, and allocating 384 MB of it takes longer than
// the copy saves. A build happens once, so the pageable copy is the
// cheaper one.
func gpuStageFloat32CUDA(rows [][]float32, dim int) (interface{}, bool) {
	if dim <= 0 || len(rows) == 0 {
		return nil, false
	}
	for _, r := range rows {
		if len(r) < dim {
			return nil, false
		}
	}
	host := make([]float32, len(rows)*dim)
	for i, r := range rows {
		copy(host[i*dim:(i+1)*dim], r[:dim])
	}
	bytes := C.size_t(len(host)) * 4
	dev := C.vectorMalloc(bytes)
	if dev == nil {
		return nil, false
	}
	// The copy is split into chunks: the driver stages one while the
	// device takes the last.
	if C.vectorUploadChunked(dev, unsafe.Pointer(&host[0]), bytes, 8) != 0 {
		C.vectorFree(dev)
		return nil, false
	}
	return &gpuFloat32Store{dev: dev, count: len(rows), dim: dim}, true
}

func gpuStageComplexCUDA(rows [][]complex128, dim int) (interface{}, bool) {
	if dim <= 0 || len(rows) == 0 {
		return nil, false
	}
	for _, r := range rows {
		if len(r) < dim {
			return nil, false
		}
	}
	n := len(rows) * dim
	re := make([]float32, n)
	im := make([]float32, n)
	for i, r := range rows {
		base := i * dim
		for j := 0; j < dim; j++ {
			re[base+j] = float32(real(r[j]))
			im[base+j] = float32(imag(r[j]))
		}
	}
	bytes := C.size_t(n) * 4
	reDev := C.vectorMalloc(bytes)
	if reDev == nil {
		return nil, false
	}
	imDev := C.vectorMalloc(bytes)
	if imDev == nil {
		C.vectorFree(reDev)
		return nil, false
	}
	if C.vectorUpload(reDev, unsafe.Pointer(&re[0]), bytes) != 0 ||
		C.vectorUpload(imDev, unsafe.Pointer(&im[0]), bytes) != 0 {
		C.vectorFree(reDev)
		C.vectorFree(imDev)
		return nil, false
	}
	return &gpuComplexStore{reDev: reDev, imDev: imDev, count: len(rows), dim: dim}, true
}

func gpuFreeFloat32CUDA(store interface{}) {
	st, ok := store.(*gpuFloat32Store)
	if !ok || st == nil || st.dev == nil {
		return
	}
	C.vectorFree(st.dev)
	st.dev = nil
}

func gpuFreeComplexCUDA(store interface{}) {
	st, ok := store.(*gpuComplexStore)
	if !ok || st == nil {
		return
	}
	if st.reDev != nil {
		C.vectorFree(st.reDev)
		st.reDev = nil
	}
	if st.imDev != nil {
		C.vectorFree(st.imDev)
		st.imDev = nil
	}
}

// ---- query packing --------------------------------------------------

func gpuQPackedFloat32CUDA(queries [][]float32) (interface{}, bool) {
	if len(queries) == 0 {
		return nil, false
	}
	dim := len(queries[0])
	if dim <= 0 {
		return nil, false
	}
	host := make([]float32, len(queries)*dim)
	for i, q := range queries {
		if len(q) != dim {
			return nil, false
		}
		copy(host[i*dim:(i+1)*dim], q)
	}
	bytes := C.size_t(len(host)) * 4
	dev := C.vectorMalloc(bytes)
	if dev == nil {
		return nil, false
	}
	if C.vectorUpload(dev, unsafe.Pointer(&host[0]), bytes) != 0 {
		C.vectorFree(dev)
		return nil, false
	}
	return &gpuQueryF32{dev: dev, numQ: len(queries), dim: dim}, true
}

func gpuQPackedComplexCUDA(queries [][]complex128) (interface{}, bool) {
	if len(queries) == 0 {
		return nil, false
	}
	dim := len(queries[0])
	if dim <= 0 {
		return nil, false
	}
	re := make([]float32, len(queries)*dim)
	im := make([]float32, len(queries)*dim)
	for i, q := range queries {
		if len(q) != dim {
			return nil, false
		}
		base := i * dim
		for j := 0; j < dim; j++ {
			re[base+j] = float32(real(q[j]))
			im[base+j] = float32(imag(q[j]))
		}
	}
	bytes := C.size_t(len(re)) * 4
	reDev := C.vectorMalloc(bytes)
	if reDev == nil {
		return nil, false
	}
	imDev := C.vectorMalloc(bytes)
	if imDev == nil {
		C.vectorFree(reDev)
		return nil, false
	}
	if C.vectorUpload(reDev, unsafe.Pointer(&re[0]), bytes) != 0 ||
		C.vectorUpload(imDev, unsafe.Pointer(&im[0]), bytes) != 0 {
		C.vectorFree(reDev)
		C.vectorFree(imDev)
		return nil, false
	}
	return &gpuQueryComplex{reDev: reDev, imDev: imDev, numQ: len(queries), dim: dim}, true
}

// ---- flat batch search ----------------------------------------------

func gpuSearchFloat32CUDA(store interface{}, qs interface{}, numQ, k int, l2 bool) ([][]SearchResult, bool) {
	st, ok := store.(*gpuFloat32Store)
	if !ok || st == nil {
		return nil, false
	}
	q, ok := qs.(*gpuQueryF32)
	if !ok || q == nil || q.numQ != numQ || q.dim != st.dim {
		return nil, false
	}
	l2i := mapBool(l2)
	if l2i < 0 {
		return nil, false
	}
	scoresDev := C.vectorMalloc(C.size_t(numQ) * C.size_t(st.count) * 4)
	if scoresDev == nil {
		return nil, false
	}
	defer C.vectorFree(scoresDev)
	if C.vectorMQScanF32(cFloat(q.dev), C.int(numQ), cFloat(st.dev), C.long(st.dim), C.int(q.dim), C.int(st.count), C.int(l2i), cFloat(scoresDev)) != 0 {
		return nil, false
	}
	idsDev := C.vectorMalloc(C.size_t(numQ) * C.size_t(k) * 4)
	if idsDev == nil {
		return nil, false
	}
	defer C.vectorFree(idsDev)
	valsDev := C.vectorMalloc(C.size_t(numQ) * C.size_t(k) * 4)
	if valsDev == nil {
		return nil, false
	}
	defer C.vectorFree(valsDev)
	if C.vectorTopKF32(cFloat(scoresDev), C.int(st.count), C.int(numQ), C.int(k), C.int(l2i), cInt(idsDev), cFloat(valsDev)) != 0 {
		return nil, false
	}
	ids := make([]int32, numQ*k)
	vals := make([]float32, numQ*k)
	if C.vectorDownload(unsafe.Pointer(&ids[0]), idsDev, C.size_t(len(ids))*4) != 0 ||
		C.vectorDownload(unsafe.Pointer(&vals[0]), valsDev, C.size_t(len(vals))*4) != 0 {
		return nil, false
	}
	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		row := make([]SearchResult, 0, k)
		for r := 0; r < k; r++ {
			id := ids[qi*k+r]
			if id < 0 {
				continue
			}
			score := vals[qi*k+r]
			if l2 {
				score = -score
			}
			row = append(row, SearchResult{ID: int(id), Score: score})
		}
		results[qi] = row
	}
	return results, true
}

func gpuSearchComplexCUDA(store interface{}, qs interface{}, numQ, k int, l2 bool) ([][]SearchResult, bool) {
	st, ok := store.(*gpuComplexStore)
	if !ok || st == nil {
		return nil, false
	}
	q, ok := qs.(*gpuQueryComplex)
	if !ok || q == nil || q.numQ != numQ || q.dim != st.dim {
		return nil, false
	}
	l2i := mapBool(l2)
	if l2i < 0 {
		return nil, false
	}
	scoresDev := C.vectorMalloc(C.size_t(numQ) * C.size_t(st.count) * 4)
	if scoresDev == nil {
		return nil, false
	}
	defer C.vectorFree(scoresDev)
	if C.vectorMQScanCSoA(cFloat(q.reDev), cFloat(q.imDev), C.int(numQ), cFloat(st.reDev), cFloat(st.imDev), C.int(q.dim), C.int(st.count), C.int(l2i), cFloat(scoresDev)) != 0 {
		return nil, false
	}
	idsDev := C.vectorMalloc(C.size_t(numQ) * C.size_t(k) * 4)
	if idsDev == nil {
		return nil, false
	}
	defer C.vectorFree(idsDev)
	valsDev := C.vectorMalloc(C.size_t(numQ) * C.size_t(k) * 4)
	if valsDev == nil {
		return nil, false
	}
	defer C.vectorFree(valsDev)
	if C.vectorTopKF32(cFloat(scoresDev), C.int(st.count), C.int(numQ), C.int(k), C.int(l2i), cInt(idsDev), cFloat(valsDev)) != 0 {
		return nil, false
	}
	ids := make([]int32, numQ*k)
	vals := make([]float32, numQ*k)
	if C.vectorDownload(unsafe.Pointer(&ids[0]), idsDev, C.size_t(len(ids))*4) != 0 ||
		C.vectorDownload(unsafe.Pointer(&vals[0]), valsDev, C.size_t(len(vals))*4) != 0 {
		return nil, false
	}
	results := make([][]SearchResult, numQ)
	for qi := 0; qi < numQ; qi++ {
		row := make([]SearchResult, 0, k)
		for r := 0; r < k; r++ {
			id := ids[qi*k+r]
			if id < 0 {
				continue
			}
			score := vals[qi*k+r]
			if l2 {
				score = -score
			}
			row = append(row, SearchResult{ID: int(id), Score: score})
		}
		results[qi] = row
	}
	return results, true
}

func mapBool(b bool) int {
	if b {
		return 1
	}
	return 0
}

// cFloat / cInt / cLong convert device pointers held in
// unsafe.Pointer to the typed cgo pointers the wrappers expect.
func cFloat(p unsafe.Pointer) *C.float { return (*C.float)(p) }
func cInt(p unsafe.Pointer) *C.int     { return (*C.int)(p) }
func cLong(p unsafe.Pointer) *C.long   { return (*C.long)(p) }

// ---- IVF list scan ---------------------------------------------------

func gpuStageListsCUDA(store interface{}, lists [][]int) (interface{}, bool) {
	st, ok := store.(*gpuFloat32Store)
	if !ok || st == nil {
		return nil, false
	}
	total := 0
	for _, l := range lists {
		total += len(l)
	}
	ids := make([]int32, total)
	offsets := make([]int64, len(lists)+1)
	for c, list := range lists {
		offsets[c+1] = offsets[c] + int64(len(list))
		for j, id := range list {
			ids[int(offsets[c])+j] = int32(id)
		}
	}
	idsDev := C.vectorMalloc(C.size_t(total) * 4)
	if idsDev == nil {
		return nil, false
	}
	offsetsDev := C.vectorMalloc(C.size_t(len(offsets)) * 8)
	if offsetsDev == nil {
		C.vectorFree(idsDev)
		return nil, false
	}
	if C.vectorUpload(idsDev, unsafe.Pointer(&ids[0]), C.size_t(maxInt(total, 1))*4) != 0 ||
		C.vectorUpload(offsetsDev, unsafe.Pointer(&offsets[0]), C.size_t(len(offsets))*8) != 0 {
		C.vectorFree(idsDev)
		C.vectorFree(offsetsDev)
		return nil, false
	}
	return &gpuListsStore{
		idsDev:     idsDev,
		offsetsDev: offsetsDev,
		numLists:   len(lists),
		total:      total,
	}, true
}

func gpuFreeListsCUDA(listsStore interface{}) {
	ls, ok := listsStore.(*gpuListsStore)
	if !ok || ls == nil {
		return
	}
	if ls.idsDev != nil {
		C.vectorFree(ls.idsDev)
		ls.idsDev = nil
	}
	if ls.offsetsDev != nil {
		C.vectorFree(ls.offsetsDev)
		ls.offsetsDev = nil
	}
}

func gpuScanListsFloat32CUDA(store interface{}, listsStore interface{}, q interface{}, listIDs []int32) ([]float32, []int32, bool) {
	st, ok := store.(*gpuFloat32Store)
	if !ok || st == nil {
		return nil, nil, false
	}
	ls, ok := listsStore.(*gpuListsStore)
	if !ok || ls == nil {
		return nil, nil, false
	}
	qq, ok := q.(*gpuQueryF32)
	if !ok || qq == nil || qq.numQ != 1 || qq.dim != st.dim {
		return nil, nil, false
	}
	if len(listIDs) == 0 || len(listIDs) > ls.numLists {
		return nil, nil, false
	}

	// Host-side probe plan: per-probe list index and the exclusive base
	// offsets of its score block in the flat output.
	offsetsHost := make([]int64, ls.numLists+1)
	downloadOK := C.vectorDownload(unsafe.Pointer(&offsetsHost[0]), ls.offsetsDev, C.size_t(len(offsetsHost))*8) == 0
	if !downloadOK {
		return nil, nil, false
	}
	probeLists := make([]int32, len(listIDs))
	probeBase := make([]int64, len(listIDs)+1)
	counts := make([]int32, len(listIDs))
	maxLen := 0
	total := int64(0)
	for p, cID := range listIDs {
		if cID < 0 || int(cID) >= ls.numLists {
			return nil, nil, false
		}
		probeLists[p] = cID
		base := offsetsHost[cID]
		end := offsetsHost[cID+1]
		counts[p] = int32(end - base)
		probeBase[p] = total
		total += end - base
		if int(counts[p]) > maxLen {
			maxLen = int(counts[p])
		}
	}
	probeBase[len(listIDs)] = total

	probeListsDev := C.vectorMalloc(C.size_t(len(probeLists)) * 4)
	if probeListsDev == nil {
		return nil, nil, false
	}
	defer C.vectorFree(probeListsDev)
	probeBaseDev := C.vectorMalloc(C.size_t(len(probeBase)) * 8)
	if probeBaseDev == nil {
		return nil, nil, false
	}
	defer C.vectorFree(probeBaseDev)
	if C.vectorUpload(probeListsDev, unsafe.Pointer(&probeLists[0]), C.size_t(len(probeLists))*4) != 0 ||
		C.vectorUpload(probeBaseDev, unsafe.Pointer(&probeBase[0]), C.size_t(len(probeBase))*8) != 0 {
		return nil, nil, false
	}

	scoresDev := C.vectorMalloc(C.size_t(total) * 4)
	if scoresDev == nil {
		return nil, nil, false
	}
	defer C.vectorFree(scoresDev)
	if C.vectorScanListsF32(cFloat(st.dev), C.long(st.dim), C.int(st.dim), cInt(ls.idsDev),
		cLong(ls.offsetsDev), cInt(probeListsDev), cLong(probeBaseDev),
		C.int(len(listIDs)), cFloat(qq.dev), cFloat(scoresDev), C.int(maxLen)) != 0 {
		return nil, nil, false
	}
	scores := make([]float32, total)
	if C.vectorDownload(unsafe.Pointer(&scores[0]), scoresDev, C.size_t(total)*4) != 0 {
		return nil, nil, false
	}
	return scores, counts, true
}

// ---- k-means (P0-04) --------------------------------------------------

func gpuKmeansAssignCUDA(store interface{}, sampleIdx []int32, dim int, centroids [][]float32, sums []float32, counts []int32) bool {
	st, ok := store.(*gpuFloat32Store)
	if !ok || st == nil || dim != st.dim || dim <= 0 {
		return false
	}
	numCentroids := len(centroids)
	if numCentroids <= 0 || len(sums) < numCentroids*dim || len(counts) < numCentroids {
		return false
	}
	if len(sampleIdx) == 0 {
		return false
	}
	for _, c := range centroids {
		if len(c) != dim {
			return false
		}
	}
	// Stage the samples and centroids for this pass (they change every
	// iteration; the row store stays resident).
	samplesDev := C.vectorMalloc(C.size_t(len(sampleIdx)) * 4)
	if samplesDev == nil {
		return false
	}
	defer C.vectorFree(samplesDev)
	if C.vectorUpload(samplesDev, unsafe.Pointer(&sampleIdx[0]), C.size_t(len(sampleIdx))*4) != 0 {
		return false
	}
	centHost := make([]float32, numCentroids*dim)
	for c, row := range centroids {
		copy(centHost[c*dim:(c+1)*dim], row)
	}
	centDev := C.vectorMalloc(C.size_t(len(centHost)) * 4)
	if centDev == nil {
		return false
	}
	defer C.vectorFree(centDev)
	if C.vectorUpload(centDev, unsafe.Pointer(&centHost[0]), C.size_t(len(centHost))*4) != 0 {
		return false
	}
	sumsDev := C.vectorMalloc(C.size_t(len(sums)) * 4)
	if sumsDev == nil {
		return false
	}
	defer C.vectorFree(sumsDev)
	countsDev := C.vectorMalloc(C.size_t(len(counts)) * 4)
	if countsDev == nil {
		return false
	}
	defer C.vectorFree(countsDev)
	if C.vectorKmeansAssignF32(cFloat(st.dev), C.long(st.dim), C.int(dim), cInt(samplesDev),
		C.int(len(sampleIdx)), cFloat(centDev), C.int(numCentroids), cFloat(sumsDev), cInt(countsDev)) != 0 {
		return false
	}
	if C.vectorDownload(unsafe.Pointer(&sums[0]), sumsDev, C.size_t(numCentroids*dim)*4) != 0 ||
		C.vectorDownload(unsafe.Pointer(&counts[0]), countsDev, C.size_t(numCentroids)*4) != 0 {
		return false
	}
	return true
}

func gpuFullAssignCUDA(store interface{}, dim int, centroids [][]float32) ([]int32, bool) {
	st, ok := store.(*gpuFloat32Store)
	if !ok || st == nil || dim != st.dim || dim <= 0 {
		return nil, false
	}
	numCentroids := len(centroids)
	if numCentroids <= 0 {
		return nil, false
	}
	for _, c := range centroids {
		if len(c) != dim {
			return nil, false
		}
	}
	centHost := make([]float32, numCentroids*dim)
	for c, row := range centroids {
		copy(centHost[c*dim:(c+1)*dim], row)
	}
	centDev := C.vectorMalloc(C.size_t(len(centHost)) * 4)
	if centDev == nil {
		return nil, false
	}
	defer C.vectorFree(centDev)
	if C.vectorUpload(centDev, unsafe.Pointer(&centHost[0]), C.size_t(len(centHost))*4) != 0 {
		return nil, false
	}
	bestDev := C.vectorMalloc(C.size_t(st.count) * 4)
	if bestDev == nil {
		return nil, false
	}
	defer C.vectorFree(bestDev)
	if C.vectorFullAssignF32(cFloat(st.dev), C.long(st.dim), C.int(dim), cFloat(centDev),
		C.int(numCentroids), C.int(st.count), cInt(bestDev)) != 0 {
		return nil, false
	}
	best := make([]int32, st.count)
	if C.vectorDownload(unsafe.Pointer(&best[0]), bestDev, C.size_t(st.count)*4) != 0 {
		return nil, false
	}
	return best, true
}
