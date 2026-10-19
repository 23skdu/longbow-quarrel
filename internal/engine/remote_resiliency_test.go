package engine

import (
	"context"
	"encoding/binary"
	"math"
	"net"
	"strconv"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/flight"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

type faultInjectionFlightServer struct {
	flight.BaseFlightServer
	returnPayload []byte
}

func (s *faultInjectionFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	reader, err := flight.NewRecordReader(stream)
	if err != nil {
		return err
	}
	defer reader.Release()
	for reader.Next() {
	}
	if err := reader.Err(); err != nil {
		return err
	}
	return stream.Send(&flight.PutResult{AppMetadata: s.returnPayload})
}

func encodeFloatsToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(f))
	}
	return buf
}

func TestRemoteWorkerEngine_FaultInjectionAndFailover(t *testing.T) {
	// 1. Start Fallback Worker Flight Server
	expectedOutput := []float32{1.23, 4.56, 7.89, 0.12}
	fallbackPayload := encodeFloatsToBytes(expectedOutput)

	srv := flight.NewServerWithMiddleware(nil)
	if err := srv.Init("localhost:0"); err != nil {
		t.Fatalf("Flight server Init failed: %v", err)
	}
	srv.RegisterFlightService(&faultInjectionFlightServer{returnPayload: fallbackPayload})
	go func() { _ = srv.Serve() }()
	t.Cleanup(srv.Shutdown)

	fallbackAddr := srv.Addr().String()
	host, portStr, err := net.SplitHostPort(fallbackAddr)
	if err != nil {
		t.Fatalf("SplitHostPort failed: %v", err)
	}
	fallbackPort, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("Atoi port failed: %v", err)
	}

	// 2. Create RemoteWorkerEngine pointing to a dead primary worker (closed port)
	deadListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to listen for dead primary: %v", err)
	}
	deadPort := deadListener.Addr().(*net.TCPAddr).Port
	deadListener.Close() // Close immediately to simulate total worker crash / fault injection

	cfg := config.Config{
		Layers: 1,
		Dim:    4,
	}

	engine, err := NewRemoteWorkerEngine("127.0.0.1", deadPort, cfg)
	if err != nil {
		t.Fatalf("Failed to create RemoteWorkerEngine: %v", err)
	}
	defer engine.Close()

	// 3. Configure Fallback Worker
	if err := engine.SetFallbackWorker(host, fallbackPort); err != nil {
		t.Fatalf("SetFallbackWorker failed: %v", err)
	}
	engine.SetNextStage(host, fallbackPort+100)

	// 4. Test ForwardShardedLayer with dead primary -> should fail over to fallback seamlessly
	devCtx := device.NewContext()
	defer devCtx.Free()

	inTensor := devCtx.NewTensorFP32(1, 4)
	defer inTensor.Free()
	inTensor.LoadFrom([]float32{1.0, 2.0, 3.0, 4.0})

	ctx := context.Background()
	outTensor, err := engine.ForwardShardedLayer(ctx, 0, 0, 4, inTensor)
	if err != nil {
		t.Fatalf("ForwardShardedLayer failed despite fallback worker: %v", err)
	}
	defer outTensor.Free()

	outData := outTensor.ToHostF32()
	if len(outData) != len(expectedOutput) {
		t.Fatalf("Output length expected %d, got %d", len(expectedOutput), len(outData))
	}
	for i, exp := range expectedOutput {
		if math.Abs(float64(outData[i]-exp)) > 1e-4 {
			t.Fatalf("Mismatch at index %d: expected %f, got %f", i, exp, outData[i])
		}
	}
}
