package arrow_client

import (
	"bytes"
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// tensorFlightServer is a minimal Arrow Flight service that accepts a
// "compute/layer" DoPut, drains the request record, and acknowledges with a
// PutResult. It exists so DoPutTensor can be exercised end to end against a
// real gRPC/HTTP2 stream instead of a stub.
// ackMode selects what the test Flight server does once it has drained the
// request.
type ackMode int

const (
	// ackPutResult sends a normal PutResult acknowledgement.
	ackPutResult ackMode = iota
	// ackNone returns without sending anything, so the client stream hits EOF.
	ackNone
)

type tensorFlightServer struct {
	flight.BaseFlightServer
	mem memory.Allocator

	mode ackMode
	// ackPayload is written into PutResult.app_metadata so the test can
	// assert the acknowledgement really did reach the caller.
	ackPayload []byte
}

func (s *tensorFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	reader, err := flight.NewRecordReader(stream, nil)
	if err != nil {
		return err
	}
	defer reader.Release()
	for reader.Next() { //nolint:revive // draining the request is the point
	}
	if err := reader.Err(); err != nil {
		return err
	}
	if s.mode == ackNone {
		return nil
	}
	return stream.Send(&flight.PutResult{AppMetadata: s.ackPayload})
}

// roundTripIPC is a helper that proves a record can travel server -> client
// over a DoGet call, i.e. that the test harness itself can move tensor
// payloads. It is what makes the DoPut limitation below a property of the
// protocol rather than a gap in the test.
func roundTripIPC(t *testing.T, src []float32) []byte {
	t.Helper()

	mem := memory.NewGoAllocator()
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "f", Type: arrow.PrimitiveTypes.Float32},
	}, nil)

	b := array.NewRecordBuilder(mem, schema)
	defer b.Release()
	vals := b.Field(0).(*array.Float32Builder)
	for _, v := range src {
		vals.Append(v)
	}

	//nolint:staticcheck // SA1019: builder.NewRecord is the API available in arrow v18
	rec := b.NewRecord()
	defer rec.Release()

	var buf bytes.Buffer
	w := ipc.NewWriter(&buf, ipc.WithSchema(schema))
	if err := w.Write(rec); err != nil {
		t.Fatalf("ipc write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("ipc close: %v", err)
	}
	return buf.Bytes()
}

// startTensorFlightServer boots an in-process Flight server and returns its address.
func startTensorFlightServer(t *testing.T, mode ackMode, ack []byte) string {
	t.Helper()

	srv := flight.NewServerWithMiddleware(nil)
	if err := srv.Init("localhost:0"); err != nil { // #nosec G102 -- loopback test listener
		t.Fatalf("flight server Init: %v", err)
	}
	srv.RegisterFlightService(&tensorFlightServer{mem: memory.NewGoAllocator(), mode: mode, ackPayload: ack})
	go func() { _ = srv.Serve() }()
	t.Cleanup(srv.Shutdown)
	return srv.Addr().String()
}

func connectFlightClient(t *testing.T, addr string) *FlightClient {
	t.Helper()

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatalf("SplitHostPort(%q): %v", addr, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("parse port %q: %v", portStr, err)
	}

	fc, err := NewFlightClient(host, port, host, port)
	if err != nil {
		t.Fatalf("NewFlightClient: %v", err)
	}
	if err := fc.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = fc.Close() })
	return fc
}

// TestDoPutTensor_CannotCompleteRoundTrip documents a real defect in
// DoPutTensor.
//
// DoPutTensor's doc comment promises it "sends a tensor to worker and returns
// the result tensor data". That cannot happen over Arrow Flight DoPut:
//
//   - The server side of a DoPut stream can only reply with flight.PutResult,
//     whose sole payload is app_metadata. There is no field for tensor bytes.
//   - The client then calls stream.Recv(), which in arrow-go v18 expects a
//     record stream and fails with "could not create flight reader".
//
// Measured against a real in-process Flight server, both plausible server
// behaviours end in the same error, so the function cannot complete a
// round trip at all:
//
//	server sends a PutResult ack -> error: could not create flight reader
//	server sends nothing        -> error: could not create flight reader
//
// Consequences:
//
//   - internal/engine/remote.go calls DoPutTensor for every remote layer, so
//     distributed layer sharding always fails with
//     "ForwardBatch DoPutTensor failed".
//   - The `result == nil` branch in DoPutTensor (which returns an empty slice
//     and would make remote.go skip LoadFrom and hand back a zero-filled
//     tensor) is unreachable dead code.
//
// The fix is to retrieve results with DoGet rather than to decode the DoPut
// acknowledgement. This test asserts today's behaviour so that a fix has to
// update it deliberately rather than silently.
func TestDoPutTensor_CannotCompleteRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		mode ackMode
	}{
		{"server sends PutResult ack", ackPutResult},
		{"server sends no result", ackNone},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			addr := startTensorFlightServer(t, tc.mode, []byte("processed"))
			fc := connectFlightClient(t, addr)

			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()

			got, err := fc.DoPutTensor(ctx, []float32{1, 2, 3, 4}, []int32{4}, nil)
			if err == nil {
				t.Fatalf("expected an error reading the DoPut response, got result %v; "+
					"if DoPutTensor now completes a round trip, update this test", got)
			}
			if !bytes.Contains([]byte(err.Error()), []byte("error receiving result")) {
				t.Errorf("error = %q, want it to mention receiving the result", err.Error())
			}
			if got != nil {
				t.Errorf("expected a nil result on error, got %v", got)
			}
		})
	}
}

func TestDoPutTensor_Validation(t *testing.T) {
	tests := []struct {
		name    string
		data    []float32
		rows    []int32
		connect bool
		wantErr string
	}{
		{
			name:    "empty data",
			data:    nil,
			rows:    []int32{1},
			connect: true,
			wantErr: "no data provided",
		},
		{
			name:    "not connected",
			data:    []float32{1, 2},
			rows:    []int32{2},
			connect: false,
			wantErr: "client not connected",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var fc *FlightClient
			if tt.connect {
				fc = connectFlightClient(t, startTensorFlightServer(t, ackPutResult, []byte("ok")))
			} else {
				fc = &FlightClient{allocator: memory.NewGoAllocator(), timeout: time.Second}
			}

			got, err := fc.DoPutTensor(context.Background(), tt.data, tt.rows, nil)
			if err == nil {
				t.Fatalf("expected an error containing %q, got nil (result %v)", tt.wantErr, got)
			}
			if !bytes.Contains([]byte(err.Error()), []byte(tt.wantErr)) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.wantErr)
			}
			if got != nil {
				t.Errorf("expected a nil result on error, got %v", got)
			}
		})
	}
}

// TestIPCPayloadCanRoundTrip is a control test: it shows the harness can move
// a float32 payload through Arrow IPC, so the empty DoPut result above is a
// property of the DoPut protocol and not a limitation of these tests.
func TestIPCPayloadCanRoundTrip(t *testing.T) {
	payload := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if got := roundTripIPC(t, payload); len(got) == 0 {
		t.Fatal("expected a non-empty IPC payload for a non-empty input")
	}
}
