package device

// MockContext implements a generic backend for testing orchestrator logic
type MockContext struct {
	DeviceID int
	Pool     map[int][]*Tensor
}

func NewMockContext() *Context {
	// We hijack the actual Context struct since it's already a wrapper.
	// But actually, internal/device uses backend-specific files.
	// To reach 95% coverage, we should test the backend logic too.
	return &Context{
		device: -1,
	}
}

// Ensure the actual Context has enough stubs for common ops even in non-specific builds
// or use a separate mock struct if we define an interface.

// For now, I'll add highly-covered tests to the existing CPU backend which is standard.
