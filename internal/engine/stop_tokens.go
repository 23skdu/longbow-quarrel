package engine

// stopTokenSet builds a lookup set for the configured stop tokens.
//
// config.Config.StopTokens was declared but never read anywhere in the tree:
// a user who configured stop tokens got them silently ignored. The set is
// built once per engine rather than per token, because the decode loop calls
// this for every generated token.
func stopTokenSet(stopTokens []int) map[int]struct{} {
	if len(stopTokens) == 0 {
		return nil
	}
	set := make(map[int]struct{}, len(stopTokens))
	for _, t := range stopTokens {
		set[t] = struct{}{}
	}
	return set
}

// isStopToken reports whether token is one of the configured stop tokens.
func isStopToken(set map[int]struct{}, token int) bool {
	if len(set) == 0 {
		return false
	}
	_, ok := set[token]
	return ok
}
