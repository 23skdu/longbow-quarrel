package sampler

import (
	"testing"
)

func TestGrammar_Regex(t *testing.T) {
	vocab := []string{"123", "abc", "!", "456"}
	g, err := NewRegexGrammar(`^[0-9]+$`, vocab)
	if err != nil {
		t.Fatalf("failed to create regex grammar: %v", err)
	}

	logits := []float32{1.0, 2.0, 3.0, 4.0}
	if err := g.Apply(logits); err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	// "123" (index 0) and "456" (index 3) match regex ^[0-9]+$
	// "abc" (index 1) and "!" (index 2) should be masked
	if logits[0] < 0 {
		t.Errorf("expected logits[0] not masked, got %f", logits[0])
	}
	if logits[3] < 0 {
		t.Errorf("expected logits[3] not masked, got %f", logits[3])
	}
}

func TestGrammar_JSONSchemaTransitions(t *testing.T) {
	vocab := []string{"{", "}", ":", ",", `"name"`, `"Alice"`, "123", "true"}
	g := NewJSONGrammar(vocab)

	// Step 1: At root, only '{' is allowed
	logits := make([]float32, len(vocab))
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[0] < 0 {
		t.Errorf("expected '{' to be allowed at root, got %f", logits[0])
	}
	if logits[4] > -1e8 {
		t.Errorf("expected key to be masked before '{', got %f", logits[4])
	}

	// Step 2: After '{', key `"name"` or '}' is allowed
	g.Update("{")
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[4] < 0 {
		t.Errorf("expected key to be allowed after '{', got %f", logits[4])
	}
	if logits[1] < 0 {
		t.Errorf("expected '}' to be allowed for empty object, got %f", logits[1])
	}
	if logits[2] > -1e8 {
		t.Errorf("expected ':' to be masked before key, got %f", logits[2])
	}

	// Step 3: After key `"name"`, only ':' is allowed
	g.Update(`"name"`)
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[2] < 0 {
		t.Errorf("expected ':' to be allowed after key, got %f", logits[2])
	}
	if logits[4] > -1e8 {
		t.Errorf("expected key to be masked before ':', got %f", logits[4])
	}

	// Step 4: After ':', values are allowed
	g.Update(":")
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[5] < 0 {
		t.Errorf("expected string value to be allowed, got %f", logits[5])
	}
	if logits[6] < 0 {
		t.Errorf("expected number value to be allowed, got %f", logits[6])
	}

	// Step 5: After value `"Alice"`, comma ',' or '}' is allowed
	g.Update(`"Alice"`)
	for i := range logits {
		logits[i] = 1.0
	}
	_ = g.Apply(logits)
	if logits[1] < 0 {
		t.Errorf("expected '}' to be allowed after value, got %f", logits[1])
	}
	if logits[3] < 0 {
		t.Errorf("expected ',' to be allowed after value, got %f", logits[3])
	}

	// Step 6: After '}', object is closed
	g.Update("}")
	if len(g.JSONState.Stack) != 0 {
		t.Errorf("expected stack to be empty after closing '}', got %d", len(g.JSONState.Stack))
	}
}

func TestGrammar_ExhaustiveCoverage(t *testing.T) {
	vocab := []string{"{", "}", "[", "]", ":", ",", `"key"`, `"val"`, "123", "-1", "true", "false", "null", " ", "invalid"}

	// 1. Regex grammar compile error
	if _, err := NewRegexGrammar(`[invalid(`, vocab); err == nil {
		t.Error("expected regex compile error for invalid pattern")
	}

	// 2. Nil grammar Apply & Update
	var nilG *Grammar
	if err := nilG.Apply([]float32{1.0}); err != nil {
		t.Errorf("expected nil error on nil Grammar.Apply: %v", err)
	}
	nilG.Update("test") // Should not panic

	// 3. Apply with len(logits) < limit
	gJSON := NewJSONGrammar(vocab)
	shortLogits := []float32{1.0, 2.0} // len 2 < vocab len 15
	if err := gJSON.Apply(shortLogits); err != nil {
		t.Errorf("Apply with short logits failed: %v", err)
	}

	// 4. InString escaping: \ followed by escaped char
	gJSON.Update(`"escaped\"quote\\and\nmore"`)

	// 5. Array transitions & string in array
	gArray := NewJSONGrammar(vocab)
	gArray.Update("[\r\n\t ") // Array with whitespaces
	if gArray.JSONState.Expectation != ExpectArrayValueOrEnd {
		t.Errorf("expected ExpectArrayValueOrEnd, got %v", gArray.JSONState.Expectation)
	}

	// String inside array -> ExpectArrayCommaOrEnd
	gArray.Update(`"item1"`)
	if gArray.JSONState.Expectation != ExpectArrayCommaOrEnd {
		t.Errorf("expected ExpectArrayCommaOrEnd, got %v", gArray.JSONState.Expectation)
	}

	// Comma in array -> ExpectArrayValueOrEnd
	gArray.Update(",")
	if gArray.JSONState.Expectation != ExpectArrayValueOrEnd {
		t.Errorf("expected ExpectArrayValueOrEnd, got %v", gArray.JSONState.Expectation)
	}

	// Literal in array -> ExpectArrayCommaOrEnd
	gArray.Update("123")
	if gArray.JSONState.Expectation != ExpectArrayCommaOrEnd {
		t.Errorf("expected ExpectArrayCommaOrEnd, got %v", gArray.JSONState.Expectation)
	}

	// Comma and another literal
	gArray.Update(",")
	gArray.Update("true")

	// Close array -> empty stack -> ExpectAnyValue
	gArray.Update("]")
	if gArray.JSONState.Expectation != ExpectAnyValue {
		t.Errorf("expected ExpectAnyValue after closing array, got %v", gArray.JSONState.Expectation)
	}

	// 6. Nested containers:
	// a) Object inside Object: { "sub": { "x": 1 } }
	gNested := NewJSONGrammar(vocab)
	gNested.Update(`{"sub":{`)
	gNested.Update(`"x":1}`) // Closes inner object -> parent is '{' -> ExpectObjectCommaOrEnd
	if gNested.JSONState.Expectation != ExpectObjectCommaOrEnd {
		t.Errorf("expected ExpectObjectCommaOrEnd after closing nested object, got %v", gNested.JSONState.Expectation)
	}
	gNested.Update("}") // Closes outer object -> empty stack

	// b) Array inside Object: { "arr": [ 1, 2 ] }
	gNested2 := NewJSONGrammar(vocab)
	gNested2.Update(`{"arr":[1,2]`) // Closes array inside object -> parent is '{' -> ExpectObjectCommaOrEnd
	if gNested2.JSONState.Expectation != ExpectObjectCommaOrEnd {
		t.Errorf("expected ExpectObjectCommaOrEnd after closing array in object, got %v", gNested2.JSONState.Expectation)
	}
	gNested2.Update("}")

	// c) Object inside Array: [ { "a": 1 } ]
	gNested3 := NewJSONGrammar(vocab)
	gNested3.Update(`[{"a":1}`) // Closes object inside array -> parent is '[' -> ExpectArrayCommaOrEnd
	if gNested3.JSONState.Expectation != ExpectArrayCommaOrEnd {
		t.Errorf("expected ExpectArrayCommaOrEnd after closing object in array, got %v", gNested3.JSONState.Expectation)
	}
	gNested3.Update("]")

	// d) Array inside Array: [ [ 1 ] ]
	gNested4 := NewJSONGrammar(vocab)
	gNested4.Update(`[[1]`) // Closes inner array inside array -> parent is '[' -> ExpectArrayCommaOrEnd
	if gNested4.JSONState.Expectation != ExpectArrayCommaOrEnd {
		t.Errorf("expected ExpectArrayCommaOrEnd after closing nested array, got %v", gNested4.JSONState.Expectation)
	}
	gNested4.Update("]")

	// 7. Literal inside object: {"k": 123}
	gObjLit := NewJSONGrammar(vocab)
	gObjLit.Update(`{"k":123`)
	if gObjLit.JSONState.Expectation != ExpectObjectCommaOrEnd {
		t.Errorf("expected ExpectObjectCommaOrEnd after literal in object, got %v", gObjLit.JSONState.Expectation)
	}

	// 8. isTokenAllowedJSON branches
	gChecks := NewJSONGrammar(vocab)
	// token == ""
	if !gChecks.isTokenAllowedJSON("") {
		t.Error("expected empty token to be allowed")
	}
	// JSONState == nil
	gNilState := &Grammar{JSONState: nil}
	if !gNilState.isTokenAllowedJSON("foo") {
		t.Error("expected allowed when JSONState is nil")
	}
	// Whitespace token
	if !gChecks.isTokenAllowedJSON("   \t\n") {
		t.Error("expected whitespace token to be allowed")
	}
	// InString == true
	gChecks.JSONState.InString = true
	if !gChecks.isTokenAllowedJSON("any_string_token") {
		t.Error("expected any token allowed when InString is true")
	}
	gChecks.JSONState.InString = false

	// Test ExpectAnyValue with -, t, f, n (inside container)
	gChecks.JSONState.Stack = []rune{'{'}
	gChecks.JSONState.Expectation = ExpectAnyValue
	for _, tok := range []string{"-", "true", "false", "null"} {
		if !gChecks.isTokenAllowedJSON(tok) {
			t.Errorf("expected %q allowed for ExpectAnyValue", tok)
		}
	}
	if gChecks.isTokenAllowedJSON("invalid") {
		t.Error("expected 'invalid' not allowed for ExpectAnyValue")
	}

	// Test ExpectColon
	gChecks.JSONState.Expectation = ExpectColon
	if !gChecks.isTokenAllowedJSON(":") || gChecks.isTokenAllowedJSON("x") {
		t.Error("ExpectColon check failed")
	}

	// Test ExpectObjectValue with -, t, f, n
	gChecks.JSONState.Expectation = ExpectObjectValue
	for _, tok := range []string{"-", "true", "false", "null"} {
		if !gChecks.isTokenAllowedJSON(tok) {
			t.Errorf("expected %q allowed for ExpectObjectValue", tok)
		}
	}
	if gChecks.isTokenAllowedJSON("invalid") {
		t.Error("expected 'invalid' not allowed for ExpectObjectValue")
	}

	// Test ExpectObjectCommaOrEnd
	gChecks.JSONState.Expectation = ExpectObjectCommaOrEnd
	if !gChecks.isTokenAllowedJSON(",") || !gChecks.isTokenAllowedJSON("}") || gChecks.isTokenAllowedJSON("x") {
		t.Error("ExpectObjectCommaOrEnd check failed")
	}

	// Test ExpectArrayValueOrEnd (inside array container)
	gChecks.JSONState.Stack = []rune{'['}
	gChecks.JSONState.Expectation = ExpectArrayValueOrEnd
	for _, tok := range []string{"-", "true", "false", "null", "]"} {
		if !gChecks.isTokenAllowedJSON(tok) {
			t.Errorf("expected %q allowed for ExpectArrayValueOrEnd", tok)
		}
	}
	if gChecks.isTokenAllowedJSON("invalid") {
		t.Error("expected 'invalid' not allowed for ExpectArrayValueOrEnd")
	}

	// Test ExpectArrayCommaOrEnd
	gChecks.JSONState.Expectation = ExpectArrayCommaOrEnd
	if !gChecks.isTokenAllowedJSON(",") || !gChecks.isTokenAllowedJSON("]") || gChecks.isTokenAllowedJSON("x") {
		t.Error("ExpectArrayCommaOrEnd check failed")
	}

	// Unknown expectation fallback
	gChecks.JSONState.Expectation = JSONExpectation(99)
	if !gChecks.isTokenAllowedJSON("x") {
		t.Error("expected true for unknown expectation")
	}

	// Object comma transition (line 203)
	gObjComma := NewJSONGrammar(vocab)
	gObjComma.Update(`{"k":1,`)
	if gObjComma.JSONState.Expectation != ExpectObjectKeyOrEnd {
		t.Errorf("expected ExpectObjectKeyOrEnd after comma in object, got %v", gObjComma.JSONState.Expectation)
	}

	// 9. isTokenAllowedRegex with Regex == nil
	gNilRegex := &Grammar{Regex: nil}
	if !gNilRegex.isTokenAllowedRegex("anything") {
		t.Error("expected true when Regex is nil")
	}

	// 10. recomputeBitmask with GrammarTypeCFG and unknown GrammarType
	gCFG := &Grammar{
		Active:    true,
		Type:      GrammarTypeCFG,
		Regex:     gJSON.Regex,
		Vocab:     vocab,
		VocabSize: len(vocab),
		Bitmask:   make([]byte, (len(vocab)+7)/8),
	}
	gCFG.recomputeBitmask()

	gUnknown := &Grammar{
		Active:    true,
		Type:      GrammarType(99),
		Vocab:     vocab,
		VocabSize: len(vocab),
		Bitmask:   make([]byte, (len(vocab)+7)/8),
	}
	gUnknown.recomputeBitmask()
}

