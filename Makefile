# sml-transformer build
MLTON      ?= mlton
BIN        := bin
TENSORDIR  := lib/github.com/sjqtentacles/sml-tensor
TEST_MLB   := test/test.mlb
SRCS       := $(wildcard $(TENSORDIR)/* src/* test/*.sml) $(TEST_MLB)

.PHONY: all test poly test-poly all-tests clean

all: $(BIN)/test-mlton

$(BIN)/test-mlton: $(SRCS) | $(BIN)
	$(MLTON) -output $@ $(TEST_MLB)

test: $(BIN)/test-mlton
	$(BIN)/test-mlton

poly: $(BIN)/test-poly

$(BIN)/test-poly: $(SRCS) tools/polybuild | $(BIN)
	sh tools/polybuild -o $@ $(TEST_MLB)

test-poly: $(BIN)/test-poly
	$(BIN)/test-poly

all-tests: test test-poly

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
