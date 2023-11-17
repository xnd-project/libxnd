$LOAD_PATH.unshift File.expand_path('../../lib', __FILE__)
require 'xnd'

require 'minitest/autorun'
require 'minitest/hooks'

Minitest::Test.parallelize_me!

MAX_DIM = NDT::MAX_DIM

def assert_strict_equal x1, x2
  assert x1.strict_equal(x2)
  assert_equal x1, x2
end

def assert_strict_unequal x1, x2
  assert !x1.strict_equal(x2)
  refute_equal x1, x2
end

def type_equal t, u
  t.match(u) && u.match(t)
end

def get_inf_or_normal_range start, stop, exclude_end
  if start == Float::INFINITY && stop != Float::INFINITY
    Range.new 0, stop, exclude_end
  elsif start != Float::INFINITY && stop == Float::INFINITY
    Range.new start, 9999, exclude_end
  elsif start == Float::INFINITY && stop == Float::INFINITY
    Range.new 0, 9999, exclude_end
  else
    Range.new start, stop, exclude_end
  end
end

# ======================================================================
#                             Primitive types 
# ======================================================================

PRIMITIVE = [
  'int8', 'int16', 'int32', 'int64',
  'uint8', 'uint16', 'uint32', 'uint64',
  'float32', 'float64',
  'complex64', 'complex128'
]

BOOL_PRIMITIVE = ['bool']

EMPTY_TEST_CASES = [
  [0, "%s"],
  [[], "0 * %s"],
  [[0], "1 * %s"],
  [[0, 0], "var(offsets=[0, 2]) * %s"],
  [[{"a" => 0, "b" => 0}] * 3, "3 * {a: int64, b: %s}"]
]

def empty_test_cases(val=0)
  [
    [val, "%s"],
    [[], "0 * %s"],
    [[val], "1 * %s"],
    [[val, val], "var(offsets=[0, 2]) * %s"],
    [[{"a" => 0, "b" => val}] * 3, "3 * {a: int64, b: %s}"]
  ]
end

# ======================================================================
#                             Typed values
# ======================================================================

DTYPE_EMPTY_TEST_CASES = [
  # Tuples
  [[], "()"],
  [[0], "(int8)"],
  [[0, 0], "(int8, int64)"],
  [[0, [0+0i]], "(uint16, (complex64))"],
  [[0, [0+0i]], "(uint16, (complex64), pack=1)"],
  [[0, [0+0i]], "(uint16, (complex64), pack=2)"],
  [[0, [0+0i]], "(uint16, (complex64), pack=4)"],
  [[0, [0+0i]], "(uint16, (complex64), pack=8)"],
  [[0, [0+0i]], "(uint16, (complex64), align=16)"],
  [[[]], "(0 * bytes)"],
  [[[], []], "(0 * bytes, 0 * string)"],
  [[[''], [0.0i] * 2, [""] * 3], "(1 * bytes, 2 * complex128, 3 * string)"],
  [[[""], [[0.0i, [[""] * 2] * 10]] * 2, [""] * 3], "(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * string)"],
  [[0, [[[[0.0] * 5] * 4] * 3] * 2], "(int64, 2 * 3 * Some(4 * 5 * float64))"],
  [[0, [[[[0.0] * 5] * 4] * 3] * 2], "(int64, 2 * 3 * ref(4 * 5 * float64))"],

  # Optional tuples
  [nil, "?()"],
  [nil, "?(int8)"],
  [nil, "?(int8, int64)"],
  [nil, "?(uint16, (complex64))"],
  [nil, "?(uint16, (complex64), pack=1)"],
  [nil, "?(uint16, (complex64), pack=2)"],
  [nil, "?(uint16, (complex64), pack=4)"],
  [nil, "?(uint16, (complex64), pack=8)"],
  [nil, "?(uint16, (complex64), align=16)"],
  [nil, "?(0 * bytes)"],
  [nil, "?(0 * bytes, 0 * string)"],
  [nil, "?(1 * bytes, 2 * complex128, 3 * string)"],
  [nil, "?(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * string)"],

  # Tuples with optional elements
  [[nil], "(?int8)"],
  [[nil, 0], "(?int8, int64)"],
  [[0, nil], "(int8, ?int64)"],
  [[nil, nil], "(?int8, ?int64)"],
  [nil, "?(?int8, ?int64)"],

  [[0, nil], "(uint16, ?(complex64))"],
  [[0, [nil]], "(uint16, (?complex64))"],
  [[0, nil], "(uint16, ?(?complex64))"],

  [[nil, [0+0i]], "(?uint16, (complex64), pack=1)"],
  [[0, nil], "(uint16, ?(complex64), pack=1)"],
  [[0, [nil]], "(uint16, (?complex64), pack=1)"],

  [[[]], "(0 * ?bytes)"],
  [[[nil]], "(1 * ?bytes)"],
  [[[nil] * 10], "(10 * ?bytes)"],
  [[[], []], "(0 * ?bytes, 0 * ?string)"],
  [[[nil] * 5, [""] * 2], "(5 * ?bytes, 2 * string)"],
  [[[""] * 5, [nil] * 2], "(5 * bytes, 2 * ?string)"],
  [[[nil] * 5, [nil] * 2], "(5 * ?bytes, 2 * ?string)"],

  [[[nil], [nil] * 2, [nil] * 3], "(1 * ?bytes, 2 * ?complex128, 3 * ?string)"],

  [[[nil], [ [0.0i, [ [""] * 2 ] * 10 ] ] * 2, [""] * 3 ], "(1 * ?bytes, 2 * (complex128, 10 * 2 * string), 3 * string)"],
  [[[""], [nil] * 2, [""] * 3], "(1 * bytes, 2 * ?(complex128, 10 * 2 * string), 3 * string)"],
  [[[""], [[0.0i, [[""] * 2] * 10]] * 2, [nil] * 3], "(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * ?string)"],
  [[[nil], [nil] * 2, [""] * 3], "(1 * ?bytes, 2 * ?(complex128, 10 * 2 * string), 3 * string)"],
  [[[nil], [[0.0i, [[""] * 2] * 10]] * 2, [nil] * 3], "(1 * ?bytes, 2 * (complex128, 10 * 2 * string), 3 * ?string)"],
  [[[nil], [nil] * 2, [nil] * 3], "(1 * ?bytes, 2 * ?(complex128, 10 * 2 * string), 3 * ?string)"],

  [[[""],[[0.0i, [[nil] * 2] * 10]] * 2, [""] * 3], "(1 * bytes, 2 * (complex128, 10 * 2 * ?string), 3 * string)"],
  [[[nil], [[0.0i, [[nil] * 2] * 10]] * 2, [""] * 3], "(1 * ?bytes, 2 * (complex128, 10 * 2 * ?string), 3 * string)"],
  [[[nil], [[nil , [[nil] * 2] * 10]] * 2, [""] * 3], "(1 * ?bytes, 2 * (?complex128, 10 * 2 * ?string), 3 * string)"],
  [[[""], [[nil, [[nil] * 2] * 10]] * 2, [nil] * 3], "(1 * bytes, 2 * (?complex128, 10 * 2 * ?string), 3 * ?string)"],

  [[0, [[[[nil] * 5] * 4] * 3] * 2], "(int64, 2 * 3 * Some(4 * 5 * ?float64))"],
  [[0, [[[[nil] * 5] * 4] * 3] * 2], "(int64, 2 * 3 * ref(4 * 5 * ?float64))"],

  # Records
  [{}, "{}"],
  [{'x' => 0}, "{x: int8}"],
  [{'x' => 0, 'y' => 0}, "{x: int8, y: int64}"],
  [{'x' => 0, 'y' => {'z' => 0+0i}}, "{x: uint16, y: {z: complex64}}"],
  [{'x' => 0, 'y' => {'z' => 0+0i}}, "{x: uint16, y: {z: complex64}, pack=1}"],
  [{'x' => 0, 'y' => {'z' => 0+0i}}, "{x: uint16, y: {z: complex64}, pack=2}"],
  [{'x' => 0, 'y' => {'z' => 0+0i}}, "{x: uint16, y: {z: complex64}, pack=4}"],
  [{'x' => 0, 'y' => {'z' => 0+0i}}, "{x: uint16, y: {z: complex64}, pack=8}"],
  [{'x' => 0, 'y' => {'z' => 0+0i}}, "{x: uint16, y: {z: complex64}, align=16}"],
  [{'x' => []}, "{x: 0 * bytes}"],
  [{'x' => [], 'y' => []}, "{x: 0 * bytes, y: 0 * string}"],
  [{'x' => [""], 'y' => [0.0i] * 2, 'z' => [""] * 3}, "{x: 1 * bytes, y: 2 * complex128, z: 3 * string}"],
  [{'x' => [""], 'y' => [{'a' => 0.0i, 'b' => [[""] * 2] * 10}] * 2, 'z' => [""] * 3}, "{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}"],
  [{'x' => 0, 'y' => [[[[0.0] * 5] * 4] * 3] * 2}, "{x: int64, y: 2 * 3 * Some(4 * 5 * float64)}"],
  [{'x' => 0, 'y' => [[[[0.0] * 5] * 4] * 3] * 2}, "{x: int64, y: 2 * 3 * ref(4 * 5 * float64)}"],

  # Optional records
  [nil, "?{}"],
  [nil, "?{x: int8}"],
  [nil, "?{x: int8, y: int64}"],
  [nil, "?{x: uint16, y: {z: complex64}}"],
  [nil, "?{x: uint16, y: {z: complex64}, pack=1}"],
  [nil, "?{x: uint16, y: {z: complex64}, pack=2}"],
  [nil, "?{x: uint16, y: {z: complex64}, pack=4}"],
  [nil, "?{x: uint16, y: {z: complex64}, pack=8}"],
  [nil, "?{x: uint16, y: {z: complex64}, align=16}"],
  [nil, "?{x: 0 * bytes}"],
  [nil, "?{x: 0 * bytes, y: 0 * string}"],
  [nil, "?{x: 1 * bytes, y: 2 * complex128, z: 3 * string}"],
  [nil, "?{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}"],

  # Records with optional elements
  [{'x' => nil}, "{x: ?int8}"],
  [{'x' => nil, 'y' => 0}, "{x: ?int8, y: int64}"],
  [{'x' => 0, 'y' => nil}, "{x: int8, y: ?int64}"],
  [{'x' => nil, 'y' => nil}, "{x: ?int8, y: ?int64}"],
  [nil, "?{x: ?int8, y: ?int64}"],

  [{'x' => 0, 'y' => nil}, " {x: uint16, y: ?{z: complex64}}"],
  [{'x' => 0, 'y' => {'z' => nil}}, "{x: uint16, y: {z: ?complex64}}"],
  [{'x' => 0, 'y' => nil}, "{x: uint16, y: ?{z: ?complex64}}"],

  [{'x' => nil, 'y' => {'z' => 0+0i}}, "{x: ?uint16, y: {z: complex64}, pack=1}"],
  [{'x' => 0, 'y' => nil}, "{x: uint16, y: ?{z: complex64}, pack=1}"],
  [{'x' => 0, 'y' => {'z' => nil}}, "{x: uint16, y: {z: ?complex64}, pack=1}"],

  [{'x' => []}, "{x: 0 * ?bytes}"],
  [{'x' => [nil] * 1}, "{x: 1 * ?bytes}"],
  [{'x' => [nil] * 10}, "{x: 10 * ?bytes}"],
  [{'x' => [], 'y' => []}, "{x: 0 * ?bytes, y: 0 * ?string}"],
  [{'x' => [nil] * 5, 'y' => [""] * 2}, "{x: 5 * ?bytes, y: 2 * string}"],
  [{'x' => [""] * 5, 'y' => [nil] * 2}, "{x: 5 * bytes, y: 2 * ?string}"],
  [{'x' => [nil] * 5, 'y' => [nil] * 2}, "{x: 5 * ?bytes, y: 2 * ?string}"],

  [{'x' => [nil], 'y' => [nil] * 2, 'z' => [nil] * 3}, "{x: 1 * ?bytes, y: 2 * ?complex128, z: 3 * ?string}"],
  [{'x' => [nil], 'y' => [{'a' => 0.0i, 'b' => [[""] * 2] * 10}] * 2, 'z' => [""] * 3}, "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}"],
  [{'x' => [""], 'y' => [nil] * 2, 'z' => [""] * 3}, "{x: 1 * bytes, y: 2 * ?{a: complex128, b: 10 * 2 * string}, z: 3 * string}"],
  [{'x' => [""], 'y' => [{'a' => 0.0i, 'b' => [[""] * 2] * 10}] * 2, 'z' => [nil] * 3}, "{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * ?string}"],
  [{'x' => [nil], 'y' => [nil] * 2, 'z' => [""] * 3}, "{x: 1 * ?bytes, y: 2 * ?{a: complex128, b: 10 * 2 * string}, z: 3 * string}"],
  [{'x' => [nil], 'y' => [{'a' => 0.0i, 'b' => [[""] * 2] * 10}] * 2, 'z' => [nil] * 3}, "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * ?string}"],
  [{'x' => [nil], 'y' => [nil] * 2, 'z' => [nil] * 3}, "{x: 1 * ?bytes, y: 2 * ?{a: complex128, b: 10 * 2 * string}, z: 3 * ?string}"],

  [{'x' => [""], 'y' => [{'a' => 0.0i, 'b' => [[nil] * 2] * 10}] * 2, 'z' => [""] * 3}, "{x: 1 * bytes, y: 2 * {a: complex128, b: 10 * 2 * ?string}, z: 3 * string}"],
  [{'x' => [nil], 'y' => [{'a' => 0.0i, 'b' => [[nil] * 2] * 10}] * 2, 'z' => [""] * 3}, "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * ?string}, z: 3 * string}"],
  [{'x' => [nil], 'y' => [{'a' => nil, 'b' => [[nil] * 2] * 10}] * 2, 'z' => [""] * 3}, "{x: 1 * ?bytes, y: 2 * {a: ?complex128, b: 10 * 2 * ?string}, z: 3 * string}"],
  [{'x' => [""], 'y' => [{'a' => nil, 'b' => [[nil] * 2] * 10}] * 2, 'z' => [nil] * 3}, "{x: 1 * bytes, y: 2 * {a: ?complex128, b: 10 * 2 * ?string}, z: 3 * ?string}"],

  [{'x' => 0, 'y' => [[[[nil] * 5] * 4] * 3] * 2}, "{x: int64, y: 2 * 3 * Some(4 * 5 * ?float64)}"],
  [{'x' => 0, 'y' => [[[[nil] * 5] * 4] * 3] * 2}, "{x: int64, y: 2 * 3 * ref(4 * 5 * ?float64)}"],

  # Primitive types
  [false, "bool"],

  [0, "int8"],
  [0, "int16"],
  [0, "int32"],
  [0, "int64"],

  [0, "uint8"],
  [0, "uint16"],
  [0, "uint32"],
  [0, "uint64"],

  [0.0, "float32"],
  [0.0, "float64"],

  [0+0i, "complex64"],
  [0+0i, "complex128"],

  [0+0i, "complex64"],
  [0+0i, "complex128"],

  # Optional primitive types
  [nil, "?bool"],

  [nil, "?int8"],
  [nil, "?int16"],
  [nil, "?int32"],
  [nil, "?int64"],

  [nil, "?uint8"],
  [nil, "?uint16"],
  [nil, "?uint32"],
  [nil, "?uint64"],

  [nil, "?float32"],
  [nil, "?float64"],

  [nil, "?complex64"],
  [nil, "?complex128"],

  [nil, "?complex64"],
  [nil, "?complex128"],

  # References
  [false, "&bool"],

  [0, "&int8"],
  [0, "&int16"],
  [0, "&int32"],
  [0, "&int64"],

  [0, "ref(uint8)"],
  [0, "ref(uint16)"],
  [0, "ref(uint32)"],
  [0, "ref(uint64)"],

  [0, "ref(ref(uint8))"],
  [0, "ref(ref(uint16))"],
  [0, "ref(ref(uint32))"],
  [0, "ref(ref(uint64))"],

  [0.0, "ref(float32)"],
  [0.0, "ref(float64)"],

  [0+0i, "ref(complex64)"],
  [0+0i, "ref(complex128)"],

  [[], "ref(0 * bool)"],
  [[0], "ref(1 * int16)"],
  [[0] * 2, "ref(2 * int32)"],
  [[[0] * 3] * 2, "ref(2 * 3 * int8)"],

  [[], "ref(ref(0 * bool))"],
  [[0], "ref(ref(1 * int16))"],
  [[0] * 2, "ref(ref(2 * int32))"],
  [[[0] * 3] * 2, "ref(ref(2 * 3 * int8))"],

  [[], "ref(!0 * bool)"],
  [[0], "ref(!1 * int16)"],
  [[0] * 2, "ref(!2 * int32)"],
  [[[0] * 3] * 2, "ref(!2 * 3 * int8)"],

  [[], "ref(ref(!0 * bool))"],
  [[0], "ref(ref(!1 * int16))"],
  [[0] * 2, "ref(ref(!2 * int32))"],
  [[[0] * 3] * 2, "ref(ref(!2 * 3 * int8))"],

  # Optional references
  [nil, "?&bool"],

  [nil, "?&int8"],
  [nil, "?&int16"],
  [nil, "?&int32"],
  [nil, "?&int64"],

  [nil, "?ref(uint8)"],
  [nil, "?ref(uint16)"],
  [nil, "?ref(uint32)"],
  [nil, "?ref(uint64)"],

  [nil, "?ref(ref(uint8))"],
  [nil, "?ref(ref(uint16))"],
  [nil, "?ref(ref(uint32))"],
  [nil, "?ref(ref(uint64))"],

  [nil, "?ref(float32)"],
  [nil, "?ref(float64)"],

  [nil, "?ref(complex64)"],
  [nil, "?ref(complex128)"],

  [nil, "?ref(0 * bool)"],
  [nil, "?ref(1 * int16)"],
  [nil, "?ref(2 * int32)"],
  [nil, "?ref(2 * 3 * int8)"],

  [nil, "?ref(ref(0 * bool))"],
  [nil, "ref(?ref(0 * bool))"],
  [nil, "?ref(?ref(0 * bool))"],
  [nil, "?ref(ref(1 * int16))"],
  [nil, "ref(?ref(1 * int16))"],
  [nil, "?ref(?ref(1 * int16))"],
  [nil, "?ref(ref(2 * int32))"],

  [nil, "?ref(!2 * 3 * int8)"],
  [nil, "?ref(ref(!2 * 3 * int32))"],

  # References to types with optional data
  [nil, "&?bool"],

  [nil, "&?int8"],
  [nil, "&?int16"],
  [nil, "&?int32"],
  [nil, "&?int64"],

  [nil, "ref(?uint8)"],
  [nil, "ref(?uint16)"],
  [nil, "ref(?uint32)"],
  [nil, "ref(?uint64)"],

  [nil, "ref(ref(?uint8))"],
  [nil, "ref(ref(?uint16))"],
  [nil, "ref(ref(?uint32))"],
  [nil, "ref(ref(?uint64))"],

  [nil, "ref(?float32)"],
  [nil, "ref(?float64)"],

  [nil, "ref(?complex64)"],
  [nil, "ref(?complex128)"],

  [[], "ref(0 * ?bool)"],
  [[nil], "ref(1 * ?int16)"],
  [[nil] * 2, "ref(2 * ?int32)"],
  [[[nil] * 3] * 2, "ref(2 * 3 * ?int8)"],

  [[], "ref(ref(0 * ?bool))"],
  [[nil], "ref(ref(1 * ?int16))"],
  [[nil] * 2, "ref(ref(2 * ?int32))"],

  [[[nil] * 3] * 2, "ref(!2 * 3 * ?int8)"],
  [[[nil] * 3] * 2, "ref(ref(!2 * 3 * ?int8))"],

  # Constructors
  [false, "Some(bool)"],

  [0, "Some(int8)"],
  [0, "Some(int16)"],
  [0, "Some(int32)"],
  [0, "Some(int64)"],

  [0, "Some(uint8)"],
  [0, "Some(uint16)"],
  [0, "Some(uint32)"],
  [0, "Some(uint64)"],

  [0.0, "Some(float32)"],
  [0.0, "Some(float64)"],

  [0+0i, "Some(complex64)"],
  [0+0i, "Some(complex128)"],

  [[0], "ThisGuy(1 * int16)"],
  [[0] * 2, "ThisGuy(2 * int32)"],
  [[[0.0] * 3] * 2, "ThisGuy(2 * 3 * float32)"],

  [[[0.0] * 3] * 2, "ThisGuy(!2 * 3 * float32)"],

  # Optional constructors
  [nil, "?Some(bool)"],

  [nil, "?Some(int8)"],
  [nil, "?Some(int16)"],
  [nil, "?Some(int32)"],
  [nil, "?Some(int64)"],

  [nil, "?Some(uint8)"],
  [nil, "?Some(uint16)"],
  [nil, "?Some(uint32)"],
  [nil, "?Some(uint64)"],

  [nil, "?Some(float32)"],
  [nil, "?Some(float64)"],

  [nil, "?Some(complex64)"],
  [nil, "?Some(complex128)"],

  [nil, "?ThisGuy(0 * int16)"],
  [nil, "?ThisGuy(1 * int16)"],
  [nil, "?ThisGuy(2 * int32)"],
  [nil, "?ThisGuy(2 * 3 * float32)"],

  [nil, "?ThisGuy(!2 * 3 * float32)"],

  # Constructors with an optional data type argument
  [nil, "Some(?bool)"],

  [nil, "Some(?int8)"],
  [nil, "Some(?int16)"],
  [nil, "Some(?int32)"],
  [nil, "Some(?int64)"],

  [nil, "Some(?uint8)"],
  [nil, "Some(?uint16)"],
  [nil, "Some(?uint32)"],
  [nil, "Some(?uint64)"],

  [nil, "Some(?float32)"],
  [nil, "Some(?float64)"],

  [nil, "Some(?complex64)"],
  [nil, "Some(?complex128)"],

  [[], "ThisGuy(0 * ?int16)"],
  [[nil], "ThisGuy(1 * ?int16)"],
  [[nil] * 2, "ThisGuy(2 * ?int32)"],
  [[[nil] * 3] * 2, "ThisGuy(2 * 3 * ?float32)"],

  [[[nil] * 3] * 2, "ThisGuy(!2 * 3 * ?float32)"],
]

#
# Test case for richcompare:
#   v: value
#   t: type of v
#   u: equivalent type for v
#   w: value different from v
#   eq: expected comparison result
#
Struct.new("T", :v, :t, :u, :w, :eq)
T = Struct::T

EQUAL_TEST_CASES = [
  # Tuples
  T.new([],
        "()",
        nil,
        nil,
        true),

  T.new([100],
        "(int8)",
        "?(int8)",
        [101],
        true),

  T.new([2**7-1, 2**63-1],
        "(int8, int64)",
        "(int8, ?int64) ",
        [2**7-2, 2**63-1],
        true),

  T.new([2**16-1, [1.2312222+28i]],
        "(uint16, (complex64))",
        "(uint16, ?(complex64))",
        [2**16-1, [1.23122+28i]],
        true),

  T.new([1, [1e22+2i]],
        "(uint32, (complex64), pack=1)",
        "(uint32, (complex64), align=16)",
        [1, [1e22+3i]],
        true),

  T.new([[]],
        "(0 * bytes)",
        nil,
        nil,
        true),

  T.new([[], []],
        "(0 * bytes, 0 * string)",
        nil,
        nil,
        true),

  T.new([['x'], [1.2i] * 2, ["abc"] * 3],
        "(1 * bytes, 2 * complex128, 3 * string)",
        "(1 * bytes, 2 * ?complex128, 3 * string)",
        [['x'], [1.2i] * 2, ["ab"] * 3],
        true),

  T.new([['123'], [[3e22+0.2i, [["xyz"] * 2] * 10]] * 2, ["1"] * 3],
        "(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * string)",
        "(1 * bytes, 2 * ?(complex128, 10 * 2 * string), 3 * string)",
        [['1234'], [[3e22+0.2i, [["xyz"] * 2] * 10]] * 2, ["1"] * 3],
        true),

  T.new([10001, [[[[2.250] * 5] * 4] * 3] * 2],
        "(int64, 2 * 3 * Some(4 * 5 * float64))",
        "(int64, 2 * 3 * Some(4 * 5 * ?float64))",
        [10001, [[[[2.251] * 5] * 4] * 3] * 2],
        true),

  T.new([-2**63, [[[[10.1] * 5] * 4] * 3] * 2],
        "(int64, 2 * 3 * ref(4 * 5 * float64))",
        "(int64, 2 * 3 * ?ref(4 * 5 * float64))",
        [-2**63+1, [[[[10.1] * 5] * 4] * 3] * 2],
        true),

  # Optional tuples
  T.new(nil,
        "?()",
        nil,
        nil,
        false),

  T.new(nil,
        "?(int8)",
        nil,
        nil,
        false),

  T.new(nil,
        "?(1 * bytes, 2 * (complex128, 10 * 2 * string), 3 * string)",
        nil,
        nil,
        false),

  # Tuples with optional elements
  T.new([nil],
        "(?int8)",
        nil,
        nil,
        false),

  T.new([nil, 0],
        "(?int8, int64)",
        nil,
        nil,
        false),

  T.new([0, nil],
        "(int8, ?int64)",
        nil,
        nil,
        false),

  T.new([nil, nil],
        "(?int8, ?int64)",
        nil,
        nil,
        false),

  T.new(nil,
        "?(?int8, ?int64)",
        nil,
        nil,
        false),

  T.new([0, nil],
        "(uint16, ?(complex64))",
        nil,
        nil,
        false),

  T.new([0, [nil]],
        "(uint16, (?complex64))",
        nil,
        nil,
        false),

  T.new([0, nil],
        "(uint16, ?(?complex64))",
        nil,
        nil,
        false),

  T.new([nil, [0+0i]],
        "(?uint16, (complex64), pack=1)",
        nil,
        nil,
        false),

  T.new([0, nil],
        "(uint16, ?(complex64), pack=1)",
        nil,
        nil,
        false),

  T.new([0, [nil]],
        "(uint16, (?complex64), pack=1)",
        nil,
        nil,
        false),

  # Records
  T.new({},
        "{}",
        nil,
        nil,
        true),

  T.new({'x' => 2**31-1},
        "{x: int32}",
        "{x: ?int32}",
        {'x' => 2**31-2},
        true),

  T.new({'x' => -128, 'y' => -1},
        "{x: int8, y: int64}",
        "{x: int8, y: int64, pack=1}",
        {'x' => -127, 'y' => -1},
        true),

  T.new({'x' => 2**32-1, 'y' => {'z' => 10000001e3+36.1e7i}},
        "{x: uint32, y: {z: complex64}}",
        "{x: uint32, y: {z: complex64}}",
        {'x' => 2**32-2, 'y' => {'z' => 10000001e3+36.1e7i}},
        true),

  T.new({'x' => 255, 'y' => {'z' => 2+3i}},
        "{x: uint8, y: {z: complex64}, pack=1}",
        "{x: uint8, y: {z: complex64}, pack=4}",
        {'x' => 255, 'y' => {'z' => 3+3i}},
        true),

  T.new({'x' => 10001, 'y' => {'z' => "abc"}},
        "{x: uint16, y: {z: fixed_string(100)}, pack=2}",
        "{x: uint16, y: {z: fixed_string(100)}, align=8}",
        {'x' => 10001, 'y' => {'z' => "abcd"}},
        true),

  T.new({'x' => []},
        "{x: 0 * bytes}",
        nil,
        nil,
        true),

  T.new({'x' => [], 'y' => []},
        "{x: 0 * bytes, y: 0 * string}",
        nil,
        nil,
        true),

  T.new({'x' => ["".b], 'y' => [2.0i] * 2, 'z' => ["y"] * 3},
        "{x: 1 * fixed_bytes(size=512), y: 2 * complex128, z: 3 * string}",
        "{x: 1 * fixed_bytes(size=512, align=256), y: 2 * complex128, z: 3 * string}",
        nil,
        true),

  T.new({'x' => 100, 'y' => [[[[301.0] * 5] * 4] * 3] * 2},
        "{x: int64, y: 2 * 3 * Some(4 * 5 * ?float64)}",
        "{x: int64, y: 2 * 3 * ?Some(4 * 5 * ?float64)}",
        {'x' => 100, 'y' => [[[[nil] * 5] * 4] * 3] * 2},
        true),

  # Optional records
  T.new(nil,
        "?{}",
        "?{}",
        nil,
        false),

  T.new(nil,
        "?{x: int8}",
        "?{x: int8}",
        nil,
        false),

  T.new({'x' => 101},
        "?{x: int8}",
        "{x: int8}",
        {'x' => 100},
        true),

  T.new(nil,
        "?{x: 0 * bytes}",
        "?{x: 0 * bytes}",
        nil,
        false),

  T.new(nil,
        "?{x: 0 * bytes, y: 0 * string}",
        "?{x: 0 * bytes, y: 0 * string}",
        nil,
        false),

  # Records with optional elements
  T.new({'x' => nil},
        "{x: ?int8}",
        "{x: ?int8}",
        nil,
        false),

  T.new({'x' => nil, 'y' => 0},
        "{x: ?int8, y: int64}",
        "{x: ?int8, y: int64}",
        {'x' => 0, 'y' => 0},
        false),

  T.new({'x' => 100, 'y' => nil},
        "{x: int8, y: ?int64}",
        "{x: int8, y: ?int64}",
        {'x' => 100, 'y' => 10},
        false),

  T.new({'x' => nil, 'y' => nil},
        "{x: ?int8, y: ?int64}",
        "{x: ?int8, y: ?int64}",
        {'x' => 1, 'y' => 1},
        false),

  T.new(nil,
        "?{x: ?int8, y: ?int64}",
        "?{x: ?int8, y: ?int64}",
        nil,
        false),

  T.new({'x' => 0, 'y' => nil},
        "{x: uint16, y: ?{z: complex64}}",
        "{x: uint16, y: ?{z: complex64}}",
        {'x' => 1, 'y' => nil},
        false),

  #  FIXME: The 'z' => 2 does not seem to bode well with XND. Fix.
  # T.new({'x' => 0, 'y' => {'z' => nil}},
  #   "{x: uint16, y: {z: ?complex64}}",
  #   "{x: uint16, y: {z: ?complex64}}",
  #   {'x' => 0, 'y' => {'z' => 2}},
  #   false),

  T.new({'x' => 0, 'y' => nil},
        "{x: uint16, y: ?{z: ?complex64}}",
        "{x: uint16, y: ?{z: ?complex64}}",
        {'x' => 0, 'y' => {'z' => 1+10i}},
        false),

  T.new({'x' => nil, 'y' => {'z' => 0+0i}},
        "{x: ?uint16, y: {z: complex64}, pack=1}",
        "{x: ?uint16, y: {z: complex64}, align=16}",
        {'x' => 256, 'y' => {'z' => 0+0i}},
        false),

  T.new({'x' => 0, 'y' => nil},
        "{x: uint16, y: ?{z: complex64}, pack=1}",
        "{x: uint16, y: ?{z: complex64}, pack=1}",
        {'x' => 0, 'y' => {'z' => 0+1i}},
        false),

  T.new({'x' => 0, 'y' => {'z' => nil}},
        "{x: ?uint16, y: {z: ?complex64}, pack=1}",
        "{x: ?uint16, y: {z: ?complex64}, pack=1}",
        {'x' => nil, 'y' => {'z' => nil}},
        false),

  T.new({'x' => []},
        "{x: 0 * ?bytes}",
        "{x: 0 * ?bytes}",
        nil,
        true),

  T.new({'x' => [nil] * 1},
        "{x: 1 * ?bytes}",
        "{x: 1 * ?bytes}",
        nil,
        false),

  T.new({'x' => [nil] * 10},
        "{x: 10 * ?bytes}",
        "{x: 10 * ?bytes}",
        {'x' => ["123"] * 10},
        false),

  T.new({'x' => [], 'y' => []},
        "{x: 0 * ?bytes, y: 0 * ?string}",
        "{x: 0 * ?bytes, y: 0 * ?string}",
        nil,
        true),

  T.new({'x' => ["123"] * 5, 'y' => ["abc"] * 2},
        "{x: 5 * ?bytes, y: 2 * string}",
        "{x: 5 * bytes, y: 2 * string}",
        {'x' => ["12345"] * 5, 'y' => ["abc"] * 2},
        true),

  T.new({'x' => ["-123"], 'y' => [1e200+10i] * 2, 'z' => ["t" * 100] * 3},
        "{x: 1 * ?bytes, y: 2 * ?complex128, z: 3 * ?string}",
        "{x: 1 * ?bytes, y: 2 * ?complex128, z: 3 * ?string}",
        {'x' => ["-123"], 'y' => [1e200+10i] * 2, 'z' => ["t" * 100 + "u"] * 3},
        true),

  T.new({'x' => ["x"], 'y' => [{'a' => 0.0i, 'b' => [["c"] * 2] * 10}] * 2, 'z' => ["a"] * 3},
        "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * string}, z: 3 * string}",
        "{x: 1 * ?bytes, y: 2 * {a: complex128, b: 10 * 2 * ?string}, z: 3 * string}",
        {'x' => ["x"], 'y' => [{'a' => 1.0i, 'b' => [["c"] * 2] * 10}] * 2, 'z' => ["a"] * 3},
        true),

  # Primitive types
  T.new(false, "bool", "?bool", true, true),
  T.new(0, "?bool", "bool", 1, true),

  T.new(127, "int8", "?int8", 100, true),
  T.new(127, "?int8", "int8", 100, true),

  T.new(-127, "int16", "?int16", 100, true),
  T.new(-127, "?int16", "int16", 100, true),

  T.new(127, "int32", "?int32", 100, true),
  T.new(127, "?int32", "int32", 100, true),

  T.new(127, "int64", "?int64", 100, true),
  T.new(127, "?int64", "int64", 100, true),

  T.new(127, "uint8", "?uint8", 100, true),
  T.new(127, "?uint8", "uint8", 100, true),

  T.new(127, "uint16", "?uint16", 100, true),
  T.new(127, "?uint16", "uint16", 100, true),

  T.new(127, "uint32", "?uint32", 100, true),
  T.new(127, "?uint32", "uint32", 100, true),

  T.new(127, "uint64", "?uint64", 100, true),
  T.new(127, "?uint64", "uint64", 100, true),

  T.new(1.122e11, "float32", "?float32", 2.111, true),
  T.new(1.233e10, "?float32", "float32", 3.111, true),

  T.new(1.122e11, "float64", "?float64", 2.111, true),
  T.new(1.233e10, "?float64", "float64", 3.111, true),

  T.new(1.122e11+1i, "complex64", "?complex64", 1.122e11+2i, true),
  T.new(1.122e11+1i, "?complex64", "complex64", 1.1e11+1i, true),

  T.new(1.122e11-100i, "complex128", "?complex128", 1.122e11-101i, true),
  T.new(1.122e11-100i, "?complex128", "complex128", 1.122e10-100i, true),
]
