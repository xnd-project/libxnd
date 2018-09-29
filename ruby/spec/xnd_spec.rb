require 'spec_helper'

describe XND do
  context ".new" do
    context "Type Inference" do
      context "Tuple" do
        d = {'a' => XND::T.new(2.0, "bytes".b), 'b' => XND::T.new("str", Float::INFINITY) }
        typeof_d = "{a: (float64, bytes), b: (string, float64)}"

        [
          [XND::T.new(), "()"],
          [XND::T.new(XND::T.new()), "(())"],
          [XND::T.new(XND::T.new(), XND::T.new()), "((), ())"],
          [XND::T.new(XND::T.new(XND::T.new()), XND::T.new()), "((()), ())"],
          [XND::T.new(XND::T.new(XND::T.new()), XND::T.new(XND::T.new(), XND::T.new())),
           "((()), ((), ()))"],
          [XND::T.new(1, 2, 3), "(int64, int64, int64)"],
          [XND::T.new(1.0, 2, "str"), "(float64, int64, string)"],
          [XND::T.new(1.0, 2, XND::T.new("str", "bytes".b, d)),
           "(float64, int64, (string, bytes, #{typeof_d}))"]
        ].each do |v, t|
          it "type: #{t}" do
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(XND::TypeInference.convert_xnd_t_to_ruby_array(v))
          end
        end
      end

      context "Record" do
        d = {'a' => XND::T.new(2.0, "bytes".b), 'b' => XND::T.new("str", Float::INFINITY) }
        typeof_d = "{a: (float64, bytes), b: (string, float64)}"

        [
          [{}, "{}"],
          [{'x' => {}}, "{x: {}}"],
          [{'x' => {}, 'y' => {}}, "{x: {}, y: {}}"],
          [{'x' => {'y' => {}}, 'z' => {}}, "{x: {y: {}}, z: {}}"],
          [{'x' => {'y' => {}}, 'z' => {'a' => {}, 'b' => {}}}, "{x: {y: {}}, z: {a: {}, b: {}}}"],
          [d, typeof_d]
        ].each do |v, t|
          it "type: #{t}" do
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(v)            
          end
        end
      end

      context "Float64" do
        d = {'a' => 2.221e100, 'b' => Float::INFINITY}
        typeof_d = "{a: float64, b: float64}"

        [
          # 'float64' is the default dtype if there is no data at all.
          [[], "0 * float64"],
          [[[]], "1 * 0 * float64"],
          [[[], []], "2 * 0 * float64"],
          [[[[]], [[]]], "2 * 1 * 0 * float64"],
          [[[[]], [[], []]],
           "var(offsets=[0, 2]) * var(offsets=[0, 1, 3]) * var(offsets=[0, 0, 0, 0]) * float64"],

          [[0.0], "1 * float64"],
          [[0.0, 1.2], "2 * float64"],
          [[[0.0], [1.2]], "2 * 1 * float64"],

          [d, typeof_d],
          [[d] * 2, "2 * %s" % typeof_d],
          [[[d] * 2] * 10, "10 * 2 * #{typeof_d}"]
        ].each do |v, t|
          it "type: #{t}" do
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(v)            
          end
        end
      end

      context "Complex128" do
        d = {'a' => 3.123+10i, 'b' => Complex(Float::INFINITY, Float::INFINITY)}
        typeof_d = "{a: complex128, b: complex128}"

        [
          [[1+3e300i], "1 * complex128"],
          [[-2.2-5i, 1.2-10i], "2 * complex128"],
          [[-2.2-5i, 1.2-10i, nil], "3 * ?complex128"],
          [[[-1+3i], [-3+5i]], "2 * 1 * complex128"],

          [d, typeof_d],
          [[d] * 2, "2 * #{typeof_d}"],
          [[[d] * 2] * 10, "10 * 2 * #{typeof_d}"]
        ].each do |v, t|
          it "type: #{t}" do
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(v)            
          end
        end
      end

      context "Int64" do
        t = XND::T.new(1, -2, -3)
        typeof_t = "(int64, int64, int64)"

        [
          [[0], "1 * int64"],
          [[0, 1], "2 * int64"],
          [[[0], [1]], "2 * 1 * int64"],

          [t, typeof_t],
          [[t] * 2, "2 * #{typeof_t}"],
          [[[t] * 2] * 10, "10 * 2 * #{typeof_t}"]
        ].each do |v, t|
          it "type: #{t}" do  
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(XND::TypeInference.convert_xnd_t_to_ruby_array(v))
          end
        end
      end

      context "String" do
        t = XND::T.new("supererogatory", "exiguous")
        typeof_t = "(string, string)"

        [
          [["mov"], "1 * string"],
          [["mov", "$0"], "2 * string"],
          [[["cmp"], ["$0"]], "2 * 1 * string"],

          [t, typeof_t],
          [[t] * 2, "2 * %s" % typeof_t],
          [[[t] * 2] * 10, "10 * 2 * %s" % typeof_t]
        ].each do |v, t|
          it "type: #{t}" do
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(XND::TypeInference.convert_xnd_t_to_ruby_array(v))
          end
        end
      end

      context "Bytes" do
        t = XND::T.new("lagrange".b, "points".b)
        typeof_t = "(bytes, bytes)"

        [
          [["L1".b], "1 * bytes"],
          [["L2".b, "L3".b, "L4".b], "3 * bytes"],
          [[["L5".b], ["none".b]], "2 * 1 * bytes"],

          [t, typeof_t],
          [[t] * 2, "2 * %s" % typeof_t],
          [[[t] * 2] * 10, "10 * 2 * %s" % typeof_t]
        ].each do |v, t|
          it "type: {t}" do
            x = XND.new v
            
            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(XND::TypeInference.convert_xnd_t_to_ruby_array(v))
          end
        end
      end

      context "Optional" do
        [
          [nil, "?float64"],
          [[nil], "1 * ?float64"],
          [[nil, nil], "2 * ?float64"],
          [[nil, 10], "2 * ?int64"],
          [[nil, 'abc'.b], "2 * ?bytes"],
          [[nil, 'abc'], "2 * ?string"]
        ].each do |v, t|
          it "type: #{t}" do
            x = XND.new v

            expect(x.type).to eq(NDT.new(t))
            expect(x.value).to eq(v)
          end
        end

        [
          [nil, []],
          [[], nil],
          [nil, [10]],
          [[nil, [0, 1]], [[2, 3]]]
        ].each do |v|
          it "not implemented for value: #{v}" do 
            expect {
              XND.new v
            }.to raise_error(NotImplementedError)           
          end
        end
      end # context Optional
    end # context TypeInference
    
    context "FixedDim" do
      it "creates a fixed array" do
        o = XND.new([[1,2,3], [2,3,4]])
        expect(o.type).to eq(NDTypes.new("2 * 3 * int64"))
      end

      it "accepts a type for fixed array" do
        t = NDT.new("2 * 3 * int64")
        o = XND.new([[1,2,3], [2,3,4]], type: t)

        expect(o.type).to eq(t)
      end

      it "raises ArgumentError for type and input mismatch" do
        t = NDT.new "3 * 3 * int64"
        expect {
          XND.new([[1,2,3], [2,3,4]], type: t)
        }.to raise_error(ArgumentError)
      end

      it "raises ValueError for wrong input type in int64 array" do
        t = NDT.new "2 * 3 * int64"
        expect {
          XND.new([[1,2,"peep!"], [2,3,4]], type: t)
        }.to raise_error(TypeError)      
      end
    end

    context "VarDim" do
      
    end

    skip "FixedString" do
      it "creates FixedString utf16" do
        t = "2 * fixed_string(3, 'utf16')"
        v = ["\u1111\u2222\u3333", "\u1112\u2223\u3334"]
        x = XND.new v, type: t
        
        expect(x.value).to eq(v)
      end

      it "creates FixedString utf32 - figure a way to specify 32bit codepoints." do
        t = "2 * fixed_string(3, 'utf32')"
        v = ["\x00\x01\x11\x11\x00\x02\x22\x22\x00\x03\x33\x33".encode('UTF-32'),
             "\x00\x01\x11\x12\x00\x02\x22\x23\x00\x03\x33\x34".encode('UTF-32')]
        x = XND.new v, type: t
        
        expect(x.value).to eq(v)
      end
    end

    context "String" do
      it "creates new String array" do
        t = '2 * {a: complex128, b: string}'
        x = XND.new([{'a' => 2+3i, 'b' => "thisguy"},
                     {'a' => 1+4i, 'b' => "thatguy"}], type: t)

        expect(x[0]['b'].value).to eq("thisguy")
        expect(x[1]['b'].value).to eq("thatguy")
      end
    end # context String

    context "Bool" do
      it "from bool" do
        x = XND.new true, type: "bool"
        expect(x.value).to eq(true)

        x = XND.new false, type: "bool"
        expect(x.value).to eq(false)
      end

      it "from int" do
        x = XND.new 1, type: "bool"
        expect(x.value).to eq(true)

        x = XND.new 0, type: "bool"
        expect(x.value).to eq(false)
      end

      it "from object" do
        x = XND.new [1,2,3], type: "bool"
        expect(x.value).to eq(true)

        x = XND.new nil, type: "?bool"
        expect(x.value).to eq(nil)

        expect {
          XND.new nil, type: "bool"
        }.to raise_error(TypeError)
      end

      skip "tests broken input - how can this be done in Ruby?" do
       n 
      end
    end # context Bool

    context "Signed" do
      [8, 16, 32, 64].each do |n|
        it "tests bounds for n=#{n}" do
          t = "int#{n}"

          v = -2**(n-1)
          x = XND.new(v, type: t)
          expect(x.value).to eq(v)
          expect { XND.new v-1, type: t }.to raise_error(RangeError)

          v = 2**(n-1) - 1
          x = XND.new(v, type: t)
          expect(x.value).to eq(v)
          expect { XND.new v+1, type: t }.to raise_error(RangeError)
        end
      end
    end # context Signed

    context "Unsigned" do
      [8, 16, 32, 64].each do |n|
        it "tests bounds v-1. n=#{n}" do
          t = "uint#{n}"

          v = 0
          x = XND.new v, type: t
          expect(x.value).to eq(v)
          expect { XND.new v-1, type: t }.to raise_error(RangeError)
        end
        
        it "tests bounds v+1. n=#{n}" do
          t = "uint#{n}"
          
          v = 2**n - 2
          x = XND.new v, type: t
          expect(x.value).to eq(v)
          expect { XND.new v+2, type: t }.to raise_error(RangeError)
        end
      end
    end # context Unsigned

    context "Float32" do
      it "tests inf bounds" do
        inf = Float("0x1.ffffffp+127")

        expect { XND.new(inf, type: "float32") }.to raise_error(RangeError)
        expect { XND.new(-inf, type: "float32") }.to raise_error(RangeError)
      end

      it "tests denorm_min bounds" do
        denorm_min = Float("0x1p-149")

        x = XND.new denorm_min, type: "float32"
        expect(x.value).to eq(denorm_min)
      end

      it "tests lowest bounds" do
        lowest = Float("-0x1.fffffep+127")
        
        x = XND.new lowest, type: "float32"
        expect(x.value.nan?).to eq(lowest.nan?)        
      end

      it "tests max bounds" do
        max = Float("0x1.fffffep+127")

        x = XND.new max, type: "float32"
        expect(x.value).to eq(max)
      end

      it "tests special values" do
        x = XND.new Float::INFINITY, type: "float32"
        expect(x.value.infinite?).to eq(1)

        x = XND.new Float::NAN, type: "float32"
        expect(x.value.nan?).to eq(true)
      end
    end # context Float32

    context "Float64" do
      it "tests bounds" do
        denorm_min = Float("0x0.0000000000001p-1022")
        lowest = Float("-0x1.fffffffffffffp+1023")
        max = Float("0x1.fffffffffffffp+1023")

        x = XND.new denorm_min, type: "float64"
        expect(x.value).to eq(denorm_min)

        x = XND.new lowest, type: "float64"
        expect(x.value).to eq(lowest)

        x = XND.new max, type: "float64"
        expect(x.value).to eq(max)
      end

      it "tests special values" do
        x = XND.new Float::INFINITY, type: "float64"
        expect(x.value.infinite?).to eq(1)

        x = XND.new Float::NAN, type: "float64"
        expect(x.value.nan?).to eq(true)
      end
    end # context Float64

    context "Complex64" do
      it "tests bounds" do
        denorm_min = Float("0x1p-149")
        lowest = Float("-0x1.fffffep+127")
        max = Float("0x1.fffffep+127")
        inf = Float("0x1.ffffffp+127")

        v = Complex(denorm_min, denorm_min)
        x = XND.new v, type: "complex64"
        expect(x.value).to eq(v)

        v = Complex(lowest, lowest)
        x = XND.new v, type: "complex64"
        expect(x.value).to eq(v)

        v = Complex(max, max)
        x = XND.new v, type: "complex64"
        expect(x.value).to eq(v)

        v = Complex(inf, inf)
        expect { XND.new v, type: "complex64" }.to raise_error(RangeError)

        v = Complex(-inf, -inf)
        expect { XND.new v, type: "complex64" }.to raise_error(RangeError)
      end

      it "tests special values" do
        x = XND.new Complex(Float::INFINITY, 0), type: "complex64"
        expect(x.value.real.infinite?).to eq(1)
        expect(x.value.imag).to eq(0.0)

        x = XND.new Complex(Float::NAN, 0), type: "complex64"
        expect(x.value.real.nan?).to eq(true)
        expect(x.value.imag).to eq(0.0)
      end
    end # context Complex64

    context "Complex128" do
      it "tests bounds" do
        denorm_min = Float("0x0.0000000000001p-1022")
        lowest = Float("-0x1.fffffffffffffp+1023")
        max = Float("0x1.fffffffffffffp+1023")

        v = Complex(denorm_min, denorm_min)
        x = XND.new v, type: "complex128"
        expect(x.value).to eq(v)

        v = Complex(lowest, lowest)
        x = XND.new v, type: "complex128"
        expect(x.value).to eq(v)

        v = Complex(max, max)
        x = XND.new v, type: "complex128"
        expect(x.value).to eq(v)
      end

      it "tests special values" do
        x = XND.new Complex(Float::INFINITY), type: "complex128"

        expect(x.value.real.infinite?).to eq(1)
        expect(x.value.imag).to eq(0.0)

        x = XND.new Complex(Float::NAN), type: "complex128"

        expect(x.value.real.nan?).to eq(true)
        expect(x.value.imag).to eq(0.0)
      end
    end # context Complex128
  end # context .new

  context ".empty" do

    context "Constr" do
      DTYPE_EMPTY_TEST_CASES.each do |v, s|
        [
          [v, "SomeConstr(#{s})"],
          [v, "Just(Some(#{s}))"],

          [[v] * 0, "Some(0 * #{s})"],
          [[v] * 1, "Some(1 * #{s})"],
          [[v] * 3, "Maybe(3 * #{s})"]
        ].each do |vv, ss|
          it "type: #{ss}" do
            t = NDT.new ss
            x = XND.empty ss
            
            expect(x.type).to eq(t)
            expect(x.value).to eq(vv)
            if vv == 0
              expect {
                x.size
              }.to raise_error(NoMethodError)
            end
          end
        end
      end

      it "returns empty view" do
        # If a constr is a dtype but contains an array itself, indexing should
        # return a view and not a Python value.
        inner = [[""] * 5] * 4
        x = XND.empty("2 * 3 * InnerArray(4 * 5 * string)")

        y = x[1][2]
        expect(y.is_a?(XND)).to eq(true)
        expect(y.value).to eq(inner)

        y = x[1, 2]
        expect(y.is_a?(XND)).to eq(true)
        expect(y.value).to eq(inner)
      end
    end

    context "Nominal" do
      c = 0
      DTYPE_EMPTY_TEST_CASES.each do |v, s|
        NDT.typedef "some#{c}", s
        NDT.typedef "just#{c}", "some#{c}"
        
        [
          [v, "some#{c}"],
          [v, "just#{c}"]
        ].each do |vv, ss|
          it "type: #{ss}" do
            t = NDT.new ss
            x = XND.empty ss
            
            expect(x.type).to eq(t)
            expect(x.value).to eq(vv)
            if vv == 0
              expect {
                x.size
              }.to raise_error(NoMethodError)
            end
          end          
        end

        c += 1
      end

      it "returns empty view" do
        # If a typedef is a dtype but contains an array itself, indexing should
        # return a view and not a Python value.
        NDT.typedef("inner_array", "4 * 5 * string")
        inner = [[""] * 5] * 4
        x = XND.empty("2 * 3 * inner_array")

        y = x[1][2]
        expect(y.is_a?(XND)).to eq(true)
        expect(y.value).to eq(inner)

        y = x[1, 2]
        expect(y.is_a?(XND)).to eq(true)
        expect(y.value).to eq(inner)
      end
    end

    context "Categorical" do
      # Categorical values are stored as indices into the type's categories.
      # Since empty xnd objects are initialized to zero, the value of an
      # empty categorical entry is always the value of the first category.
      # This is safe, since categorical types must have at least one entry.
      r = {'a' => "", 'b' => 1.2}
      rt = "{a: string, b: categorical(1.2, 10.0, NA)}"

      [
        ["January", "categorical('January')"],
        [[nil], "(categorical(NA, 'January', 'August'))"],
        [[[1.2] * 2] * 10, "10 * 2 * categorical(1.2, 10.0, NA)"],
        [[[100] * 2] * 10, "10 * 2 * categorical(100, 'mixed')"],
        [[[r] * 2] * 10, "10 * 2 * #{rt}"],
        [[[r] * 2, [r] * 5, [r] * 3], "var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * #{rt}"]
      ].each do |v, s|

        it "type: #{s}" do
          t = NDT.new s
          x = XND.empty s

          expect(x.type).to eq(t)
          expect(x.value).to eq(v)     
        end
      end
    end

    context "FixedString" do
      it "tests kind of string" do
        expect {
          XND.empty "FixedString"
        }.to raise_error(ValueError)
      end

      [
        ["fixed_string(1)", ""],
        ["fixed_string(3)", "" * 3],
        ["fixed_string(1, 'ascii')", ""],
        ["fixed_string(3, 'utf8')", "" * 3],
        ["fixed_string(3, 'utf16')", "" * 3],
        ["fixed_string(3, 'utf32')", "" * 3],
        ["2 * fixed_string(3, 'utf32')", ["" * 3] * 2],
      ].each do |s, v|
        
        it "type: #{s}" do
          t = NDT.new s
          x = XND.empty s

          expect(x.type).to eq(t)
          expect(x.value).to eq(v)
        end
      end
    end

    context "FixedBytes" do
      r = {'a' => "\x00".b * 3, 'b' => "\x00".b * 10}

      [
        ["\x00".b, 'fixed_bytes(size=1)'],
        ["\x00".b * 100, 'fixed_bytes(size=100)'],
        ["\x00".b * 4, 'fixed_bytes(size=4, align=2)'],
        ["\x00".b * 128, 'fixed_bytes(size=128, align=16)'],
        [r, '{a: fixed_bytes(size=3), b: fixed_bytes(size=10)}'],
        [[[r] * 3] * 2, '2 * 3 * {a: fixed_bytes(size=3), b: fixed_bytes(size=10)}']
      ].each do |v, s|
        it "type: #{s}" do
          t = NDT.new s
          x = XND.empty s

          expect(x.type).to eq(t)
          expect(x.value).to eq(v)          
        end
      end
    end

    context "String" do
      [
        'string',
        '(string)',
        '10 * 2 * string',
        '10 * 2 * (string, string)',
        '10 * 2 * {a: string, b: string}',
        'var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * {a: string, b: string}'
      ].each do |s|

        it "type: #{s}" do
          t = NDT.new s
          x = XND.empty s
          expect(x.type).to eq(t)          
        end
      end

      it "tests for single value" do
        t = NDT.new "string"
        x = XND.empty t

        expect(x.type).to eq(t)
        expect(x.value).to eq('')
      end

      xit "tests for multiple values" do
        t = NDT.new "10 * string"
        x = XND.empty t

        expect(x.type).to eq(t)
        0.upto(10) do |i| 
          expect(x[i]).to eq(XND.new(['']))
        end
      end
    end

    context "Bytes" do
      r = { 'a' => "".b, 'b' => "".b }
      
      [
        [''.b, 'bytes(align=16)'],
        [[''.b], '(bytes(align=32))'],
        [[[''.b] * 2] * 3, '3 * 2 * bytes'],
        [[[[''.b, ''.b]] * 2] * 10, '10 * 2 * (bytes, bytes)'],
        [[[r] * 2] * 10, '10 * 2 * {a: bytes(align=32), b: bytes(align=1)}'],
        [[[r] * 2] * 10, '10 * 2 * {a: bytes(align=1), b: bytes(align=32)}'],
        [[[r] * 2, [r] * 5, [r] * 3], 'var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * {a: bytes(align=32), b: bytes}']
      ].each do |v, s|
        it "type: #{s}" do
          t = NDT.new s
          x = XND.empty t

          expect(x.type).to eq(t)
          expect(x.value).to eq(v)
        end
      end
    end

    context "Char" do
      it "raises ValueError" do
        expect {
          XND.empty "char('utf8')"
        }.to raise_error(ValueError)
      end
    end

    context "SignedKind" do
      it "raises ValueError" do
        expect {
          XND.empty "Signed"
        }.to raise_error(ValueError)
      end
    end

    context "UnsignedKind" do
      it "raises ValueError" do
        expect {
          XND.empty "Unsigned"
        }.to raise_error(ValueError)
      end
    end

    context "FloatKind" do
      it "raises ValueError" do
        expect {
          XND.empty "Float"
        }.to raise_error(ValueError)
      end
    end

    context "ComplexKind" do
      it "raises ValueError" do
        expect {
          XND.empty "Complex"
        }.to raise_error(ValueError)
      end
    end

    context "FixedBytesKind" do
      it "raises ValueError" do
        expect {
          XND.empty "FixedBytes"
        }.to raise_error(ValueError)
      end
    end

    context "Primitive" do
      empty_test_cases.each do |value, type_string|
        PRIMITIVE.each do |p|
          ts = type_string % p

          it "type: #{ts}" do
            x = XND.empty ts

            expect(x.value).to eq(value)
            expect(x.type).to eq(NDT.new(ts))
          end
        end
      end

      empty_test_cases(false).each do |value, type_string|
        BOOL_PRIMITIVE.each do |p|
          ts = type_string % p

          it "type: #{ts}" do
            x = XND.empty ts

            expect(x.value).to eq(value)
            expect(x.type).to eq(NDT.new(ts))
          end
        end
      end
    end

    context "TypeVar" do
      [
        "T",
        "2 * 10 * T",
        "{a: 2 * 10 * T, b: bytes}"
      ].each do |ts|
        it "#{ts} raises ValueError" do
          expect {
            XND.empty ts
          }.to raise_error(ValueError)
        end        
      end
    end
  end

  context ".from_buffer" do
    it "can import data from nmatrix objects" do
      
    end

    it "can import data from narray objects" do
      
    end
  end

  context "#[]" do
    context "FixedDim" do
      it "returns single number slice for 1D array/1 number" do
        xnd = XND.new([1,2,3,4])
        expect(xnd[1]).to eq(XND.new(2))
      end

      it "returns single number slice for 2D array and 2 indices" do
        xnd = XND.new([[1,2,3], [4,5,6]])
        expect(xnd[0,0]).to eq(XND.new(1)) 
      end

      it "returns row for single index in 2D array" do
        x = XND.new [[1,2,3], [4,5,6], [7,8,9]]
        expect(x[1]).to eq(XND.new([4,5,6]))
      end

      it "returns single column in 2D array" do
        x = XND.new [[1,2,3], [4,5,6], [7,8,9]]
        expect(x[0..Float::INFINITY, 0]).to eq(XND.new([1,4,7]))
      end

      it "returns the entire array" do
        x = XND.new [[1,2,3], [4,5,6], [7,8,9]]
        expect(x[0..Float::INFINITY]).to eq(x)
      end

      [
        [
          [
            [11.12-2.3i, -1222+20e8i],
            [Complex(Float::INFINITY, Float::INFINITY), -0.00002i],
            [0.201+1i, -1+1e301i]
          ], "3 * 2 * complex128"],
        [
          [
            [11.12-2.3i, nil],
            [Complex(Float::INFINITY, Float::INFINITY), nil],
            [0.201+1i, -1+1e301i]
          ], "3 * 2 * ?complex128"]
      ].each do |v, s|
        context "type: #{s}" do
          before do
            @arr = v
            @t = NDT.new s
            @x = XND.new v, type: @t
          end

          it "check values" do
            expect(@x.to_a).to eq(@arr.to_a)
          end

          0.upto(2) do |i|
            it "value: i= #{i}" do
              expect(@x[i].to_a).to eq(@arr[i])
            end
          end

          3.times do |i|
            2.times do |k|
              it "value: i=#{i}. k=#{k}" do
                expect(@x[i][k].value).to eq(@arr[i][k])
                expect(@x[i, k].value).to eq(@arr[i][k])
              end
            end
          end

          it "tests full slices" do
            expect(@x[INF].value).to eq(@arr)
          end

          ((-3...4).to_a + [Float::INFINITY]).each do |start|
            ((-3...4).to_a + [Float::INFINITY]).each do |stop|
              [true, false].each do |exclude_end|
                # FIXME: add step count when ruby supports it.
                arr_s = get_inf_or_normal_range start, stop, exclude_end

                it "Range[#{start}, #{stop}#{exclude_end ? ')' : ']'}" do
                  r = Range.new(start, stop, exclude_end)
                  expect(@x[r].value).to eq(@arr[arr_s])
                end
              end
            end
          end

          it "tests single rows" do
            expect(@x[INF, 0].value).to eq(@arr.transpose[0])
            expect(@x[INF, 1].value).to eq(@arr.transpose[1])
          end
        end
      end
    end

    context "Constr" do
      before do
        # FIXME: If a constr is a dtype but contains an array itself, indexing through
        # the constructor should work transparently. However, for now it returns
        # an XND object, however this will likely change in the future.
        @inner = [['a', 'b', 'c', 'd', 'e'],
                  ['f', 'g', 'h', 'i', 'j'],
                  ['k', 'l', 'm', 'n', 'o'],
                  ['p', 'q', 'r', 's', 't']]
        @v = [[@inner] * 3] * 2
        @x = XND.new(@v, type: "2 * 3 * InnerArray(4 * 5 * string)")
      end

      (0).upto(1) do |i|
        (0).upto(2) do |j|
          (0).upto(3) do |k|
            (0).upto(4) do |l|
              it "slice: i=#{i} j=#{j} k=#{k} l=#{l}" do
                
                expect(@x[i][j][k][l].value).to eq(@inner[k][l])
                expect(@x[i, j, k, l].value).to eq(@inner[k][l])
              end
            end
          end
        end
      end
    end # context Constr

    context "Nominal" do
      before do
        # FIXME: If a typedef is a dtype but contains an array itself, indexing through
        # the constructor should work transparently. However, for now it returns an XND
        # object, however this will likely change in the future.
        NDT.typedef("inner", "4 * 5 * string")
        @inner = [['a', 'b', 'c', 'd', 'e'],
                  ['f', 'g', 'h', 'i', 'j'],
                  ['k', 'l', 'm', 'n', 'o'],
                  ['p', 'q', 'r', 's', 't']]
        @v = [[@inner] * 3] * 2
        @x = XND.new(@v, type: "2 * 3 * inner")        
      end


      (0).upto(1) do |i|
        (0).upto(2) do |j|
          (0).upto(3) do |k|
            (0).upto(4) do |l|
              it "slice: i=#{i} j=#{j} k=#{k} l=#{l}" do
                expect(@x[i][j][k][l].value).to eq(@inner[k][l])
                expect(@x[i, j, k, l].value).to eq(@inner[k][l])
              end
            end
          end
        end
      end      
    end # context Nominal
  end # context #[]

  context "#[]=" do
    context "Ref" do
      # TODO: see line 1341 of pycode.
    end

    context "Constr" do
      # TODO: see line 1484
    end

    context "Nominal" do
      # TODO: see line 1637
    end

    context "Categorical" do
      before do
        @s = "2 * categorical(NA, 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')"
        @x = XND.new [nil, nil], type: @s
      end
      
      it "assigns regular data" do
        @x[0] = "August"
        @x[1] = "December"

        expect(@x.value).to eq(["August", "December"])
      end

      it "assigns nil" do
        @x[0] = nil
        @x[1] = "December"

        expect(@x.value).to eq([nil, "December"])
      end
    end # context Categorical

    skip "FixedString - need to figure out UTF-32 in Ruby." do
      it "assigns a fixed string" do
        t = "2 * fixed_string(3, 'utf32')"
        v = ["\U00011111\U00022222\U00033333", "\U00011112\U00022223\U00033334"]
        x = XND.new(v, type: t)

        x[0] = "a"
        expect(x.value).to eq(["a", "\U00011112\U00022223\U00033334"])

        x[0] = "a\x00\x00"
        expect(x.value).to eq(["a", "\U00011112\U00022223\U00033334"])

        x[1] = "b\x00c"
        expect(x.value).to eq(["a", "b\x00c"])
      end
    end # context FixedString

    context "FixedBytes" do
      it "assign single element" do
        t = "2 * fixed_bytes(size=3, align=1)"
        v = ["abc".b, "123".b]
        x = XND.new(v, type: t)
        
        x[0] = "xyz".b
        expect(x.value).to eq(["xyz".b, "123".b])
      end
    end # context FixedString

    context "String" do
      it "assigns single" do
        t = '2 * {a: complex128, b: string}'
        x = XND.new([{ 'a' => 2+3i, 'b' => "thisguy"},
                     { 'a' => 1+4i, 'b' => "thatguy" }], type: t)

        x[0] = { 'a' => 220i, 'b' => 'y'}
        x[1] = { 'a' => -12i, 'b' => 'z'}

        expect(x.value).to eq([
                                { 'a' => 220i, 'b' => 'y' },
                                { 'a' => -12i, 'b' => 'z' }
                              ])
      end
    end # context String

    context "Bytes" do
      it "assigns bytes by tuple" do
        t = "2 * SomeByteArray(3 * bytes)"
        inner = ["a".b, "b".b, "c".b]
        v = [inner] * 2
        x = XND.new v, type: t

        2.times do |i|
          3.times do |k|
            x[i, k] = inner[k] = ['x'.chr.ord + k].pack("C")
          end
        end

        expect(x.value).to eq(v)
      end
    end # context Bytes
  end # context #[]=
  
  context "#strict_equal" do
    context "Ref" do
    end # context Ref

    context "Constr" do
      it "simple tests" do
        x = XND.new [1,2,3,4], type: "A(4 * float32)"

        expect_strict_equal x, XND.new([1,2,3,4], type: "A(4 * float32)")

        expect_strict_unequal x, XND.new([1,2,3,4], type: "B(4 * float32)")
        expect_strict_unequal x, XND.new([1,2,3,4,5], type: "A(5 * float32)")
        expect_strict_unequal x, XND.new([1,2,3], type: "A(3 * float32)")
        expect_strict_unequal x, XND.new([1,2,3,4,55], type: "A(5 * float32)")
      end

      it "corner cases and dtypes" do
        EQUAL_TEST_CASES.each do |struct|
          v = struct.v
          t = struct.t
          u = struct.u

          [
            [[v] * 0, "A(0 * #{t})", "A(0 * #{u})"],
            [[v] * 0, "A(B(0 * #{t}))", "A(B(0 * #{u}))"],
            [[v] * 0, "A(B(C(0 * #{t})))", "A(B(C(0 * #{u})))"]
          ].each do |vv, tt, uu|
            ttt = NDT.new tt
            uuu = NDT.new tt

            x = XND.new vv, type: ttt

            y = XND.new vv, type: ttt
            expect_strict_equal x, y

            y = XND.new vv, type: uuu
            expect_strict_equal x, y            
          end
        end
      end

      it "more dtypes" do
        EQUAL_TEST_CASES.each do |struct|
          v = struct.v
          t = struct.t
          u = struct.u
          w = struct.w
          eq = struct.eq

          [
            [v, "A(#{t})", "A(#{u})", []],
            [v, "A(B(#{t}))", "A(B(#{u}))", []],
            [v, "A(B(C(#{t})))", "A(B(C(#{u})))", []],
            [[v] * 1, "A(1 * #{t})", "A(1 * #{u})", 0],
            [[v] * 3, "A(3 * #{t})", "A(3 * #{u})", 2]
          ].each do |vv, tt, uu, indices|
            ttt = NDT.new tt
            uuu = NDT.new tt

            x = XND.new vv, type: ttt

            y = XND.new vv, type: ttt
            if eq
              expect_strict_equal x, y
            else
              expect_strict_unequal x, y
            end

            unless u.nil?
              y = XND.new vv, type: uuu
              if eq
                expect_strict_equal x, y
              else
                expect_strict_unequal x, y
              end
            end

            unless w.nil?
              y = XND.new vv, type: ttt
              y[indices] = w

              expect_strict_unequal x, y
            end
          end
        end
      end
    end # context Constr

    context "Nominal" do
      it "simple tests" do
        NDT.typedef "some1000", "4 * float32"
        NDT.typedef "some1001", "4 * float32"

        x = XND.new([1,2,3,4], type: "some1000")

        expect_strict_equal x, XND.new([1,2,3,4], type: "some1000")

        expect_strict_unequal x, XND.new([1,2,3,4], type: "some1001")
        expect_strict_unequal x, XND.new([1,2,3,5], type: "some1000")
      end
    end # context Nominal

    context "Categorical" do
      it "simple tests" do
        t = "3 * categorical(NA, 'January', 'August')"
        x = XND.new ['August', 'January', 'January'], type: t

        y = XND.new ['August', 'January', 'January'], type: t
        expect_strict_equal x, y

        y = XND.new ['August', 'January', 'August'], type: t
        expect_strict_unequal x, y

        x = XND.new ['August', nil, 'August'], type: t
        y = XND.new ['August', nil, 'August'], type: t

        expect_strict_unequal x, y
      end
    end # context Categorical

    context "FixedString" do
      it "compare" do
        [
          ["fixed_string(1)", "", "x"],
          ["fixed_string(3)", "y" * 3, "yyz"],
          ["fixed_string(1, 'ascii')", "a".b, "b".b],
          ["fixed_string(3, 'utf8')", "a" * 3, "abc"],
          #["fixed_string(3, 'utf16')", "\u1234" * 3, "\u1234\u1235\u1234"],
          #["fixed_string(3, 'utf32')", "\U00001234" * 3, "\U00001234\U00001234\U00001235"]
        ].each do |t, v, w| 
          x = XND.new v, type: t
          y = XND.new v, type: t

          expect_strict_equal x, y

          y[[]] = w

          expect_strict_unequal x, y
        end        
      end
    end # context FixedString

    context "FixedBytes" do
      it "simple test" do 
        [
          ["a".b, "fixed_bytes(size=1)", "b".b],
          ["a".b * 100, "fixed_bytes(size=100)", "a".b * 99 + "b".b],
          ["a".b * 4, "fixed_bytes(size=4, align=2)", "a".b * 2 + "b".b]
        ].each do |v, t, w|
          x = XND.new v, type: t
          y = XND.new v, type: t

          expect_strict_equal x, y

          y[[]] = w
          expect_strict_unequal x, y
        end
      end

      it "align" do
        x = XND.new("a".b * 128, type: "fixed_bytes(size=128, align=16)")
        y = XND.new("a".b * 128, type: "fixed_bytes(size=128, align=16)")
        expect_strict_equal x, y
      end
    end # context FixedBytes

    context "String" do
      it "compare" do
        x = XND.new "abc"

        expect_strict_equal x, XND.new("abc")
        expect_strict_equal x, XND.new("abc\0\0")

        expect_strict_unequal x, XND.new("acb")
      end
    end # context String

    context "Bool" do
      it "compare" do
        expect_strict_equal XND.new(true), XND.new(true)
        expect_strict_equal XND.new(false), XND.new(false)
        expect_strict_unequal XND.new(true), XND.new(false)
        expect_strict_unequal XND.new(false), XND.new(true)
      end
    end # context Bool

    context "Signed" do
      it "compare" do
        ["int8", "int16", "int32", "int64"].each do |t|
          expect_strict_equal XND.new(-10, type: t), XND.new(-10, type: t)
          expect_strict_unequal XND.new(-10, type: t), XND.new(100, type: t)
        end
      end
    end # context Signed

    context "Unsigned" do
      it "compare" do
        ["uint8", "uint16", "uint32", "uint64"].each do |t|
          expect_strict_equal XND.new(10, type: t), XND.new(10, type: t)
          expect_strict_unequal XND.new(10, type: t), XND.new(100, type: t)
        end
      end
    end # context Unsigned

    context "Float32" do
      it "compare" do
        expect_strict_equal XND.new(1.2e7, type: "float32"),
                            XND.new(1.2e7, type: "float32")
        expect_strict_equal XND.new(Float::INFINITY, type: "float32"),
                            XND.new(Float::INFINITY, type: "float32")
        expect_strict_equal XND.new(-Float::INFINITY, type: "float32"),
                            XND.new(-Float::INFINITY, type: "float32")

        expect_strict_unequal XND.new(1.2e7, type: "float32"),
                              XND.new(-1.2e7, type: "float32")
        expect_strict_unequal XND.new(Float::INFINITY, type: "float32"),
                              XND.new(-Float::INFINITY, type: "float32")
        expect_strict_unequal XND.new(-Float::INFINITY, type: "float32"),
                              XND.new(Float::INFINITY, type: "float32")
        expect_strict_unequal XND.new(Float::NAN, type: "float32"),
                              XND.new(Float::NAN, type: "float32")
      end
    end # context Float32

    context "Float64" do
      it "compare" do
        expect_strict_equal XND.new(1.2e7, type: "float64"),
                            XND.new(1.2e7, type: "float64")
        expect_strict_equal XND.new(Float::INFINITY, type: "float64"),
                            XND.new(Float::INFINITY, type: "float64")
        expect_strict_equal XND.new(-Float::INFINITY, type: "float64"),
                            XND.new(-Float::INFINITY, type: "float64")

        expect_strict_unequal XND.new(1.2e7, type: "float64"),
                              XND.new(-1.2e7, type: "float64")
        expect_strict_unequal XND.new(Float::INFINITY, type: "float64"),
                              XND.new(-Float::INFINITY, type: "float64")
        expect_strict_unequal XND.new(-Float::INFINITY, type: "float64"),
                              XND.new(Float::INFINITY, type: "float64")
        expect_strict_unequal XND.new(Float::NAN, type: "float64"),
                              XND.new(Float::NAN, type: "float64")              
      end
    end # context Float64

    context "Complex64" do
      it "compare" do
        t = "complex64"

        inf = Float("0x1.ffffffp+127")
        denorm_min = Float("0x1p-149")
        lowest = Float("-0x1.fffffep+127")
        max = Float("0x1.fffffep+127")

        c = [denorm_min, lowest, max, Float::INFINITY, -Float::INFINITY, Float::NAN]

        c.each do |r|
          c.each do |s|
            c.each do |i|
              c.each do |j|
                x = XND.new Complex(r, i), type: t
                y = XND.new Complex(s, j), type: t

                if r == s && i == j
                  expect_strict_equal x, y
                else
                  expect_strict_unequal x, y
                end
              end
            end
          end
        end
      end
    end # context Complex64

    context "Complex128" do
      it "compare" do
        t = "complex128"

        denorm_min = Float("0x0.0000000000001p-1022")
        lowest = Float("-0x1.fffffffffffffp+1023")
        max = Float("0x1.fffffffffffffp+1023")

        c = [denorm_min, lowest, max, Float::INFINITY, -Float::INFINITY, Float::NAN]

        c.each do |r|
          c.each do |s|
            c.each do |i|
              c.each do |j|
                x = XND.new Complex(r, i), type: t
                y = XND.new Complex(s, j), type: t

                if r == s && i == j
                  expect_strict_equal x, y
                else
                  expect_strict_unequal x, y
                end
              end
            end
          end
        end        
      end
    end # context complex128
  end # Context #strict_equal

  context "#match" do
    context "VarDim" do
      
    end # context VarDim
  end # context #match

  context "#to_a" do
    context "FixedDim" do
      it "returns simple array" do
        x = XND.new [1,2,3,4]

        expect(x.to_a).to eq([1,2,3,4])
      end

      it "returns multi-dim array" do
        x = XND.new [[1,2,3], [4,5,6]]

        expect(x.to_a).to eq([[1,2,3], [4,5,6]])
      end      
    end
  end # context to_a

  context "#type" do
    it "returns the type of the XND array" do
      x = XND.new [[1,2,3], [4,5,6]], type: NDT.new("2 * 3 * int64")

      expect(x.type).to eq(NDT.new("2 * 3 * int64"))
    end
  end # context type

  context "#to_s" do
    it "returns String representation" do
      
    end
  end # context to_s

  context "#size" do
    context "FixedDim" do
      it "returns the size of the XND array" do
        x = XND.new [1,2,3,4,5]
        expect(x.size).to eq(5)
      end
    end
    
    context "Bool" do
      it "raises error" do
        x = XND.new true, type: "bool"
        expect {
          x.size
        }.to raise_error(NoMethodError)
      end
    end

    context "Signed" do
      it "raises error" do
        x = XND.new 10, type: "int16"
        expect { x.size }.to raise_error(NoMethodError)
      end
    end

    context "Unsigned" do
      it "raises error" do
        x = XND.new 10, type: "uint64"
        expect { x.size }.to raise_error(NoMethodError)
      end
    end
  end # context #size

  context "#each" do
    context "FixedDim" do
      it "iterates over all elements" do
        x = XND.new [1,2,3,4,5], type: "5 * int64"
        sum = 0
        x.each do |a|
          sum += x
        end

        expect(sum).to eq(15)
      end
    end
  end # context #each
end
