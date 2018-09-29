yrequire 'test_helper'

class TestModule < Minitest::Test
  def test_module
    test_cases = [
      "Foo:: 2 * 3 * ?int64",
      "Foo:: 10 * 2 * ?string",
      "Bar:: !10 * 2 * {a: !2 * ?int64}",
      "Quux:: {a: string, b: ?bytes}"
    ]

    test_cases.each do |s|
      assert_raises(ValueError) { XND.empty(s) }
    end
  end
end # class TestModule

class TestFunction < Minitest::Test
  def test_function
    test_cases = [
      "(2 * 3 * ?int64, complex128) -> (T, T)",
      "(2 * 3 * ?int64, {a: float64, b: bytes}) -> bytes",
    ]

    test_cases.each do |s|
      assert_raises(ValueError) { XND.empty(s) }
    end
  end
end # class TestFunction

class TestVoid < Minitest::Test
  def test_void
    assert_raises(ValueError) { XND.empty("void") }
    assert_raises(ValueError) { XND.empty("10 * 2 * void") }
  end
end # class TestVoid

class TestAny < Minitest::Test
  def test_any
    test_cases = [
      "Any",
      "10 * 2 * Any",
      "10 * N * int64",
      "{a: string, b: Any}"
    ]

    test_cases.each do |s|
      assert_raises(ValueError) { XND.empty(s) }
    end
  end
end # class TestAny

class TestFixedDim < Minitest::Test
  def test_fixed_dim_empty
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [[v] * 0, "0 * #{s}" ],
        [[v] * 1, "1 * #{s}" ],
        [[v] * 2, "2 * #{s}" ],
        [[v] * 1000, "1000 * #{s}" ],
        
        [[[v] * 0] * 0, "0 * 0 * #{s}" ],
        [[[v] * 1] * 0, "0 * 1 * #{s}" ],
        [[[v] * 0] * 1, "1 * 0 * #{s}" ],
        
        [[[v] * 1] * 1, "1 * 1 * #{s}" ],
        [[[v] * 2] * 1, "1 * 2 * #{s}" ],
        [[[v] * 1] * 2, "2 * 1 * #{s}" ],
        [[[v] * 2] * 2, "2 * 2 * #{s}" ],
        [[[v] * 3] * 2, "2 * 3 * #{s}" ],
        [[[v] * 2] * 3, "3 * 2 * #{s}" ],
        [[[v] * 40] *3 , "3 * 40 * #{s}" ]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss

        assert_equal t, x.type
        assert_equal vv, x.value
        assert_equal vv.size, x.size
      end
    end
  end

  def test_overflow
    assert_raises(ValueError) { XND.empty "2147483648 * 2147483648 * 2 * uint8" }
  end

  def test_equality
    x = XND.new [1,2,3,4]

    assert_strict_equal x, XND.new([1,2,3,4])

    # different shape and/or data.
    assert_strict_unequal x, XND.new([1,2,3,5])
    assert_strict_unequal x, XND.new([1,2,3,100])
    assert_strict_unequal x, XND.new([4,2,3,4,5])

    # different shape.
    assert_strict_unequal x, XND.new([1,2,3])
    assert_strict_unequal x, XND.new([[1,2,3,4]])
    assert_strict_unequal x, XND.new([[1,2], [3,4]])

    # tests simple multidim array
    x = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    y = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])

    assert_strict_equal x, y

    # C <-> Fortran.
    x = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    y = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type: "!4 * 3 * int64")

    assert_strict_equal x, y

    # slices
    x = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
    y = XND.new([1,2,3])
    assert_strict_equal x[0], y

    y = XND.new [1,4,7,10]
    assert_strict_equal x[0..Float::INFINITY,0], y

    # test corner cases and many dtypes.
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      
      [
        [[v] * 0, "0 * #{t}", "0 * #{u}"],
        [[[v] * 0] * 0, "0 * 0 * #{t}", "0 * 0 * #{u}"],
        [[[v] * 1] * 0, "0 * 1 * #{t}", "0 * 1 * #{u}"],
        [[[v] * 0] * 1, "1 * 0 * #{t}", "1 * 0 * #{u}"]
      ].each do |vv, tt, uu|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        y = XND.new vv, type: uuu
        assert_strict_equal x, y
      end
    end # EQUAL_TEST_CASES.each

    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      w = struct.w
      eq = struct.eq
      
      [
        [[v] * 1, "1 * #{t}", "1 * #{u}", [0]],
        [[v] * 2, "2 * #{t}", "2 * #{u}", [1]],
        [[v] * 1000, "1000 * #{t}", "1000 * #{u}", [961]],

        [[[v] * 1] * 1, "1 * 1 * #{t}", "1 * 1 * #{u}", [0, 0]],
        [[[v] * 2] * 1, "1 * 2 * #{t}", "1 * 2 * #{u}", [0, 1]],
        [[[v] * 1] * 2, "2 * 1 * #{t}", "2 * 1 * #{u}", [1, 0]],
        [[[v] * 2] * 2, "2 * 2 * #{t}", "2 * 2 * #{u}", [1, 1]],
        [[[v] * 3] * 2, "2 * 3 * #{t}", "2 * 3 * #{u}", [1, 2]],
        [[[v] * 2] * 3, "3 * 2 * #{t}", "3 * 2 * #{u}", [2, 1]],
        [[[v] * 40] * 3, "3 * 40 * #{t}", "3 * 40 * #{u}", [1, 32]]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        x = XND.new vv, type: ttt

        unless u.nil?
          y = XND.new vv, type: uuu

          if eq
            assert_strict_equal x, y
          else
            assert_strict_unequal x, y                  
          end
        end              
        
        if tt == "2 * 2 * {x: uint16, y: {z: ?complex64}}" &&
           uu == "2 * 2 * {x: uint16, y: {z: ?complex64}}"
          x = XND.new vv, type: ttt

          unless w.nil?
            y = XND.new vv, type: ttt

            y[*indices] = w
            assert_strict_unequal x, y
            
            y = XND.new vv, type: uuu
            y[*indices] = w
            assert_strict_unequal x, y
          end
        end
      end
    end
  end

  def test_fixed_dim_assign
    #### full data
    x = XND.empty "2 * 4 * float64"
    v = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]

    # assigns full slice
    x[INF] = v
    assert_equal x.value, v

    # assigns subarray
    x[INF] = v
    
    x[0] = v[0] = [1.2, -3e45, Float::INFINITY, -322.25]
    assert_equal x.value, v

    x[1] = v[1] = [-11.25, 3.355e301, -0.000002, -5000.2]
    assert_equal x.value, v

    # assigns single values
    0.upto(1) do |i|
      0.upto(3) do |j|
        x[i][j] = v[i][j] = 3.22 * i + j
      end
    end

    assert_equal x.value, v

    # supports tuple indexing
    0.upto(1) do |i|
      0.upto(3) do |j|
        x[i, j] = v[i][j] = -3.002e1 * i + j
      end
    end

    assert_equal x.value, v

    ### optional data
    x = XND.empty "2 * 4 * ?float64"
    v = [[10.0, nil, 2.0, 100.12], [nil, nil, 6.0, 7.0]]     

    # assigns full slice
    x[INF] = v
    assert_equal x.value, v

    # assigns subarray
    x[INF] = v
    
    x[0] = v[0] = [nil, 3e45, Float::INFINITY, nil]
    assert_equal x.value, v

    x[1] = v[1] = [-11.25, 3.355e301, -0.000002, nil]
    assert_equal x.value, v

    # assigns single values
    2.times do |i|
      4.times do |j|
        x[i][j] = v[i][j] = -325.99 * i + j
      end
    end

    assert_equal x.value, v

    # supports assignment by tuple indexing
    2.times do |i|
      4.times do |j|
        x[i, j] = v[i][j] = -8.33e1 * i + j
      end
    end

    assert_equal x.value, v
  end
end # class TestFixedDim

class TestFortran < Minitest::Test
  def test_fortran_empty
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [[v] * 0, "!0 * #{s}"],
        [[v] * 1, "!1 * #{s}"],
        [[v] * 2, "!2 * #{s}"],
        [[v] * 1000, "!1000 * #{s}"],

        [[[v] * 0] * 0, "!0 * 0 * #{s}"],
        [[[v] * 1] * 0, "!0 * 1 * #{s}"],
        [[[v] * 0] * 1, "!1 * 0 * #{s}"],

        [[[v] * 1] * 1, "!1 * 1 * #{s}"],
        [[[v] * 2] * 1, "!1 * 2 * #{s}"],
        [[[v] * 1] * 2, "!2 * 1 * #{s}"],
        [[[v] * 2] * 2, "!2 * 2 * #{s}"],
        [[[v] * 3] * 2, "!2 * 3 * #{s}"],
        [[[v] * 2] * 3, "!3 * 2 * #{s}"],
        [[[v] * 40] * 3, "!3 * 40 * #{s}"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss

        assert_equal t, x.type
        assert_equal vv, x.value
        assert_equal vv.size, x.size
      end
    end
  end

  def test_fortran_slices
    [
      [[[11.12-2.3i, -1222+20e8i],
        [Complex(Float::INFINITY, Float::INFINITY), -0.00002i],
        [0.201+1i, -1+1e301i]], "!3 * 2 * complex128"],
      [[[11.12-2.3i, nil],
        [Complex(Float::INFINITY, Float::INFINITY), nil],
        [0.201+1i, -1+1e301i]], "!3 * 2 * ?complex128"]
    ].each do |v, s|
      arr = v
      t = NDT.new s
      x = XND.new v, type: t

      (0).upto(2) do |i|
        assert_equal x[i].value, arr[i]
      end

      (0).upto(2) do |i|
        (0).upto(1) do |k|
          assert_equal x[i][k].value, arr[i][k]
          assert_equal x[i, k].value, arr[i][k]
        end
      end

      # checks full slice
      assert_equal x[INF].to_a, arr

      # slice with ranges
      ((-3..-3).to_a + [Float::INFINITY]).each do |start|
        ((-3..-3).to_a + [Float::INFINITY]).each do |stop|
          [true, false].each do |exclude_end|
            # FIXME: add step count loop post Ruby 2.6
            arr_s = get_inf_or_normal_range start, stop, exclude_end
            r = Range.new start, stop, exclude_end
            assert_equal x[r].value, arr[arr_s]
          end
        end
      end

      # checks column slices"
      assert_equal x[INF, 0].value, arr.transpose[0]
      assert_equal x[INF, 1].value, arr.transpose[1]
    end
  end

  def test_fortran_assign
    #### Full data
    x = XND.empty "!2 * 4 * float64"
    v = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]

    # assigns full slice
    x[INF] = v
    assert_equal x.value, v

    # assigns subarray
    x[INF] = v
    
    x[0] = v[0] = [1.2, -3e45, Float::INFINITY, -322.25]
    assert_equal x.value, v

    x[1] = v[1] = [-11.25, 3.355e301, -0.000002, -5000.2]
    assert_equal x.value, v
    

    # assigns single values
    0.upto(1) do |i|
      0.upto(3) do |j|
        x[i][j] = v[i][j] = 3.22 * i + j
      end
    end

    assert_equal x.value, v
    
    # supports tuple indexing
    0.upto(1) do |i|
      0.upto(3) do |j|
        x[i, j] = v[i][j] = -3.002e1 * i + j
      end
    end

    assert_equal x.value, v

    ### Optional data
    x = XND.empty "!2 * 4 * ?float64"
    v = [[10.0, nil, 2.0, 100.12], [nil, nil, 6.0, 7.0]]     

    # assigns full slice
    x[INF] = v
    assert_equal x.value, v

    # assigns subarray
    x[INF] = v
    
    x[0] = v[0] = [nil, 3e45, Float::INFINITY, nil]
    assert_equal x.value, v

    x[1] = v[1] = [-11.25, 3.355e301, -0.000002, nil]
    assert_equal x.value, v

    # assigns single values
    2.times do |i|
      4.times do |j|
        x[i][j] = v[i][j] = -325.99 * i + j
      end
    end

    assert_equal x.value, v

    # supports assignment by tuple indexing
    2.times do |i|
      4.times do |j|
        x[i, j] = v[i][j] = -8.33e1 * i + j
      end
    end

    assert_equal x.value, v
  end

  def test_equality
    x = XND.new [1,2,3,4], type: "!4 * int64"
    
    # test basic case
    assert_strict_equal x, XND.new([1,2,3,4], type: "!4 * int64")

    # tests different shape and/or data
    assert_strict_unequal x, XND.new([1,2,3,100], type: "!4 * int64")
    assert_strict_unequal x, XND.new([1,2,3], type: "!3 * int64")
    assert_strict_unequal x, XND.new([1,2,3,4,5], type: "!5 * int64")

    # tests different shapes
    assert_strict_unequal x, XND.new([1,2,3], type: "!3 * int64")
    assert_strict_unequal x, XND.new([[1,2,3,4]], type: "!1 * 4 * int64")
    assert_strict_unequal x, XND.new([[1,2], [3,4]], type: "!2 * 2 * int64")

    # tests simple multidimensional arrays
    x = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type: "!4 * 3 * int64")
    y = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type: "!4 * 3 * int64")

    assert_strict_equal x, y

    # equality after assignment
    x = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type: "!4 * 3 * int64")
    y = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type: "!4 * 3 * int64")
    4.times do |i|
      3.times do |k|
        v = y[i, k]
        y[i, k] = 100

        assert_strict_unequal x, y
        y[i, k] = v
      end
    end

    # tests slices
    x = XND.new([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], type: "!4 * 3 * int64")
    y = XND.new([[1,2,3], [4,5,6]])

    assert_strict_equal x[0..1], y

    y = XND.new([1,4,7,10], type: "!4 * int64")

    assert_strict_equal x[INF, 0], y

    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      
      [
        [[v] * 0, "!0 * #{t}", "!0 * #{u}"],
        [[[v] * 0] * 0, "!0 * 0 * #{t}", "!0 * 0 * #{u}"],
        [[[v] * 1] * 0, "!0 * 1 * #{t}", "!0 * 1 * #{u}"],
        [[[v] * 0] * 1, "!1 * 0 * #{t}", "!1 * 0 * #{u}"]
      ].each do |vv, tt, uu|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        y = XND.new vv, type: uuu
        assert_strict_equal x, y
      end
    end

    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      w = struct.w
      eq = struct.eq
      
      [
        [[v] * 1, "!1 * #{t}", "!1 * #{u}", [0]],
        [[v] * 2, "!2 * #{t}", "!2 * #{u}", [1]],
        [[v] * 1000, "!1000 * #{t}", "!1000 * #{u}", [961]],

        [[[v] * 1] * 1, "!1 * 1 * #{t}", "!1 * 1 * #{u}", [0, 0]],
        [[[v] * 2] * 1, "!1 * 2 * #{t}", "!1 * 2 * #{u}", [0, 1]],
        [[[v] * 1] * 2, "!2 * 1 * #{t}", "!2 * 1 * #{u}", [1, 0]],
        [[[v] * 2] * 2, "!2 * 2 * #{t}", "!2 * 2 * #{u}", [1, 1]],
        [[[v] * 3] * 2, "!2 * 3 * #{t}", "!2 * 3 * #{u}", [1, 2]],
        [[[v] * 2] * 3, "!3 * 2 * #{t}", "!3 * 2 * #{u}", [2, 1]],
        [[[v] * 40] * 3, "!3 * 40 * #{t}", "!3 * 40 * #{u}", [1, 32]]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt

        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          y = XND.new vv, type: uuu
          if eq
            assert_strict_equal x, y
          else
            assert_strict_unequal x, y
          end
        end

        unless w.nil?
          y = XND.new vv, type: ttt
          y[*indices] = w
          assert_strict_unequal x, y

          y = XND.new vv, type: uuu
          y[*indices] = w
          assert_strict_unequal x, y
        end
      end
    end
  end
end # class TestFortran

class TestVarDim < Minitest::Test
  def test_var_dim_empty
    DTYPE_EMPTY_TEST_CASES[0..10].each do |v, s|
      [
        [[v] * 0, "var(offsets=[0,0]) * #{s}"],
        [[v] * 1, "var(offsets=[0,1]) * #{s}"],
        [[v] * 2, "var(offsets=[0,2]) * #{s}"],
        [[v] * 1000, "var(offsets=[0,1000]) * #{s}"],
        
        [[[v] * 0] * 1, "var(offsets=[0,1]) * var(offsets=[0,0]) * #{s}"],
        
        [[[v], []], "var(offsets=[0,2]) * var(offsets=[0,1,1]) * #{s}"],
        [[[], [v]], "var(offsets=[0,2]) * var(offsets=[0,0,1]) * #{s}"],
        
        [[[v], [v]], "var(offsets=[0,2]) * var(offsets=[0,1,2]) * #{s}"],
        [[[v], [v] * 2, [v] * 5], "var(offsets=[0,3]) * var(offsets=[0,1,3,8]) * #{s}"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss
        
        assert_equal x.type, t
        assert_equal x.value, vv
        assert_equal x.size, vv.size
      end
    end

    # returns empty view
    inner = [[0+0i] * 5] * 4
    x = XND.empty "2 * 3 * ref(4 * 5 * complex128)"

    y = x[1][2]
    assert_equal y.is_a?(XND), true
    assert_equal y.value, inner

    y = x[1, 2]
    assert_equal y.is_a?(XND), true
    assert_equal y.value, inner
  end

  def test_var_dim_assign
    ### regular data

    x = XND.empty "var(offsets=[0,2]) * var(offsets=[0,2,5]) * float64"
    v = [[0.0, 1.0], [2.0, 3.0, 4.0]]


    # assigns full slice
    x[INF] = v
    assert_equal x.value, v


    # assigns subarray
    x[INF] = v
    
    x[0] = v[0] = [1.2, 2.5]
    assert_equal x.value, v

    x[1] = v[1] = [1.2, 2.5, 3.99]
    assert_equal x.value, v

    # assigns individual values
    2.times do |i|
      x[0][i] = v[0][i] = 100.0 * i
    end

    3.times do |i|
      x[1][i] = v[1][i] = 200.0 * i
    end

    assert_equal x.value, v

    # assigns tuple
    2.times do |i|
      x[0, i] = v[0][i] = 300.0 * i + 1.222
    end

    3.times do |i|
      x[1, i] = v[1][i] = 400.0 * i + 1.333
    end

    assert_equal x.value, v

    ### optional data
    x = XND.empty "var(offsets=[0,2]) * var(offsets=[0,2,5]) * ?float64"
    v = [[0.0, nil], [nil, 3.0, 4.0]]

    # assigns full slice
    x[INF] = v
    assert_equal x.value, v

    # assigns subarray
    x[INF] = v
    
    x[0] = v[0] = [nil, 2.0]
    assert_equal x.value, v

    x[1] = v[1] = [1.22214, nil, 10.0]
    assert_equal x.value, v

    # assigns individual values
    2.times do |i|
      x[0][i] = v[0][i] = 3.14 * i + 1.2222
    end

    3.times do |i|
      x[1][i] = v[1][i] = 23.333 * i
    end

    assert_equal x.value, v

    # assigns tuple
    2.times do |i|
      x[0, i] = v[0][i] = -122.5 * i + 1.222
    end

    3.times do |i|
      x[1, i] = v[1][i] = -3e22 * i
    end

    assert_equal x.value, v
  end

  def test_var_dim_overflow
    s = "var(offsets=[0, 2]) * var(offsets=[0, 1073741824, 2147483648]) * uint8"
    assert_raises(ValueError) { XND.empty(s) }
  end

  def test_var_dim_match
    x = XND.new([0,1,2,3,4], type: "var(offsets=[0,5]) * complex128")
    sig = NDT.new("var... * complex128 -> var... * complex128")

    spec = sig.apply([x.type])
    assert type_equal(spec.out_types[0], x.type)

    y = x[1..3]
    spec = sig.apply([y.type])
    assert type_equal(spec.out_types[0], x.type)

    sig = NDT.new("var... * complex128, var... * complex128 -> var... * complex128")
    spec = sig.apply([x.type, y.type])
    assert type_equal(spec.out_types[0], x.type)

    x = XND.new([[0], [1, 2], [3, 4, 5]], dtype: "complex128")
    y = XND.new([[5, 4, 3], [2, 1], [0]], dtype: "complex128")
    spec = sig.apply([x.type, y.type])
    assert type_equal(spec.out_types[0], x.type)
  end

  def test_var_dim_equality
    x = XND.new [1,2,3,4], type: "var(offsets=[0,4]) * int64"
    
    # compares full array
    assert_strict_equal x, XND.new([1,2,3,4], type: "var(offsets=[0,4]) * int64")

    # tests for different shape and/or data
    assert_strict_unequal x, XND.new([1,2,3,100], type: "var(offsets=[0,4]) * int64")
    assert_strict_unequal x, XND.new([1,2,3], type: "var(offsets=[0,3]) * int64")
    assert_strict_unequal x, XND.new([1,2,3,4,5], type: "var(offsets=[0,5]) * int64")

    # tests different shape
    assert_strict_unequal x, XND.new([1,2,3], type: "var(offsets=[0,3]) * int64")
    assert_strict_unequal x, XND.new([[1,2,3,4]],
                                     type: "var(offsets=[0,1]) * var(offsets=[0,4]) * int64")
    assert_strict_unequal x, XND.new(
                            [[1,2], [3,4]], type: "var(offsets=[0,2]) * var(offsets=[0,2,4]) * int64")

    # tests multidimensional arrays
    x = XND.new([[1], [2,3,4,5], [6,7], [8,9,10]])
    y = XND.new([[1], [2,3,4,5], [6,7], [8,9,10]])
    
    assert_strict_equal(x, y)

    # tests multidim arrays after assign
    x = XND.new([[1], [2,3,4,5], [6,7], [8,9,10]])
    y = XND.new([[1], [2,3,4,5], [6,7], [8,9,10]])

    (0..3).to_a.zip([1,4,2,3]).each do |i, shape|
      shape.times do |k|
        v = y[i, k]
        y[i, k] = 100

        assert_strict_unequal x, y
        
        y[i, k] = v
      end
    end

    # tests slices
    x = XND.new([[1], [4,5], [6,7,8], [9,10,11,12]])
    
    y = XND.new([[1], [4,5]])
    assert_strict_equal x[0..1], y

    y = XND.new([[4,5], [6,7,8]])
    assert_strict_equal x[1..2], y

    # TODO: make this pass after Ruby 2.6 step-range
    # y = XND.new([[12,11,10,9], [5,4]])
    # assert_strict_equal x[(0..) % -2, (0..) % -1], y

    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      [
        [[v] * 0, "var(offsets=[0,0]) * #{t}",
         "var(offsets=[0,0]) * #{u}"],
        [[[v] * 0] * 1, "var(offsets=[0,1]) * var(offsets=[0,0]) * #{t}",
         "var(offsets=[0,1]) * var(offsets=[0,0]) * #{u}"]
      ].each do |vv, tt, uu|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        y = XND.new vv, type: uuu
        assert_strict_equal x, y
      end
    end

    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      w = struct.w
      eq = struct.eq
      
      [
        [[v] * 1, "var(offsets=[0,1]) * #{t}", "var(offsets=[0,1]) * #{u}", [0]],
        [[v] * 2, "var(offsets=[0,2]) * #{t}", "var(offsets=[0,2]) * #{u}", [1]],
        [[v] * 1000, "var(offsets=[0,1000]) * #{t}", "var(offsets=[0,1000]) * #{u}", [961]],
        [[[v], []], "var(offsets=[0,2]) * var(offsets=[0,1,1]) * #{t}",
         "var(offsets=[0,2]) * var(offsets=[0,1,1]) * #{u}", [0, 0]],
        [[[], [v]], "var(offsets=[0,2]) * var(offsets=[0,0,1]) * #{t}",
         "var(offsets=[0,2]) * var(offsets=[0,0,1]) * #{u}", [1, 0]],
        [[[v], [v]], "var(offsets=[0,2]) * var(offsets=[0,1,2]) * #{t}",
         "var(offsets=[0,2]) * var(offsets=[0,1,2]) * #{u}", [1, 0]],
        [[[v], [v] * 2, [v] * 5], "var(offsets=[0,3]) * var(offsets=[0,1,3,8]) * #{t}",
         "var(offsets=[0,3]) * var(offsets=[0,1,3,8]) * #{u}", [2, 3]]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt

        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          y = XND.new vv, type: uuu
          if eq
            assert_strict_equal x, y
          else
            assert_strict_unequal x, y
          end
        end

        unless w.nil?
          y = XND.new vv, type: ttt
          y[*indices] = w
          assert_strict_unequal x, y

          y = XND.new vv, type: uuu
          y[*indices] = w
          assert_strict_unequal x, y
        end
      end
    end
  end
end # class TestVarDim

class TestSymbolicDim < Minitest::Test
  def test_symbolic_dim_raise
    DTYPE_EMPTY_TEST_CASES.each do |_, s|
      [
        [ValueError, "N * #{s}"],
        [ValueError, "10 * N * #{s}"],
        [ValueError, "N * 10 * N * #{s}"],
        [ValueError, "X * 10 * N * #{s}"]
      ].each do |err, ss|
        t = NDT.new ss
        
        assert_raises(ValueError) { XND.empty t } 
      end
    end
  end
end # class TestSymbolicDim

class TestEllipsisDim < Minitest::Test
  def test_tuple_empty
    DTYPE_EMPTY_TEST_CASES.each do |_, s|
      [
        [ValueError, "... * #{s}"],
        [ValueError, "Dims... * #{s}"],
        [ValueError, "... * 10 * #{s}"],
        [ValueError, "B... *2 * 3 * ref(#{s})"],
        [ValueError, "A... * 10 * Some(ref(#{s}))"],
        [ValueError, "B... * 2 * 3 * Some(ref(ref(#{s})))"]
      ].each do |err, ss|
        t = NDT.new ss
        
        assert_raises(err) { XND.empty ss } 
      end
    end
  end
end # class TestEllipsisDim

class TestTuple < Minitest::Test
  def test_tuple_empty
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [[v], "(#{s})"],
        [[[v]], "((#{s}))"],
        [[[[v]]], "(((#{s})))"],

        [[[v] * 0], "(0 * #{s})"],
        [[[[v] * 0]], "((0 * #{s}))"],
        [[[v] * 1], "(1 * #{s})"],
        [[[[v] * 1]], "((1 * #{s}))"],
        [[[v] * 3], "(3 * #{s})"],
        [[[[v] * 3]], "((3 * #{s}))"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss
        
        assert_equal x.type, t
        assert_equal x.value, vv
        assert_equal x.size, vv.size
      end
    end
  end

  def test_tuple_assign
    ### regular data
    x = XND.empty "(complex64, bytes, string)"
    v = [1+20i, "abc".b, "any"]          

    # assigns each element
    x[0] = v[0]
    x[1] = v[1]
    x[2] = v[2]

    assert_equal x.value, v

    ### optional data

    x = XND.empty "(complex64, ?bytes, ?string)"
    v = [1+20i, nil, "Some"]          


    # assigns each element
    x[0] = v[0]
    x[1] = v[1]
    x[2] = v[2]

    assert_equal x.value, v

    # assigns new each element
    v = [-2.5+125i, nil, nil]
    x[0] = v[0]
    x[1] = v[1]
    x[2] = v[2]

    assert_equal x.value, v

    # assigns tuple and individual values
    x = XND.new([
                  XND::T.new("a", 100, 10.5),
                  XND::T.new("a", 100, 10.5)
                ])
    x[0][1] = 200000000

    assert_equal x[0][1].value, 200000000
    assert_equal x[0, 1].value, 200000000
  end

  def test_tuple_overflow
    # Type cannot be created.
    s = "(4611686018427387904 * uint8, 4611686018427387904 * uint8)"
    assert_raises(ValueError) { XND.empty(s) }
  end

  def test_tuple_optional_values
    lst = [[nil, 1, 2], [3, nil, 4], [5, 6, nil]]
    x = XND.new(lst, dtype: "(?int64, ?int64, ?int64)")
    assert_equal x.value, lst
  end

  def test_tuple_equality
    ### simple test
    x = XND.new XND::T.new(1, 2.0, "3", "123".b)
    
    # checks simple equality
    assert_strict_equal x, XND.new(XND::T.new(1, 2.0, "3", "123".b))

    # checks simple inequality
    assert_strict_unequal x, XND.new(XND::T.new(2, 2.0, "3", "123".b))
    assert_strict_unequal x, XND.new(XND::T.new(1, 2.1, "3", "123".b))
    assert_strict_unequal x, XND.new(XND::T.new(1, 2.0, "", "123".b))
    assert_strict_unequal x, XND.new(XND::T.new(1, 2.0, "345", "123".b))
    assert_strict_unequal x, XND.new(XND::T.new(1, 2.0, "3", "".b))
    assert_strict_unequal x, XND.new(XND::T.new(1, 2.0, "3", "12345".b))

    ### nested structures
    t = "(uint8,
          fixed_string(100, 'utf8'),
          (complex128, 2 * 3 * (fixed_bytes(size=64, align=32), bytes)),
          ref(string))"

    v = [
      10,
      "\u00001234\u00001001abc",
      [
        12.1e244+3i,
        [[
           ["123".b, "22".b * 10],
           ["123456".b, "23".b * 10],
           ["123456789".b, "24".b * 10]
         ],
         [
           ["1".b, "a".b],
           ["12".b, "ab".b],
           ["123".b, "abc".b]
         ]]],
      "xyz"
    ]

    x = XND.new v, type: t
    y = XND.new v, type: t

    # simple equality
    assert_strict_equal x, y

    # unequal after assignment
    w = y[0].value
    y[0] = 11

    assert_strict_unequal x, y

    # equal after assignment
    w = y[0].value
    y[0] = w

    assert_strict_equal x, y

    # unequal after UTF-8 assign
    w = y[1].value
    y[1] = "\U00001234\U00001001abx"

    assert_strict_unequal x, y

    y[1] = w
    assert_strict_equal x, y

    # equal after tuple assign
    w = y[2,0].value
    y[2,0] = 12.1e244-3i
    assert_strict_unequal x, y

    y[2,0] = w
    assert_strict_equal x, y

    # assigns large index value
    w = y[2,1,1,2,0].value
    y[2,1,1,2,0] = "abc".b
    assert_strict_unequal x, y

    y[2,1,1,2,0] = w
    assert_strict_equal x, y

    # assign empty string
    w = y[3].value
    y[3] = ""
    assert_strict_unequal x, y

    y[3] = w
    assert_strict_equal x, y

    ### simple corner cases
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u

      [
        [[[v] * 0], "(0 * #{t})", "(0 * #{u})"],
        [[[[v] * 0]], "((0 * #{t}))", "((0 * #{u}))"]
      ].each do |vv, tt, uu|
        uu = uu
        vv = vv
        tt = tt
        ttt = NDT.new tt
        uuu = NDT.new uu
        x = XND.new vv, type: ttt
        y = XND.new(vv, type: ttt)

        
        assert_strict_equal x, y
        

        y = XND.new vv, type: uuu
        assert_strict_equal x, y
      end
    end

    # tests complex corner cases
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      w = struct.w
      eq = struct.eq

      [
        [[v], "(#{t})", "(#{u})", [0]],
        [[[v]], "((#{t}))", "(#{u})", [0, 0]],
        [[[[v]]], "(((#{t})))", "(((#{u})))", [0, 0, 0]],

        [[[v] * 1], "(1 * #{t})", "(1 * #{u})", [0, 0]],
        [[[[v] * 1]], "((1 * #{t}))", "((1 * #{u}))", [0, 0, 0]],
        [[[v] * 3], "(3 * #{t})", "(3 * #{u})", [0, 2]]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        uuu = NDT.new uu
        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          y = XND.new vv, type: uuu
          if eq
            assert_strict_equal x, y
          else
            assert_strict_unequal x, y
          end
        end

        unless w.nil?
          y = XND.new vv, type: ttt
          y[*indices] = w
          assert_strict_unequal x, y

          y = XND.new vv, type: uuu
          y[*indices] = w
          assert_strict_unequal x, y
        end
      end
    end
  end
end # class TestTuple

class TestRecord < Minitest::Test
  def test_record_empty
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [{'x' => v}, "{x: #{s}}"],
        [{'x' => {'y' => v}}, "{x: {y: #{s}}}"],

        [{'x' => [v] * 0}, "{x: 0 * #{s}}"],
        [{'x' => {'y' => [v] * 0}}, "{x: {y: 0 * #{s}}}"],
        [{'x' => [v] * 1}, "{x: 1 * #{s}}"],
        [{'x' => [v] * 3}, "{x: 3 * #{s}}"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss
        
        assert_equal x.type , t
        assert_equal x.value, vv
        assert_equal x.size, vv.size
      end
    end
  end

  def test_record_assign
    ### assigns regular data
    x = XND.empty "{x: complex64, y: bytes, z: string}"
    v = { 'x' => 1+20i, 'y' => "abc".b, 'z' => "any" }

    x['x'] = v['x']
    x['y'] = v['y']
    x['z'] = v['z']

    assert_equal x.value, v

    ### optional data
    x = XND.empty "{x: complex64, y: ?bytes, z: ?string}"
    [
      { 'x' => 1+20i, 'y' => nil, 'z' => "Some"  },
      { 'x' => -2.5+125i, 'y' => nil, 'z' => nil }
    ].each do |v| 
      x['x'] = v['x']
      x['y'] = v['y']
      x['z'] = v['z']
      
      assert_equal @x.value, v
    end
  end

  def test_record_overflow
    # Type cannot be created.
    s = "{a: 4611686018427387904 * uint8, b: 4611686018427387904 * uint8}"
    assert_raises(ValueError) { XND.empty(s) }
  end

  def test_record_optional_values
    lst = [
      {'a' => nil, 'b' => 2, 'c' => 3},
      {'a'=> 4, 'b' => nil, 'c' => 5},
      {'a'=> 5, 'b' => 6, 'c'=> nil}
    ]
    x = XND.new(lst, dtype: "{a: ?int64, b: ?int64, c: ?int64}")

    assert_equal x.value, lst
  end

  def test_record_equality
    ### simple tests
    x = XND.new({'a' => 1, 'b' => 2.0, 'c' => "3", 'd' => "123".b})

    assert_strict_equal x, XND.new({'a' => 1, 'b' => 2.0, 'c' => "3", 'd' => "123".b})

    assert_strict_unequal x, XND.new({'z' => 1, 'b' => 2.0, 'c' => "3", 'd' => "123".b})
    assert_strict_unequal x, XND.new({'a' => 2, 'b' => 2.0, 'c' => "3", 'd' => "123".b})
    assert_strict_unequal x, XND.new({'a' => 1, 'b' => 2.1, 'c' => "3", 'd' => "123".b})
    assert_strict_unequal x, XND.new({'a' => 1, 'b' => 2.0, 'c' => "", 'd' => "123".b})
    assert_strict_unequal x, XND.new({'a' => 1, 'b' => 2.0, 'c' => "345", 'd' => "123"})
    assert_strict_unequal x, XND.new({'a' => 1, 'b' => 2.0, 'c' => "3", 'd' => "".b})
    assert_strict_unequal x, XND.new({'a' => 1, 'b' => 2.0, 'c' => "3", 'd' => "12345".b})

    ### nested structures
    t = "
            {a: uint8,
             b: fixed_string(100, 'utf8'),
             c: {x: complex128,
                 y: 2 * 3 * {v: fixed_bytes(size=64, align=32),
                             u: bytes}},
             d: ref(string)}
            "
    v = {
      'a' => 10,
      'b' => "\U00001234\U00001001abc",
      'c' => {'x' => 12.1e244+3i,
              'y' => [[{'v' => "123".b, 'u' => "22".b * 10},
                       {'v' => "123456".b, 'u' => "23".b * 10},
                       {'v' => "123456789".b, 'u' => "24".b * 10}],
                      [{'v' => "1".b, 'u' => "a".b},
                       {'v' => "12".b, 'u' => "ab".b},
                       {'v' => "123".b, 'u' => "abc".b}]]
             },
      'd' => "xyz"
    }

    x = XND.new v, type: t
    y = XND.new v, type: t
    assert_strict_equal x, y

    w = y[0].value
    y[0] = 11
    assert_strict_unequal x, y
    y[0] = w
    assert_strict_equal x, y

    w = y[1].value
    y[1] = "\U00001234\U00001001abx"
    assert_strict_unequal x, y
    y[1] = w
    assert_strict_equal x, y

    w = y[2,0].value
    y[2,0] = 12.1e244-3i
    assert_strict_unequal x, y

    y[2, 0] = w
    assert_strict_equal x, y

    w = y[2,1,1,2,0].value
    y[2,1,1,2,0] = "abc".b
    assert_strict_unequal x, y

    y[2,1,1,2,0] = w
    assert_strict_equal x, y

    w = y[3].value
    y[3] = ""
    assert_strict_unequal x, y
    y[3] = w
    assert_strict_equal x, y

    # test corner cases
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u

      [
        [{'x' => [v] * 0}, "{x: 0 * #{t}}", "{x: 0 * #{u}}"],
        [{'x' => {'y' => [v] * 0}}, "{x: {y: 0 * #{t}}}", "{x: {y: 0 * #{u}}}"]
      ].each do |vv, tt, uu|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        y = XND.new vv, type: uuu
        assert_strict_equal x, y
      end
    end

    # test many dtypes
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      w = struct.w
      eq = struct.eq

      [
        [{'x' => v}, "{x: #{t}}", "{x: #{u}}", [0]],
        [{'x' => {'y' => v}}, "{x: {y: #{t}}}", "{x: {y: #{u}}}", [0, 0]],
        [{'x' => [v] * 1}, "{x: 1 * #{t}}", "{x: 1 * #{u}}", [0, 0]],
        [{'x' => [v] * 3}, "{x: 3 * #{t}}", "{x: 3 * #{u}}", [0, 2]]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          y = XND.new vv, type: uuu
          if eq
            assert_strict_equal x, y
          else
            assert_strict_unequal x, y
          end
        end

        unless w.nil?
          y = XND.new vv, type: ttt
          y[*indices] = w

          assert_strict_unequal x, y
        end
      end
    end
  end
end # class TestRecord

class TestRef < Minitest::Test
  def test_ref_empty
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [v, "ref(#{s})"],
        [v, "ref(ref(#{s}))"],
        [v, "ref(ref(ref(#{s})))"],

        [[v] * 0, "ref(0 * #{s})"],
        [[v] * 0, "ref(ref(0 * #{s}))"],
        [[v] * 0, "ref(ref(ref(0 * #{s})))"],
        [[v] * 1, "ref(1 * #{s})"],
        [[v] * 1, "ref(ref(1 * #{s}))"],
        [[v] * 1, "ref(ref(ref(1 * #{s})))"],
        [[v] * 3, "ref(3 * #{s})"],
        [[v] * 3, "ref(ref(3 * #{s}))"],
        [[v] * 3, "ref(ref(ref(3 * #{s})))"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss
        
        assert_equal x.type, t
        assert_equal x.value, vv
      end
    end
  end

  def test_ref_empty_view
    # If a ref is a dtype but contains an array itself, indexing should
    # return a view and not a Python value.
    inner = 4 * [5 * [0+0i]]
    x = XND.empty("2 * 3 * ref(4 * 5 * complex128)")

    y = x[1][2]
    assert_kind_of XND, y
    assert_equal inner, y.value

    y = x[1, 2]
    assert_kind_of XND, y
    assert_equal inner, y.value
  end

  def test_ref_indexing
    # FIXME: If a ref is a dtype but contains an array itself, indexing through
    # the ref should work transparently. Make equality transparent.
    inner = [['a', 'b', 'c', 'd', 'e'],
             ['f', 'g', 'h', 'i', 'j'],
             ['k', 'l', 'm', 'n', 'o'],
             ['p', 'q', 'r', 's', 't']]
    v = [[inner] * 3] * 2
    x = XND.new(v, type: "2 * 3 * ref(4 * 5 * string)")

    (0).upto(1) do |i|
      (0).upto(2) do |j|
        (0).upto(3) do |k|
          (0).upto(4) do |l|
            assert_equal x[i][j][k][l], inner[k][l]
            assert_equal x[i, j, k, l], inner[k][l]
          end
        end
      end
    end
  end

  def test_ref_assign
    # TODO
  end

  def test_ref_equality
    x = XND.new [1,2,3,4], type: "ref(4 * float32)"

    assert_strict_equal x, XND.new([1,2,3,4], type: "ref(4 * float32)")

    assert_strict_unequal x, XND.new([1,2,3,4,5], type: "ref(5 * float32)")
    assert_strict_unequal x, XND.new([1,2,3], type: "ref(3 * float32)")
    assert_strict_unequal x, XND.new([1,2,3,43,5], type: "ref(5 * float32)")

    # corner cases and many dtypes
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u

      [
        [[v] * 0, "ref(0 * #{t})", "ref(0 * #{u})"],
        [[v] * 0, "ref(ref(0 * #{t}))", "ref(ref(0 * #{u}))"],
        [[v] * 0, "ref(ref(ref(0 * #{t})))", "ref(ref(ref(0 * #{u})))"]
      ].each do |vv, tt, uu|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        y = XND.new vv, type: uuu
        assert_strict_equal x, y
      end
    end

    # many dtypes and indices
    EQUAL_TEST_CASES.each do |struct|
      v = struct.v
      t = struct.t
      u = struct.u
      w = struct.w
      eq = struct.eq

      [
        [v, "ref(#{t})", "ref(#{u})", []],
        [v, "ref(ref(#{t}))", "ref(ref(#{u}))", []],
        [v, "ref(ref(ref(#{t})))", "ref(ref(ref(#{u})))", []],
        [[v] * 1, "ref(1 * #{t})", "ref(1 * #{u})", 0],
        [[v] * 3, "ref(3 * #{t})", "ref(3 * #{u})", 2]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        uuu = NDT.new uu

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          y = XND.new vv, type: uuu
          if eq
            assert_strict_equal x, y
          else
            assert_strict_unequal x, y
          end
        end

        unless w.nil?
          y = XND.new vv, type: ttt
          y[indices] = w

          assert_strict_unequal x, y
        end
      end
    end
  end
end # class TestRef
