require_relative 'test_helper'

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
        [[[v] * 40] * 3 , "3 * 40 * #{s}" ]
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
        
        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        unless u.nil?
          uuu = NDT.new uu          
          y = XND.new vv, type: uuu
          assert_strict_equal x, y
        end

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

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu
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

          unless u.nil?
            uuu = NDT.new uu
            y = XND.new vv, type: uuu
            y[*indices] = w
            assert_strict_unequal x, y
          end
        end
      end
    end
  end

  def test_fixed_dim_indexing
    # returns single number slice for 1D array/1 number
    xnd = XND.new([1,2,3,4])
    assert_equal xnd[1], XND.new(2)
    
    # returns single number slice for 2D array and 2 indices
    xnd = XND.new([[1,2,3], [4,5,6]])
    assert_equal xnd[0,0], XND.new(1)
    
    # returns row for single index in 2D array
    x = XND.new [[1,2,3], [4,5,6], [7,8,9]]
    assert_equal x[1], XND.new([4,5,6])

    # returns single column in 2D array
    x = XND.new [[1,2,3], [4,5,6], [7,8,9]]
    assert_equal x[0..Float::INFINITY, 0], XND.new([1,4,7])

    # returns the entire array
    x = XND.new [[1,2,3], [4,5,6], [7,8,9]]
    assert_equal x[0..Float::INFINITY], x

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
      @arr = v
      @t = NDT.new s
      @x = XND.new v, type: @t

      assert_equal @x.to_a, @arr.to_a
      
      0.upto(2) do |i|
        assert_equal @x[i].to_a, @arr[i]
      end

      3.times do |i|
        2.times do |k|
          assert @x[i][k].value == @arr[i][k]
          assert @x[i, k].value == @arr[i][k]    
        end
      end

      assert_equal @x[INF].value, @arr
      
      ((-3...4).to_a + [Float::INFINITY]).each do |start|
        ((-3...4).to_a + [Float::INFINITY]).each do |stop|
          [true, false].each do |exclude_end|
            # FIXME: add step count when ruby supports it.
            arr_s = get_inf_or_normal_range start, stop, exclude_end
            r = Range.new(start, stop, exclude_end)
            assert_equal @x[r].value, @arr[arr_s]
          end
        end
      end
      assert_equal @x[INF, 0].value, @arr.transpose[0]
      assert_equal @x[INF, 1].value, @arr.transpose[1]
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
          assert x[i][k].value == arr[i][k]
          assert x[i, k].value == arr[i][k]
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

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        unless u.nil?
          uuu = NDT.new uu
          y = XND.new vv, type: uuu
          assert_strict_equal x, y
        end
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

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt

        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu
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
    x = XND.new([Complex(0),Complex(1),Complex(2),Complex(3),Complex(4)],
                type: "var(offsets=[0,5]) * complex128")
    sig = NDT.new("var... * complex128 -> var... * complex128")

    spec = sig.apply([x.type])
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

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        unless u.nil? 
          uuu = NDT.new uu
          y = XND.new vv, type: uuu
          assert_strict_equal x, y
        end
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

        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt

        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu
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

          unless u.nil?
            uuu = NDT.new uu
            y = XND.new vv, type: uuu
            y[*indices] = w
            assert_strict_unequal x, y
          end
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
        
        assert_raises(err) { XND.empty t } 
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

        x = XND.new vv, type: ttt
        y = XND.new(vv, type: ttt)
        assert_strict_equal x, y
        
        unless u.nil?
          uuu = NDT.new uu
          y = XND.new vv, type: uuu
          assert_strict_equal x, y          
        end
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
        [[[v]], "((#{t}))", "((#{u}))", [0, 0]],
        [[[[v]]], "(((#{t})))", "(((#{u})))", [0, 0, 0]],

        [[[v] * 1], "(1 * #{t})", "(1 * #{u})", [0, 0]],
        [[[[v] * 1]], "((1 * #{t}))", "((1 * #{u}))", [0, 0, 0]],
        [[[v] * 3], "(3 * #{t})", "(3 * #{u})", [0, 2]]
      ].each do |vv, tt, uu, indices|
        ttt = NDT.new tt
        
        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu        
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

          unless u.nil?
            uuu = NDT.new uu
            y = XND.new vv, type: uuu
            y[*indices] = w
            assert_strict_unequal x, y
          end
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
      
      assert_equal x.value, v
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

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        unless u.nil?
          uuu = NDT.new uu
          y = XND.new vv, type: uuu
          assert_strict_equal x, y
        end
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

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu
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
        
        assert x.type == t
        assert x.value == vv
      end
    end
  end

  def test_ref_empty_view
    # If a ref is a dtype but contains an array itself, indexing should
    # return a view and not a Python value.
    inner = [[0+0i] * 5] * 4
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
    # If a ref is a dtype but contains an array itself, assigning through
    # the ref should work transparently.
    inner = [['a', 'b', 'c', 'd', 'e'],
             ['f', 'g', 'h', 'i', 'j'],
             ['k', 'l', 'm', 'n', 'o'],
             ['p', 'q', 'r', 's', 't']]
    v = [[inner] * 3] * 2

    x = XND.new(v, type: "2 * 3 * ref(4 * 5 * string)")

    2.times do |i|
      3.times do |j|
        4.times do |k|
          5.times do |l|
            x[i, j, k, l] = inner[k][l] = "#{k * 5 + l}"
          end
        end
      end
    end

    assert_equal x.value, v

    2.times do |i|
      3.times do |j|
        4.times do |k|
          5.times do |l|
            x[i, j, k, l] = inner[k][l] = "#{k * 5 + l}"
          end
        end
      end
    end

    assert_equal x.value, v
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
        
        x = XND.new vv, type: ttt
        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        unless u.nil?
          uuu = NDT.new uu
          y = XND.new vv, type: uuu
          assert_strict_equal x, y
        end
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

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu
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

class TestConstr < Minitest::Test
  def test_constr_empty
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [v, "SomeConstr(#{s})"],
        [v, "Just(Some(#{s}))"],

        [[v] * 0, "Some(0 * #{s})"],
        [[v] * 1, "Some(1 * #{s})"],
        [[v] * 3, "Maybe(3 * #{s})"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss
        
        assert x.type == t
        assert x.value == vv
        if vv == 0
          assert_raises(NoMethodError) { x.size }
        end
      end
    end
  end

  def test_constr_empty_view
    # If a constr is a dtype but contains an array itself, indexing should
    # return a view and not a Python value.
    inner = [[""] * 5] * 4
    x = XND.empty("2 * 3 * InnerArray(4 * 5 * string)")

    y = x[1][2]
    assert_kind_of XND, y
    assert_equal y.value, inner

    y = x[1, 2]
    assert_kind_of XND, y
    assert_equal y.value, inner
  end

  def test_constr_indexing
    # FIXME: If a constr is a dtype but contains an array itself, indexing through
    # the constructor should work transparently. However, for now it returns
    # an XND object, however this will likely change in the future.
    inner = [['a', 'b', 'c', 'd', 'e'],
             ['f', 'g', 'h', 'i', 'j'],
             ['k', 'l', 'm', 'n', 'o'],
             ['p', 'q', 'r', 's', 't']]
    v = [[inner] * 3] * 2
    x = XND.new(v, type: "2 * 3 * InnerArray(4 * 5 * string)")

    (0).upto(1) do |i|
      (0).upto(2) do |j|
        (0).upto(3) do |k|
          (0).upto(4) do |l|
            
            assert_equal x[i][j][k][l].value, inner[k][l]
            assert_equal x[i, j, k, l].value,  inner[k][l]
          end
        end
      end
    end
  end

  def test_constr_assign
    # If a constr is a dtype but contains an array itself, assigning through
    # the constructor should work transparently.
    inner = [['a', 'b', 'c', 'd', 'e'],
             ['f', 'g', 'h', 'i', 'j'],
             ['k', 'l', 'm', 'n', 'o'],
             ['p', 'q', 'r', 's', 't']]

    v = [[inner] * 3] * 2
    x = XND.new(v, type: "2 * 3 * InnerArray(4 * 5 * string)")

    2.times do |i|
      3.times do |j|
        4.times do |k|
          5.times do |l|
            x[i][j][k][l] = inner[k][l] = "#{k * 5 + l}"
          end
        end
      end
    end

    assert_equal x.value, v

    2.times do |i|
      3.times do |j|
        4.times do |k|
          5.times do |l|
            x[i][j][k][l] = inner[k][l] = "#{k * 5 + l + 1}"
          end
        end
      end
    end

    assert_equal x.value, v
  end

  def test_constr_equality
    # simple tests
    x = XND.new [1,2,3,4], type: "A(4 * float32)"

    assert_strict_equal x, XND.new([1,2,3,4], type: "A(4 * float32)")

    assert_strict_unequal x, XND.new([1,2,3,4], type: "B(4 * float32)")
    assert_strict_unequal x, XND.new([1,2,3,4,5], type: "A(5 * float32)")
    assert_strict_unequal x, XND.new([1,2,3], type: "A(3 * float32)")
    assert_strict_unequal x, XND.new([1,2,3,4,55], type: "A(5 * float32)")

    # corner cases and dtypes
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

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        assert_strict_equal x, y

        unless u.nil?
          uuu = NDT.new uu
          y = XND.new vv, type: uuu
          assert_strict_equal x, y          
        end
      end
    end

    # more dtypes
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

        x = XND.new vv, type: ttt

        y = XND.new vv, type: ttt
        if eq
          assert_strict_equal x, y
        else
          assert_strict_unequal x, y
        end

        unless u.nil?
          uuu = NDT.new uu
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
end # class TestConstr

class TestNominal < Minitest::Test
  def test_nominal_empty
    c = 0
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      NDT.typedef "some#{c}", s
      NDT.typedef "just#{c}", "some#{c}"
      
      [
        [v, "some#{c}"],
        [v, "just#{c}"]
      ].each do |vv, ss|
        t = NDT.new ss
        x = XND.empty ss
        
        assert x.type == t
        assert x.value == vv
        if vv == 0
          assert_raises(NoMethodError) { x.size }
        end
      end 
      c += 1
    end
  end

  def test_nominal_empty_view
    # If a typedef is a dtype but contains an array itself, indexing should
    # return a view and not a Python value.
    NDT.typedef("inner_array", "4 * 5 * string")
    inner = [[""] * 5] * 4
    x = XND.empty("2 * 3 * inner_array")

    y = x[1][2]
    assert_equal y.is_a?(XND), true
    assert_equal y.value, inner

    y = x[1, 2]
    assert_equal y.is_a?(XND), true
    assert_equal y.value, inner
  end

  def test_nominal_indexing
    # FIXME: If a typedef is a dtype but contains an array itself, indexing through
    # the constructor should work transparently.
    NDT.typedef("inner", "4 * 5 * string")
    inner = [['a', 'b', 'c', 'd', 'e'],
             ['f', 'g', 'h', 'i', 'j'],
             ['k', 'l', 'm', 'n', 'o'],
             ['p', 'q', 'r', 's', 't']]
    v = [[inner] * 3] * 2
    x = XND.new(v, type: "2 * 3 * inner")        

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

  def test_nominal_assign
    # If a typedef is a dtype but contains an array itself, assigning through
    # the constructor should work transparently.
    NDT.typedef("in", "4 * 5 * string")
    inner = [['a', 'b', 'c', 'd', 'e'],
             ['f', 'g', 'h', 'i', 'j'],
             ['k', 'l', 'm', 'n', 'o'],
             ['p', 'q', 'r', 's', 't']]

    v = [[inner] * 3] * 2
    x = XND.new(v, type: "2 * 3 * in")

    2.times do |i|
      3.times do |j|
        4.times do |k|
          5.times do |l|
            x[i][j][k][l] = inner[k][l] = "#{k * 5 + l}"
          end
        end
      end
    end

    assert_equal v, x.value

    2.times do |i|
      3.times do |j|
        4.times do |k|
          5.times do |l|
            x[i][j][k][l] = inner[k][l] = "#{k * 5 + l + 1}"
          end
        end
      end
    end

    assert_equal v, x.value
  end

  def test_nominal_error
    assert_raises(ValueError) { XND.empty("undefined_t") }
  end

  def test_nominal_equality
    NDT.typedef "some1000", "4 * float32"
    NDT.typedef "some1001", "4 * float32"

    x = XND.new([1,2,3,4], type: "some1000")

    assert_strict_equal x, XND.new([1,2,3,4], type: "some1000")

    assert_strict_unequal x, XND.new([1,2,3,4], type: "some1001")
    assert_strict_unequal x, XND.new([1,2,3,5], type: "some1000")
  end
end # class TestNominal

class TestScalarKind < Minitest::Test
  def test_scalar_kind
    assert_raises(ValueError) { XND.empty("Scalar") }
  end
end # class TestScalarKind

class TestCategorical < Minitest::Test
  def test_categorical_empty
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

      t = NDT.new s
      x = XND.empty s

      assert_equal x.type, t
      assert_equal x.value, v
    end
  end

  def test_categorical_assign
    s = "2 * categorical(NA, 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')"
    x = XND.new [nil, nil], type: s
    
    # assigns regular data
    x[0] = "August"
    x[1] = "December"

    assert_equal x.value, ["August", "December"]

    # assigns nil
    x[0] = nil
    x[1] = "December"

    assert_equal x.value, [nil, "December"]
  end

  def test_categorical_equality
    t = "3 * categorical(NA, 'January', 'August')"
    x = XND.new ['August', 'January', 'January'], type: t

    y = XND.new ['August', 'January', 'January'], type: t
    assert_strict_equal x, y

    y = XND.new ['August', 'January', 'August'], type: t
    assert_strict_unequal x, y

    x = XND.new ['August', nil, 'August'], type: t
    y = XND.new ['August', nil, 'August'], type: t

    assert_strict_unequal x, y
  end
end # class TestCategorical

class TestFixedStringKind < Minitest::Test
  def test_fixed_string_kind
    assert_raises(ValueError) { XND.empty("FixedString") }
  end
end # class TestFixedStringKind

class TestFixedString < Minitest::Test
  def test_fixed_string_empty
    # tests kind of string
    assert_raises(ValueError) {
      XND.empty "FixedString"
    }

    [
      ["fixed_string(1)", ""],
      ["fixed_string(3)", "" * 3],
      ["fixed_string(1, 'ascii')", ""],
      ["fixed_string(3, 'utf8')", "" * 3],
      ["fixed_string(3, 'utf16')", "" * 3],
      ["fixed_string(3, 'utf32')", "" * 3],
      ["2 * fixed_string(3, 'utf32')", ["" * 3] * 2],
    ].each do |s, v|
      
      t = NDT.new s
      x = XND.empty s

      assert_equal x.type, t
      assert_equal x.value, v
    end
  end

  def test_fixed_string
    skip "figure out the best way to do this in ruby."
    # creates FixedString utf16
    t = "2 * fixed_string(3, 'utf16')"
    v = ["\u1111\u2222\u3333", "\u1112\u2223\u3334"]
    x = XND.new v, type: t
    
    assert_equal x.value, v.map { |q| q.encode("UTF-16") }

    # creates FixedString utf32 - figure a way to specify 32bit codepoints
    t = "2 * fixed_string(3, 'utf32')"
    v = ["\x00\x01\x11\x11\x00\x02\x22\x22\x00\x03\x33\x33".encode('UTF-32'),
         "\x00\x01\x11\x12\x00\x02\x22\x23\x00\x03\x33\x34".encode('UTF-32')]
    x = XND.new v, type: t
    
    assert_equal x.value, v
  end

  def test_fixed_string_equality
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

      assert_strict_equal x, y

      y[[]] = w

      assert_strict_unequal x, y
    end
  end

  def test_fixed_string_assign
    skip "Figure how to deal with UTF-32 strings in Ruby."
    
    t = "2 * fixed_string(3, 'utf32')"
    v = ["\U00011111\U00022222\U00033333", "\U00011112\U00022223\U00033334"]
    x = XND.new(v, type: t)

    x[0] = "a"
    assert_equal x.value, ["a", "\U00011112\U00022223\U00033334"]

    x[0] = "a\x00\x00"
    assert_equal x.value, ["a", "\U00011112\U00022223\U00033334"]

    x[1] = "b\x00c"
    assert_equal x.value, ["a", "b\x00c"]
  end

  def test_fixed_string_overflow
    ["fixed_string(9223372036854775808)",
     "fixed_string(4611686018427387904, 'utf16')",
     "fixed_string(2305843009213693952, 'utf32')"
    ].each do |s|
      assert_raises(ValueError) { XND.empty(s) }
    end
  end
end # class TestFixedString

class TestFixedBytesKind < Minitest::Test
  def test_fixed_bytes_kind
    assert_raises(ValueError) { XND.empty("FixedBytes") }
  end
end # class TestFixedBytesKind

class TestFixedBytes < Minitest::Test
  def test_fixed_bytes_empty
    r = {'a' => "\x00".b * 3, 'b' => "\x00".b * 10}

    [
      ["\x00".b, 'fixed_bytes(size=1)'],
      ["\x00".b * 100, 'fixed_bytes(size=100)'],
      ["\x00".b * 4, 'fixed_bytes(size=4, align=2)'],
      ["\x00".b * 128, 'fixed_bytes(size=128, align=16)'],
      [r, '{a: fixed_bytes(size=3), b: fixed_bytes(size=10)}'],
      [[[r] * 3] * 2, '2 * 3 * {a: fixed_bytes(size=3), b: fixed_bytes(size=10)}']
    ].each do |v, s|
      t = NDT.new s
      x = XND.empty s

      assert_equal x.type, t
      assert_equal x.value, v
    end
  end

  def test_fixed_bytes_assign
    t = "2 * fixed_bytes(size=3, align=1)"
    v = ["abc".b, "123".b]
    x = XND.new(v, type: t)
    
    x[0] = "xyz".b
    assert_equal x.value, ["xyz".b, "123".b]
  end

  def test_fixed_bytes_overflow
    # Type cannot be created.
    s = "fixed_bytes(size=9223372036854775808)"

    assert_raises(ValueError) { XND.empty(s) }
  end

  def test_fixed_bytes_equality
    [
      ["a".b, "fixed_bytes(size=1)", "b".b],
      ["a".b * 100, "fixed_bytes(size=100)", "a".b * 99 + "b".b],
      ["a".b * 4, "fixed_bytes(size=4, align=2)", "a".b * 2 + "b".b]
    ].each do |v, t, w|
      x = XND.new v, type: t
      y = XND.new v, type: t

      assert_strict_equal x, y

      y[[]] = w
      assert_strict_unequal x, y
    end

    # align
    x = XND.new("a".b * 128, type: "fixed_bytes(size=128, align=16)")
    y = XND.new("a".b * 128, type: "fixed_bytes(size=128, align=16)")
    assert_strict_equal x, y
  end
end # class TestFixedBytes

class TestString < Minitest::Test
  def test_string_empty
    [
      'string',
      '(string)',
      '10 * 2 * string',
      '10 * 2 * (string, string)',
      '10 * 2 * {a: string, b: string}',
      'var(offsets=[0,3]) * var(offsets=[0,2,7,10]) * {a: string, b: string}'
    ].each do |s|

      t = NDT.new s
      x = XND.empty s
      assert_equal x.type, t
    end

    # tests for single value
    t = NDT.new "string"
    x = XND.empty t

    assert_equal x.type, t
    assert_equal x.value, ''

    # tests for multiple values
    t = NDT.new "10 * string"
    x = XND.empty t

    assert_equal x.type, t
    10.times do |i| 
      assert_equal x[i], ''
    end
  end

  def test_string
    t = '2 * {a: complex128, b: string}'
    x = XND.new([{'a' => 2+3i, 'b' => "thisguy"},
                 {'a' => 1+4i, 'b' => "thatguy"}], type: t)

    assert_equal x[0]['b'].value, "thisguy"
    assert_equal x[1]['b'].value, "thatguy"
  end

  def test_string_assign
    t = '2 * {a: complex128, b: string}'
    x = XND.new([{ 'a' => 2+3i, 'b' => "thisguy"},
                 { 'a' => 1+4i, 'b' => "thatguy" }], type: t)

    x[0] = { 'a' => 220i, 'b' => 'y'}
    x[1] = { 'a' => -12i, 'b' => 'z'}

    assert_equal x.value, [{ 'a' => 220i, 'b' => 'y' }, { 'a' => -12i, 'b' => 'z' }]
  end

  def test_string_equality
    x = XND.new "abc"

    assert_strict_equal x, XND.new("abc")
    assert_strict_equal x, XND.new("abc\0\0")

    assert_strict_unequal x, XND.new("acb")
  end
end # class TestString

class TestBytes < Minitest::Test
  def test_bytes_empty
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
      t = NDT.new s
      x = XND.empty t

      assert_equal x.type, t
      assert_equal x.value, v
    end
  end

  def test_bytes_assign
    t = "2 * SomeByteArray(3 * bytes)"
    inner = ["a".b, "b".b, "c".b]
    v = [inner] * 2
    x = XND.new v, type: t

    2.times do |i|
      3.times do |k|
        x[i, k] = inner[k] = ['x'.chr.ord + k].pack("C")
      end
    end

    assert_equal x.value, v
  end
end # class TestBytes

class TestChar < Minitest::Test
  def test_char
    assert_raises(ValueError) { XND.empty("char('utf8')")}
    assert_raises(ValueError) { XND.new(1, type: "char('utf8')")}
  end
end # class TestChar

class TestBool < Minitest::Test
  def test_bool
    # from bool
    x = XND.new true, type: "bool"
    assert_equal x.value, true

    x = XND.new false, type: "bool"
    assert_equal x.value, false

    # from int
    x = XND.new 1, type: "bool"
    assert_equal x.value, true

    x = XND.new 0, type: "bool"
    assert_equal x.value, false

    # from object
    x = XND.new [1,2,3], type: "bool"
    assert_equal x.value, true

    x = XND.new nil, type: "?bool"
    assert_nil x.value

    assert_raises(TypeError) {
      XND.new nil, type: "bool"
    }
  end

  def test_bool_equality
    assert_strict_equal XND.new(true), XND.new(true)
    assert_strict_equal XND.new(false), XND.new(false)
    assert_strict_unequal XND.new(true), XND.new(false)
    assert_strict_unequal XND.new(false), XND.new(true)
  end
end # class TestBool

class TestSignedKind < Minitest::Test
  def test_signed_kind
    assert_raises(ValueError) { XND.empty("Signed")}
  end
end # class TestSignedKind

class TestSigned < Minitest::Test
  def test_signed
    [8, 16, 32, 64].each do |n|
      t = "int#{n}"

      v = -2**(n-1)
      x = XND.new(v, type: t)
      assert_equal x.value, v
      assert_raises(RangeError) { XND.new v-1, type: t }

      v = 2**(n-1) - 1
      x = XND.new(v, type: t)
      assert_equal x.value, v
      assert_raises(RangeError) { XND.new v+1, type: t }
    end
  end

  def test_signed_equality
    ["int8", "int16", "int32", "int64"].each do |t|
      assert_strict_equal XND.new(-10, type: t), XND.new(-10, type: t)
      assert_strict_unequal XND.new(-10, type: t), XND.new(100, type: t)
    end
  end
end # class TestSigned

class TestUnsignedKind < Minitest::Test
  def test_unsigned_kind
    assert_raises(ValueError) { XND.empty("Unsigned") }
  end
end # class TestUnsignedKind

class TestUnsigned < Minitest::Test
  def test_unsigned
    [8, 16, 32, 64].each do |n|
      t = "uint#{n}"

      v = 0
      x = XND.new v, type: t
      assert_equal x.value, v
      assert_raises(RangeError) { XND.new v-1, type: t }
      
      t = "uint#{n}"
      
      v = 2**n - 2
      x = XND.new v, type: t
      assert_equal x.value, v
      assert_raises(RangeError) { XND.new v+2, type: t }
    end
  end

  def test_equality
    ["uint8", "uint16", "uint32", "uint64"].each do |t|
      assert_strict_equal XND.new(10, type: t), XND.new(10, type: t)
      assert_strict_unequal XND.new(10, type: t), XND.new(100, type: t)
    end
  end
end # class TestUnsigned

class TestFloatKind < Minitest::Test
  def test_float_kind
    assert_raises(ValueError) { XND.empty("Float")}
  end
end # class TestFloatKind

class TestFloat < Minitest::Test
  def test_float32
    # tests inf bounds
    inf = Float("0x1.ffffffp+127")

    assert_raises(RangeError) { XND.new(inf, type: "float32") }
    assert_raises(RangeError) { XND.new(-inf, type: "float32") }

    # tests denorm_min bounds
    denorm_min = Float("0x1p-149")

    x = XND.new denorm_min, type: "float32"
    assert_equal x.value, denorm_min

    # tests lowest bounds
    lowest = Float("-0x1.fffffep+127")
    
    x = XND.new lowest, type: "float32"
    assert_equal x.value.nan?, lowest.nan?
    
    # tests max bounds
    max = Float("0x1.fffffep+127")

    x = XND.new max, type: "float32"
    assert_equal x.value, max

    # tests special values
    x = XND.new Float::INFINITY, type: "float32"
    assert_equal x.value.infinite?, 1

    x = XND.new Float::NAN, type: "float32"
    assert_equal x.value.nan?, true


    # compare
    assert_strict_equal XND.new(1.2e7, type: "float32"),
                        XND.new(1.2e7, type: "float32")
    assert_strict_equal XND.new(Float::INFINITY, type: "float32"),
                        XND.new(Float::INFINITY, type: "float32")
    assert_strict_equal XND.new(-Float::INFINITY, type: "float32"),
                        XND.new(-Float::INFINITY, type: "float32")

    assert_strict_unequal XND.new(1.2e7, type: "float32"),
                          XND.new(-1.2e7, type: "float32")
    assert_strict_unequal XND.new(Float::INFINITY, type: "float32"),
                          XND.new(-Float::INFINITY, type: "float32")
    assert_strict_unequal XND.new(-Float::INFINITY, type: "float32"),
                          XND.new(Float::INFINITY, type: "float32")
    assert_strict_unequal XND.new(Float::NAN, type: "float32"),
                          XND.new(Float::NAN, type: "float32")
  end

  def test_float64
    # tests bounds
    denorm_min = Float("0x0.0000000000001p-1022")
    lowest = Float("-0x1.fffffffffffffp+1023")
    max = Float("0x1.fffffffffffffp+1023")

    x = XND.new denorm_min, type: "float64"
    assert_equal x.value, denorm_min

    x = XND.new lowest, type: "float64"
    assert_equal x.value, lowest

    x = XND.new max, type: "float64"
    assert_equal x.value, max


    # tests special values
    x = XND.new Float::INFINITY, type: "float64"
    assert_equal x.value.infinite?, 1

    x = XND.new Float::NAN, type: "float64"
    assert_equal x.value.nan?, true
    
    # compare
    assert_strict_equal XND.new(1.2e7, type: "float64"),
                        XND.new(1.2e7, type: "float64")
    assert_strict_equal XND.new(Float::INFINITY, type: "float64"),
                        XND.new(Float::INFINITY, type: "float64")
    assert_strict_equal XND.new(-Float::INFINITY, type: "float64"),
                        XND.new(-Float::INFINITY, type: "float64")

    assert_strict_unequal XND.new(1.2e7, type: "float64"),
                          XND.new(-1.2e7, type: "float64")
    assert_strict_unequal XND.new(Float::INFINITY, type: "float64"),
                          XND.new(-Float::INFINITY, type: "float64")
    assert_strict_unequal XND.new(-Float::INFINITY, type: "float64"),
                          XND.new(Float::INFINITY, type: "float64")
    assert_strict_unequal XND.new(Float::NAN, type: "float64"),
                          XND.new(Float::NAN, type: "float64")
  end
end # class Float

class TestComplexKind < Minitest::Test
  def test_complex_kind
    assert_raises(ValueError) { XND.empty("Complex") }
  end
end # class TestComplexKind

class TestComplex < Minitest::Test
  def test_complex64
    # tests bounds
    denorm_min = Float("0x1p-149")
    lowest = Float("-0x1.fffffep+127")
    max = Float("0x1.fffffep+127")
    inf = Float("0x1.ffffffp+127")

    v = Complex(denorm_min, denorm_min)
    x = XND.new v, type: "complex64"
    assert_equal x.value, v

    v = Complex(lowest, lowest)
    x = XND.new v, type: "complex64"
    assert_equal x.value, v

    v = Complex(max, max)
    x = XND.new v, type: "complex64"
    assert_equal x.value, v

    v = Complex(inf, inf)
    assert_raises(RangeError) { XND.new v, type: "complex64" }

    v = Complex(-inf, -inf)
    assert_raises(RangeError) { XND.new v, type: "complex64" }

    # tests special values
    x = XND.new Complex(Float::INFINITY, 0), type: "complex64"
    assert_equal x.value.real.infinite?, 1
    assert_equal x.value.imag, 0.0

    x = XND.new Complex(Float::NAN, 0), type: "complex64"
    assert_equal x.value.real.nan?, true
    assert_equal x.value.imag, 0.0

    # compare
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
              assert_strict_equal x, y
            else
              assert_strict_unequal x, y
            end
          end
        end
      end
    end
  end

  def test_complex128
    # tests bounds
    denorm_min = Float("0x0.0000000000001p-1022")
    lowest = Float("-0x1.fffffffffffffp+1023")
    max = Float("0x1.fffffffffffffp+1023")

    v = Complex(denorm_min, denorm_min)
    x = XND.new v, type: "complex128"
    assert_equal x.value, v

    v = Complex(lowest, lowest)
    x = XND.new v, type: "complex128"
    assert_equal x.value, v

    v = Complex(max, max)
    x = XND.new v, type: "complex128"
    assert_equal x.value, v

    # tests special values
    x = XND.new Complex(Float::INFINITY), type: "complex128"

    assert_equal x.value.real.infinite?, 1
    assert_equal x.value.imag, 0.0

    x = XND.new Complex(Float::NAN), type: "complex128"

    assert_equal x.value.real.nan?, true
    assert_equal x.value.imag, 0.0

    # compare
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
              assert_strict_equal x, y
            else
              assert_strict_unequal x, y
            end
          end
        end
      end
    end
  end
end # class TestComplex

class TestPrimitive < Minitest::Test
  def test_primitive_empty
    empty_test_cases.each do |value, type_string|
      PRIMITIVE.each do |p|
        ts = type_string % p

        x = XND.empty ts

        assert_equal x.value, value
        assert_equal x.type, NDT.new(ts)
      end
    end

    empty_test_cases(false).each do |value, type_string|
      BOOL_PRIMITIVE.each do |p|
        ts = type_string % p

        x = XND.empty ts

        assert_equal x.value, value
        assert_equal x.type, NDT.new(ts)
      end
    end
  end
end # class TestPrimitive

class TestTypevar < Minitest::Test
  def test_typevar
    [
      "T",
      "2 * 10 * T",
      "{a: 2 * 10 * T, b: bytes}"
    ].each do |ts|
      assert_raises(ValueError) {
        XND.empty ts
      }
    end
  end
end # class TestTypevar

class TestTypeInference < Minitest::Test
  def test_accumulate
    arr = [1,2,3,4,5]
    result = [1,3,6,10,15]

    assert_equal XND::TypeInference.accumulate(arr), result
  end

  def test_search
    data = [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
    result = [[3], [2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8], *(Array.new(MAX_DIM-2) { [] })]
    
    min_level = MAX_DIM + 1
    max_level = 0
    acc = Array.new(MAX_DIM + 1) { [] }
    minmax = [min_level, max_level]

    XND::TypeInference.search max_level, data, acc, minmax

    assert_equal acc, result
    assert_equal minmax[0], minmax[1]
  end

  def test_data_shapes
    # extract shape of nested Array data
    data = [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
    result = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [[2, 3, 4], [3]]]

    assert_equal XND::TypeInference.data_shapes(data), result

    # empty array
    data = []
    result = [[], [[0]]]

    assert_equal XND::TypeInference.data_shapes(data), result

    # empty nested array
    data = [[]]
    result = [[], [[0], [1]]]

    assert_equal XND::TypeInference.data_shapes(data), result
  end

  def test_type_of
    # correct ndtype of fixed array
    value = [
      [1,2,3],
      [5,6,7]
    ]
    type = NDTypes.new "2 * 3 * int64"

    assert_equal XND::TypeInference.type_of(value), type

    # generates correct ndtype for hash
    value = {
      "a" => "xyz",
      "b" => [1,2,3]
    }
    type = NDTypes.new "{a : string, b : 3 * int64}"
    assert_equal XND::TypeInference.type_of(value), type
  end
  
  def test_tuple
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
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert_equal x.value, XND::TypeInference.convert_xnd_t_to_ruby_array(v)
    end
  end

  def test_record
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
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert_equal x.value, v
    end
  end

  def test_float64
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
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert_equal x.value, v
    end
  end

  def test_complex128
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
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert_equal x.value, v
    end
  end

  def test_int64
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
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert_equal x.value, XND::TypeInference.convert_xnd_t_to_ruby_array(v)
    end
  end

  def test_string
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
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert_equal x.value, XND::TypeInference.convert_xnd_t_to_ruby_array(v)
    end
  end

  def test_bytes
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
      x = XND.new v
      
      assert_equal x.type, NDT.new(t)
      assert_equal x.value, XND::TypeInference.convert_xnd_t_to_ruby_array(v)
    end
  end

  def test_optional
    [
      [nil, "?float64"],
      [[nil], "1 * ?float64"],
      [[nil, nil], "2 * ?float64"],
      [[nil, 10], "2 * ?int64"],
      [[nil, 'abc'.b], "2 * ?bytes"],
      [[nil, 'abc'], "2 * ?string"]
    ].each do |v, t|
      x = XND.new v

      assert_equal x.type, NDT.new(t)
      assert x.value == v
    end

    [
      [nil, []],
      [[], nil],
      [nil, [10]],
      [[nil, [0, 1]], [[2, 3]]]
    ].each do |v|
      assert_raises(NotImplementedError) { XND.new v }   
    end
  end
end # class TestTypeInference

class TestEach < Minitest::Test
  def test_each
    DTYPE_EMPTY_TEST_CASES.each do |v, s|
      [
        [[[v] * 1] * 1, "!1 * 1 * #{s}"],
        [[[v] * 2] * 1, "!1 * 2 * #{s}"],
        [[[v] * 1] * 2, "!2 * 1 * #{s}"],
        [[[v] * 2] * 2, "2 * 2 * #{s}"],
        [[[v] * 3] * 2, "2 * 3 * #{s}"],
        [[[v] * 2] * 3, "3 * 2 * #{s}"]
      ].each do |vv, ss|
        x = XND.new vv, type: ss

        lst = []
        x.each do |v|
          lst << v
        end

        x.each_with_index do |i, z|
          assert_equal z.value, lst[i].value
        end
      end
    end
  end
end # class TestEach

class TestAPI < Minitest::Test
  def test_short_value
    x = XND.new [1,2]
    q = XND::Ellipsis.new
    
    assert_equal x.short_value(0), []
    assert_equal x.short_value(1), [XND::Ellipsis.new]
    assert_equal x.short_value(2), [1, XND::Ellipsis.new]
    assert_equal x.short_value(3), [1, 2]

    x = XND.new [[1,2], [3]]
    assert_equal [], x.short_value(0)
    assert_equal [XND::Ellipsis.new], x.short_value(1)
    assert_equal [[1, XND::Ellipsis.new], XND::Ellipsis.new], x.short_value(2)
    assert_equal [[1, 2], [3]], x.short_value(3)
    assert_raises(ArgumentError) { x.short_value(-1) }
    
    x = XND.new({'a' => 1, 'b' => 2 })
    assert_equal x.short_value(0), {}
    assert_equal x.short_value(3), {'a' => 1, 'b'=> 2}
    assert_raises(ArgumentError){ x.short_value(-1) }
  end
end # class TestAPI

class TestToS < Minitest::Test
  def test_to_s
    lst = [[[{'a'=> 100, 'b' => "xyz", 'c'=> ['abc', 'uvw']}] * 23] * 19] * 10
    x = XND.new lst
    r = x.to_s

    assert r.size < 100000
  end
end # class TestToS

class TestBuffer < Minitest::Test
  def test_from_nmatrix
    
  end

  def test_from_numbuffer
    
  end

  def test_from_narray
    
  end
end # class TestBuffer

class TestSplit < Minitest::Test
  def test_split
    
  end

  def test_split_limit_outer
    
  end
end # class TestSplit

class TestView < Minitest::Test
  def test_view_subscript
    
  end

  def test_view_new
    
  end
end

