#  BSD 3-Clause License
# 
#  Copyright (c) 2018, Quansight and Sameer Deshmukh
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

require 'ndtypes'
begin
  require "ruby_xnd.so"
rescue LoadError
  require 'ruby_xnd/ruby_xnd.so'
end

require 'xnd/version'

INF = Float::INFINITY

class XND
  class Ellipsis
    def to_s
      "..."
    end

    def == other
      other.is_a?(XND::Ellipsis) ? true : false
    end
  end
end

class XND < RubyXND
  # Immutable array type used when specifying XND tuples without wanting to
  # specify the type. It is highly recommended to simply specify the type
  # and use Ruby Arrays for specifying the data.
  #
  # If you call #to_a or #value on XND after specifying data as XND::T, note
  # that XND will return the data in the form of Ruby Arrays.
  #
  # The sole purpose for the existence of this class is the facilitation of
  # type inference. Before passing to the C interface, all instaces of XND::T
  # will be converted into Ruby Arrays.
  #
  # @example
  #
  # x = XND.new XND::T.new([])
  # #=> XND([], type: ())
  class T
    include Enumerable
    attr_reader :data
    
    def initialize *args, &block
      @data = args
    end

    def each &block
      @data.each(&block)
    end

    def map &block
      @data.map(&block)
    end

    def map! &block
      @data.map!(&block)
    end

    def [] *index
      @data[index]
    end
  end
  
  MAX_DIM = NDTypes::MAX_DIM
  
  # Methods for type inference.
  module TypeInference
    class << self
      # Infer the type of a Ruby value. In general, types should be explicitly
      # specified.
      def type_of value, dtype: nil
        NDTypes.new actual_type_of(value, dtype: dtype)
      end

      def actual_type_of value, dtype: nil
        ret = nil
        if value.is_a?(Array)
          data, shapes = data_shapes value
          opt = data.include? nil

          if dtype.nil?
            if data.nil?
              dtype = 'float64'
            else
              dtype = choose_dtype(data)

              data.each do |x|
                if !x.nil?
                  t = actual_type_of(x)
                  if t != dtype
                    raise ValueError, "dtype mismatch: have t=#{t} and dtype=#{dtype}"
                  end
                end
              end
            end

            dtype = '?' + dtype if opt
          end

          t = dtype

          var = shapes.map { |lst| lst.uniq.size > 1 || nil }.any?
          shapes.each do |lst|
            opt = lst.include? nil
            lst.map! { |x| x.nil? ? 0 : x }
            t = add_dim(opt: opt, shapes: lst, typ: t, use_var: var)
          end

          ret = t
        elsif !dtype.nil?
          raise TypeError, "dtype argument is only supported for Arrays."
        elsif value.is_a? Hash
          if value.keys.all? { |k| k.is_a?(String) }
            ret = "{" + value.map { |k, v| "#{k} : #{actual_type_of(v)}"}.join(", ") + "}"
          else
            raise ValueError, "all hash keys must be String."  
          end
        elsif value.is_a? XND::T # tuple
          ret = "(" + value.map { |v| actual_type_of(v) }.join(",") + ")"
        elsif value.nil?
          ret = '?float64'
        elsif value.is_a? Float
          ret = 'float64'
        elsif value.is_a? Complex
          ret = 'complex128'
        elsif value.is_a? Integer
          ret = 'int64'
        elsif value.is_a? String
          ret = value.encoding == Encoding::ASCII_8BIT ? 'bytes' : 'string'
        elsif value.is_a?(TrueClass) || value.is_a?(FalseClass)
          ret = 'bool'
        else
          raise ArgumentError, "cannot infer data type for: #{value}"
        end

        ret
      end
      
      def accumulate arr
        result = []
        arr.inject(0) do |memo, a|
          result << memo + a
          memo + a
        end

        result
      end

      # Construct a new dimension type based on the list of 'shapes' that
      # are present in a dimension.
      def add_dim *args, opt: false, shapes: nil, typ: nil, use_var: false
        if use_var
          offsets = [0] + accumulate(shapes)
          return "#{opt ? '?' : ''}var(offsets=#{offsets}) * #{typ}"
        else
          n = shapes.uniq.size
          shape = (n == 0 ? 0 : shapes[0])
          return "#{shape} * #{typ}"
        end
      end
      
      # Internal function for extracting the data and dimensions of nested arrays.
      def search level, data, acc, minmax
        raise(ValueError, "too many dimensions: #{level}") if level > MAX_DIM

        current = acc[level]
        if data.nil?
          current << data
        elsif data.is_a?(Array)
          current << data.size
          next_level = level + 1
          minmax[1] = [next_level, minmax[1]].max

          if !data
            minmax[0] = [next_level, minmax[0]].min
          else
            data.each do |item|
              search level+1, item, acc, minmax
            end
          end
        else
          acc[minmax[1]] << data
          minmax[0] = [level, minmax[0]].min
        end
      end
      
      # Extract array data and dimension shapes from a nested Array. The
      # Array may contain nil for missing data or dimensions.
      #
      # @example
      # data_shapes [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
      # #=> [[0, 1, 2, 3, 4, 5, 6, 7, 8], [[2, 3, 4], [3]]]
      # #              ^                    ^          ^
      # #              |                    |          `--- ndim=2: single shape 3
      # #              |                    `-- ndim=1: shapes 2, 3, 4
      # #              `--- ndim=0: extracted array data
      def data_shapes tree
        acc = Array.new(MAX_DIM + 1) { [] }
        min_level = MAX_DIM + 1
        max_level = 0
        minmax = [min_level, max_level]

        search max_level, tree, acc, minmax

        min_level = minmax[0]
        max_level = minmax[1]

        if acc[max_level] && acc[max_level].all? { |a| a.nil? }
          # min_level is not set in this special case. Hence the check.
        elsif min_level != max_level
          raise ValueError, "unbalanced tree: min depth #{min_level} and max depth #{max_level}"
        end

        data = acc[max_level]
        shapes = acc[0...max_level].reverse

        [data, shapes]
      end

      def choose_dtype array
        array.each do |x|
          return actual_type_of(x) if !x.nil?
        end
        
        'float64'
      end

      def convert_xnd_t_to_ruby_array data
        if data.is_a?(XND::T)
          data.map! do |d|
            convert_xnd_t_to_ruby_array d
          end

          return data.data
        elsif data.is_a? Hash
          data.each do |k, v|
            data[k] = convert_xnd_t_to_ruby_array v
          end
        elsif data.is_a? Array
          data.map! do |d|
            convert_xnd_t_to_ruby_array d
          end          
        else
          return data
        end
      end
    end
  end
  
  def initialize data, type: nil, dtype: nil, levels: nil, typedef: nil, dtypedef: nil
    if [type, dtype, levels, typedef, dtypedef].count(nil) < 2
      raise ArgumentError, "the 'type', 'dtype', 'levels' and 'typedef' arguments are "
      "mutually exclusive."
    end

    if type
      type = NDTypes.new(type) if type.is_a? String
    elsif dtype
      type = TypeInference.type_of data, dtype: dtype
    elsif levels
      args = levels.map { |l| l ? l : 'NA' }.join(', ')
      t = "#{value.size} * categorical(#{args})"
      type = NDTypes.new t
    elsif typedef
      type = NDTypes.new typedef
      if type.abstract?
        dtype = type.hidden_dtype
        t = TypeInference.type_of data, dtype: dtype
        type = NDTypes.instantiate typedef, t
      end
    elsif dtypedef
      dtype = NDTypes.new dtypedef
      type = TypeInference.type_of data, dtype: dtype
    else
      type = TypeInference.type_of data
      data = TypeInference.convert_xnd_t_to_ruby_array data
    end

    super(type, data)
  end

  alias :to_a :value
end
