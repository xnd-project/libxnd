require 'spec_helper'

describe XND::TypeInference do
  MAX_DIM = NDTypes::MAX_DIM

  context ".accumulate" do
    it "returns accumulated of sum of an Array" do
      arr = [1,2,3,4,5]
      result = [1,3,6,10,15]

      expect(XND::TypeInference.accumulate(arr)).to eq(result)
    end
  end
  
  context ".search" do
    it "searches for data and shape and loads in acc Array" do
      data = [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
      result = [[3], [2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8], *(Array.new(MAX_DIM-2) { [] })]
      
      min_level = MAX_DIM + 1
      max_level = 0
      acc = Array.new(MAX_DIM + 1) { [] }
      minmax = [min_level, max_level]

      XND::TypeInference.search max_level, data, acc, minmax

      expect(acc).to eq(result)
      expect(minmax[0]).to eq(minmax[1])
    end
  end
  
  context ".data_shapes" do
    it "extracts the shape of nested Array data" do
      data = [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
      result = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [[2, 3, 4], [3]]]

      expect(XND::TypeInference.data_shapes(data)).to eq(result)
    end

    it "works for empty array" do
      data = []
      result = [[], [[0]]]

      expect(XND::TypeInference.data_shapes(data)).to eq(result)
    end

    it "works for empty nested array" do
      data = [[]]
      result = [[], [[0], [1]]]

      expect(XND::TypeInference.data_shapes(data)).to eq(result)
    end
  end

  context ".add_dim" do
    it "calculates dimension" do
      
    end
  end
  
  context ".type_of" do
    it "generates correct ndtype for fixed array" do
      value = [
        [1,2,3],
        [5,6,7]
      ]
      type = NDTypes.new "2 * 3 * int64"

      expect(XND::TypeInference.type_of(value)).to eq(type)
    end

    it "generates correct ndtype for hash" do
      value = {
        "a" => "xyz",
        "b" => [1,2,3]
      }
      type = NDTypes.new "{a : string, b : 3 * int64}"
      expect(XND::TypeInference.type_of(value)).to eq(type)
    end
  end
end
