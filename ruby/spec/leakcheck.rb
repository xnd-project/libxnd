require_relative 'spec_helper'

100.times do
  # basic XND object
  XND.new([[1,2,3], [4,5,6]])
end

GC.start
          
