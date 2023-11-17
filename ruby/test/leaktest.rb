# Run each Test class N times and start the GC after calling the test
require 'xnd'
require_relative 'test_xnd.rb'

# ObjectSpace.each_object(Class).map(&:to_s).grep(/TestVarDim/).each do |t|
#   klass = Kernel.const_get(t)
#   methods = klass.runnable_methods

#   100.times do
#     methods.each do |m|
#       Minitest.run ['-n', m]
#     end
#   end

#   GC.start
# end

  100.times do
    # methods.each do |m|
    #Minitest.run ['-n', 'test_var_dim_empty']
    Minitest.run ['-n', 'test_fixed_bytes_equality']
    #end
  end

  GC.start
