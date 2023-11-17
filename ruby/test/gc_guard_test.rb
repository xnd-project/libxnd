require 'test_helper'

class TestGCGuard < Minitest::Test
  def test_gc_table_value
    xnd = XND.new([[1,2,3]])

    gc_table = RubyXND::GCGuard.instance_variable_get(:@__gc_guard_table)
    assert gc_table.keys.size >= 1
  end
end
