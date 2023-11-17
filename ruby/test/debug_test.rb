require 'test_helper'

class TestXND_DEBUG < Minitest::Test
  def test_nil_value
    assert_equal RubyXND::XND_DEBUG, nil
  end
end
