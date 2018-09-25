require 'spec_helper.rb'

describe RubyXND do
  context "::XND_DEBUG" do
    it "is set to nil before release" do
      expect(RubyXND::XND_DEBUG).to eq(nil)
    end
  end
end
