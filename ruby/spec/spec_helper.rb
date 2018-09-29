require 'xnd'
require 'pry'

def expect_strict_equal x1, x2
  expect(x1.strict_equal(x2)).to eq(true)
  expect(x1).to eq(x2)
end

def expect_strict_unequal x1, x2
  expect(x1.strict_equal(x2)).to eq(false)
  expect(x1).not_to eq(x2)
end

def expect_with_exception func, x, y
  return if (x.value.nil? || y.nil?) #|| (x.value&.abs == 0 && y&.abs == 0)

  xerr = nil
  begin
    xres = x.send(func)
  rescue StandardError => e
    xerr = e.class
  end

  yerr = nil
  begin
    yres = y.send(func)
  rescue StandardError => e
    yerr = e.class
  end

  if xerr.nil? && yerr.nil?
    expect(xres).to eq(yres)
  else
    expect(xerr).to eq(yerr)
  end
end

