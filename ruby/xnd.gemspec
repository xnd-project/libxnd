# coding: utf-8
$:.unshift File.expand_path("../lib", __FILE__)

require 'xnd/version.rb'

def self.get_files
  files = []
  ['ext', 'lib', 'spec'].each do |folder|
    files.concat Dir.glob "#{folder}/**/*"
  end

  files.concat(
    ["CONTRIBUTING.md", "Gemfile", "History.md", "xnd.gemspec",
     "README.md", "Rakefile"
    ])
  
  files
end
files = get_files

RubyXND::DESCRIPTION = <<MSG
XND is a library for typed data arrays in Ruby. It is a wrapper over the libxnd C library.
MSG

Gem::Specification.new do |spec|
  spec.name          = 'xnd'
  spec.version       = RubyXND::VERSION
  spec.authors       = ['Sameer Deshmukh']
  spec.email         = ['sameer.deshmukh93@gmail.com']
  spec.summary       = %q{Ruby wrapper over libxnd. A library for typed data containers.}
  spec.description   = RubyXND::DESCRIPTION
  spec.homepage      = "https://github.com/plures/xnd"
  spec.license       = 'BSD-3 Clause'

  spec.files         = files
  spec.extensions    = "ext/ruby_xnd/extconf.rb"
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.require_paths = ["lib"]

  spec.add_development_dependency 'minitest', '~> 5.11'
  spec.add_development_dependency 'minitest-hooks'
  spec.add_development_dependency 'rake-compiler'
  spec.add_development_dependency 'pry'
  spec.add_development_dependency 'pry-byebug'

  spec.add_runtime_dependency 'ndtypes', '>= 0.2.0dev5'
end
