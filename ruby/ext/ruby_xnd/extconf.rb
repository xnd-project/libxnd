require 'mkmf'

def windows?
  (/cygwin|mswin|mingw|bccwin|wince|emx/ =~ RUBY_PLATFORM) != nil
end

def mac?
  (/darwin/ =~ RUBY_PLATFORM) != nil
end

def unix?
  !windows?
end

# libndtypes config

ndtypes_version = ">= 0.2.0dev5"
ndtypes_spec = Gem::Specification.find_by_name("ndtypes", ndtypes_version)
ndtypes_extdir = File.join(ndtypes_spec.gem_dir, 'ext', 'ruby_ndtypes')
ndtypes_includedir = File.join(ndtypes_extdir, 'include')
ndtypes_libdir = File.join(ndtypes_extdir, 'lib')

find_header("ruby_ndtypes.h", ndtypes_includedir)
raise "cannot find ruby_ndtypes.h in path #{ndtypes_includedir}." unless have_header("ruby_ndtypes.h")

find_header("ndtypes.h", ndtypes_includedir)
find_library("ndtypes", nil, ndtypes_libdir)

dir_config("ndtypes", [ndtypes_includedir], [ndtypes_libdir])

# libxnd config

puts "compiling libxnd for your machine..."
Dir.chdir(File.join(File.dirname(__FILE__) + "/xnd")) do
  if unix?
    Dir.chdir("libxnd") do
      Dir.mkdir(".objs") unless Dir.exists? ".objs"
    end
    
    system("./configure --prefix=#{File.expand_path("../")} --with-docs=no " +
           "--with-includes=#{ndtypes_includedir}")
    system("make")
    system("make install")
  elsif windows?
    raise NotImplementedError, "need to specify build instructions for windows."
  end
end

binaries = File.expand_path(File.join(File.dirname(__FILE__) + "/lib/"))
headers = File.expand_path(File.join(File.dirname(__FILE__) + "/include/"))

find_library("xnd", nil, binaries)
find_header("xnd.h", headers)

FileUtils.copy_file File.expand_path(File.join(File.dirname(__FILE__) +
                                               "/ruby_xnd.h")),
                    "#{headers}/ruby_xnd.h"

dir_config("xnd", [headers], [binaries])

$INSTALLFILES = [
  ["ruby_xnd.h", "$(archdir)"],
  ["xnd.h", "$(archdir)"]
]

# for macOS
append_ldflags("-Wl,-rpath #{binaries}")

basenames = %w{util float_pack_unpack gc_guard ruby_xnd}
$objs = basenames.map { |b| "#{b}.o"   }
$srcs = basenames.map { |b| "#{b}.c" }

$CFLAGS += " -fPIC -g "
create_makefile("ruby_xnd/ruby_xnd")
