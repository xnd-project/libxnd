
AC_PREREQ([2.67])
AC_INIT(xnd, 0.1.0, plures@anaconda.com, xnd, https://github.com/plures/)
AC_CONFIG_HEADER(config.h)
AC_CONFIG_FILES([Makefile libxnd/Makefile libxnd/tests/Makefile])

# System and machine type (only used for Darwin):
AC_MSG_CHECKING(system as reported by uname -s)
ac_sys_system=`uname -s`
AC_MSG_RESULT($ac_sys_system)

LIBSTATIC=libxnd.a
case $ac_sys_system in
    darwin*|Darwin*)
        LIBRARY_PATH="DYLD_LIBRARY_PATH"
        LIBNAME="libxnd.dylib"
        LIBSONAME="libxnd.0.dylib"
        LIBSHARED="libxnd.0.1.0.dylib"
        CONFIGURE_LDFLAGS="-dynamiclib -install_name @rpath/$LIBSONAME -undefined dynamic_lookup -compatibility_version 0.1 -current_version 0.1.0"
        ;;
    *)
        LIBRARY_PATH="LD_LIBRARY_PATH"
        LIBNAME="libxnd.so"
        LIBSONAME="libxnd.so.0"
        LIBSHARED="libxnd.so.0.1.0"
        CONFIGURE_LDFLAGS="-shared -Wl,-soname,$LIBSONAME"
        ;;
esac

AC_SUBST(LIBSTATIC)
AC_SUBST(LIBRARY_PATH)
AC_SUBST(LIBNAME)
AC_SUBST(LIBSONAME)
AC_SUBST(LIBSHARED)


# Apparently purely informational for this particular build:
AC_CANONICAL_HOST
AC_SUBST(build)
AC_SUBST(host)

# Language and compiler:
AC_LANG_C
saved_cflags=$CFLAGS
AC_PROG_CC
CFLAGS=$saved_cflags

# ar and ranlib:
AC_CHECK_TOOL(AR, ar, ar)
AC_PROG_RANLIB
AC_SUBST(RANLIB)


# Checks for header files:
AC_HEADER_STDC

# Install program:
AC_PROG_INSTALL
AC_SUBST(INSTALL)


# Set the prefix of a local ndtypes directory.  If unset, libndtypes is
# assumed to be installed in the system library directories.
AC_MSG_CHECKING(for --with-ndtypes)
AC_ARG_WITH(ndtypes,
            AS_HELP_STRING([--with-ndtypes=".."],
            [relative path to a local ndtypes directory]),
            [],
            [with_ndtypes="none"])

AC_MSG_RESULT($with_ndtypes)

if test "$with_ndtypes" = "none"; then
    LIBNDTYPESDIR=
else
    LIBNDTYPESDIR=$with_ndtypes/ndtypes/libndtypes
fi

AC_SUBST(LIBNDTYPESDIR)


# Compiler dependent settings:
XND_WARN=
XND_OPT="-O2"
case $CC in
    *gcc*)
        XND_WARN="-Wall -Wextra -std=c11 -pedantic"
        XND_OPT="-O2"
        ;;
    *icc*)
        AR=xiar
        XND_WARN="-Wall"
        XND_OPT="-O2"
        ;;
    *clang*)
        XND_WARN="-Wall -Wextra -std=c11 -pedantic"
        XND_OPT="-O2"
        ;;
esac



# Substitute variables and generate output:
if test -z "$LD"; then
    LD="$CC"
fi
AC_SUBST(LD)
AC_SUBST(AR)
AC_SUBST(XND_WARN)
AC_SUBST(XND_OPT)


if test -z "$CFLAGS"; then
    CONFIGURE_CFLAGS="$XND_INCLUDE $XND_WARN $XND_CONFIG $XND_OPT"
else
    CONFIGURE_CFLAGS="$XND_INCLUDE $XND_WARN $XND_CONFIG $XND_OPT $CFLAGS"
fi

if test -z "$LDFLAGS"; then
    CONFIGURE_LDFLAGS="$XND_LINK $CONFIGURE_LDFLAGS"
else
    CONFIGURE_LDFLAGS="$XND_LINK $CONFIGURE_LDFLAGS $LDFLAGS"
fi

AC_SUBST(CONFIGURE_CFLAGS)
AC_SUBST(CONFIGURE_LDFLAGS)

AC_OUTPUT

