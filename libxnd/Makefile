
# ==============================================================================
#                          Unix Makefile for libxnd
# ==============================================================================


LIBSTATIC = libxnd.a
LIBNAME = libxnd.so
LIBSONAME = libxnd.so.0
LIBSHARED = libxnd.so.0.1.0

CC = gcc-4.9
LD = gcc-4.9
AR = ar
RANLIB = ranlib
XND_PGEN = -fprofile-generate -fprofile-values
XND_PUSE = -fprofile-use -freorder-blocks

LIBNDTYPESDIR = ./ndtypes/libndtypes
ifdef LIBNDTYPESDIR
XND_INCLUDES = -I../$(LIBNDTYPESDIR)
endif

CONFIGURE_CFLAGS =  -Wall -Wextra -std=c11 -pedantic  -O2
XND_CFLAGS = $(strip $(XND_INCLUDES) $(CONFIGURE_CFLAGS) $(CFLAGS))

CONFIGURE_LDFLAGS =  -shared -Wl,-soname,libxnd.so.0
XND_LDFLAGS = $(strip $(CONFIGURE_LDFLAGS) $(LDFLAGS))

ifeq ($(MAKECMDGOALS), profile_gen)
  XND_CFLAGS += $(XND_PGEN)
  XND_LDFLAGS += $(XND_PGEN)
endif
ifeq ($(MAKECMDGOALS), profile_use)
  XND_CFLAGS += $(XND_PUSE)
  XND_LDFLAGS += $(XND_PUSE)
endif

XND_CFLAGS_SHARED = $(XND_CFLAGS) -fPIC


default: $(LIBSTATIC) $(LIBSHARED)

OBJS = xnd.o

SHARED_OBJS = .objs/xnd.o


$(LIBSTATIC): Makefile $(OBJS)
	$(AR) rc $(LIBSTATIC) $(OBJS)
	$(RANLIB) $(LIBSTATIC)

$(LIBSHARED): Makefile $(SHARED_OBJS)
	$(LD) $(XND_LDFLAGS) -o $(LIBSHARED) $(SHARED_OBJS)
	ln -sf $(LIBSHARED) $(LIBNAME)
	ln -sf $(LIBSHARED) $(LIBSONAME)

xnd.o:\
Makefile xnd.c xnd.h
	$(CC) $(XND_CFLAGS) -c xnd.c

.objs/xnd.o:\
Makefile xnd.c xnd.h
	$(CC) $(XND_CFLAGS_SHARED) -c xnd.c -o .objs/xnd.o


# Coverage
coverage:\
Makefile clean runtest
	./tests/runtest
	for file in *.c; do gcov -l "$$file" > /dev/null 2>&1; done

FORCE:

clean: FORCE
	rm -f *.o *.so *.gch *.gcda *.gcno *.gcov *.dyn *.dpi *.lock
	rm -f $(LIBSTATIC) $(LIBSHARED) $(LIBSONAME) $(LIBNAME)
	cd .objs && rm -f *.o *.so *.gch *.gcda *.gcno *.gcov *.dyn *.dpi *.lock

distclean: clean
	rm -f Makefile



