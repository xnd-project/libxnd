
# ==============================================================================
#                            Unix Makefile for libxnd
# ==============================================================================

LIBSTATIC = libxnd.a
LIBNDTYPESDIR = libndtypes

CC ?= gcc
LD ?= gcc
AR ?= ar
RANLIB ?= ranlib
CFLAGS ?= -Wall -Wextra -std=c11 -pedantic -O2 -g

ifeq ($(MAKECMDGOALS), check)
  CFLAGS = -Wall -Wextra -Werror -std=c11 -pedantic -O2 -g
endif

ifeq ($(MAKECMDGOALS), memcheck)
  CFLAGS = -Wall -Wextra -Werror -std=c11 -pedantic -O2 -g
endif

ifeq ($(MAKECMDGOALS), coverage)
  CFLAGS = -Wall -Wextra -std=c11 -pedantic -O0 -g -fno-inline -fprofile-arcs -ftest-coverage -fpic
  LDFLAGS = -fprofile-arcs
endif


default: $(LIBSTATIC)


OBJS = xnd.o


$(LIBSTATIC):\
Makefile $(OBJS)
	$(AR) rc $(LIBSTATIC) $(OBJS)
	$(RANLIB) $(LIBSTATIC)

xnd.o:\
Makefile xnd.c xnd.h
	$(CC) -I $(LIBNDTYPESDIR) $(CFLAGS) -c xnd.c


# Tests
runtest:\
Makefile tests/runtest.c tests/test_fixed.c tests/test_var.c tests/test.h $(LIBSTATIC)
	$(CC) -I. -I $(LIBNDTYPESDIR) -L $(LIBNDTYPESDIR) $(CFLAGS) -DTEST_ALLOC \
	-o tests/runtest tests/runtest.c tests/test_fixed.c tests/test_var.c $(LIBSTATIC) libndtypes/libndtypes.a

check:\
Makefile runtest
	./tests/runtest

memcheck:\
Makefile runtest
	valgrind --leak-check=full --show-leak-kinds=all ./tests/runtest


# Coverage
coverage:\
Makefile clean runtest
	./tests/runtest
	for file in *.c; do gcov -l "$$file" > /dev/null 2>&1; done


clean: FORCE
	rm -f *.o *.gch *.gcov *.gcda *.gcno tests/runtest $(LIBSTATIC)

distclean: clean
	rm -rf libndtypes


FORCE:


