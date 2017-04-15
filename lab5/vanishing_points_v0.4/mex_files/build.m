% build mex
mex -v  -I/usr/include lsd.c
%mex -v  CFLAGS='-D_GNU_SOURCE -fexceptions -fno-omit-frame-pointer -pthread -ansi -fPIC -fopenmp' LDFLAGS='-pthread -shared -fopenmp' -I/usr/include -Ilib lib/alignments.c lib/ntuple.c  lib/misc.c lib/ntuples_aux.c  lib/nfa.c  alignments_slow.c -output alignments_slow
setenv(CFLAGS, '-D_GNU_SOURCE -fexceptions -fno-omit-frame-pointer -pthread -ansi -fPIC -fopenmp')
setenv(LDFLAGS, '-pthread -shared -fopenmp')
mkoctfile --mex -v  -I/usr/include -Ilib lib/alignments.c lib/ntuple.c  lib/misc.c lib/ntuples_aux.c  lib/nfa.c  alignments_slow.c -output alignments_slow
%mex -v  CFLAGS='-D_GNU_SOURCE -fexceptions -fno-omit-frame-pointer -pthread  -fPIC -fopenmp' LDFLAGS='-pthread -shared -fopenmp' -I/usr/include -Ilib lib/alignments.c lib/ntuple.c   lib/misc.c lib/ntuples_aux.c  lib/nfa.c  alignments_fast.c -output alignments_fast
setenv(CFLAGS, '-D_GNU_SOURCE -fexceptions -fno-omit-frame-pointer -pthread  -fPIC -fopenmp')
setenv(LDFLAGS, '-pthread -shared -fopenmp')
mkoctfile --mex -v    -I/usr/include -Ilib lib/alignments.c lib/ntuple.c   lib/misc.c lib/ntuples_aux.c  lib/nfa.c  alignments_fast.c -output alignments_fast


