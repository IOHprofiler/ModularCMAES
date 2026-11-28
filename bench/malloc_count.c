/*
 * Counts heap allocations (malloc, calloc, realloc, posix_memalign, aligned_alloc) for the benchmark
 * harness. Eigen allocates through malloc directly, so counting operator new is not enough.
 *
 * macOS: built as a dylib that the harness links. dyld applies the __interpose tuples of every image
 * loaded at launch, so all allocations of the process go through these wrappers.
 * Linux (glibc): compiled into the executable, which then defines malloc & co. itself and forwards to
 * glibc's __libc_* implementations.
 * Elsewhere: modcma_malloc_count_supported() returns 0 and the harness prints n/a (no wrappers and
 * no GCC atomics, which MSVC does not have).
 */
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__APPLE__) || defined(__GLIBC__)

static unsigned long long n_allocations = 0;

static void count_allocation(void) { __atomic_add_fetch(&n_allocations, 1, __ATOMIC_RELAXED); }

int modcma_malloc_count_supported(void) { return 1; }

unsigned long long modcma_malloc_count(void)
{
    return __atomic_load_n(&n_allocations, __ATOMIC_RELAXED);
}

#endif

#if defined(__APPLE__)

static void *count_malloc(size_t n)
{
    count_allocation();
    return malloc(n);
}
static void *count_calloc(size_t c, size_t n)
{
    count_allocation();
    return calloc(c, n);
}
static void *count_realloc(void *p, size_t n)
{
    count_allocation();
    return realloc(p, n);
}
static int count_posix_memalign(void **p, size_t a, size_t n)
{
    count_allocation();
    return posix_memalign(p, a, n);
}
static void *count_aligned_alloc(size_t a, size_t n)
{
    count_allocation();
    return aligned_alloc(a, n);
}

typedef struct
{
    const void *replacement;
    const void *original;
} interpose_t;

__attribute__((used)) static const interpose_t interposers[] __attribute__((section("__DATA,__interpose"))) = {
    {(const void *)count_malloc, (const void *)malloc},
    {(const void *)count_calloc, (const void *)calloc},
    {(const void *)count_realloc, (const void *)realloc},
    {(const void *)count_posix_memalign, (const void *)posix_memalign},
    {(const void *)count_aligned_alloc, (const void *)aligned_alloc},
};

#elif defined(__GLIBC__)

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);

void *malloc(size_t n)
{
    count_allocation();
    return __libc_malloc(n);
}
void *calloc(size_t c, size_t n)
{
    count_allocation();
    return __libc_calloc(c, n);
}
void *realloc(void *p, size_t n)
{
    count_allocation();
    return __libc_realloc(p, n);
}
int posix_memalign(void **p, size_t a, size_t n)
{
    count_allocation();
    *p = __libc_memalign(a, n);
    return *p ? 0 : ENOMEM;
}
void *aligned_alloc(size_t a, size_t n)
{
    count_allocation();
    return __libc_memalign(a, n);
}

#else

int modcma_malloc_count_supported(void) { return 0; }

unsigned long long modcma_malloc_count(void) { return 0; }

#endif
