//! Running independent pieces of work on all cores when the `parallel` feature is on.

/// `(0..n).map(f).collect()`, in parallel when there are enough items to be
/// worth it. For many small pieces of work, such as one per star.
pub(crate) fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if n >= 8 {
        return map_few(n, f);
    }
    (0..n).map(f).collect()
}

/// As [`map`], but in parallel however few items there are. For a handful of
/// large pieces of work, such as one per image plane.
pub(crate) fn map_few<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Passing `f` itself would require it to be `Send`; calling it through a reference only needs `Sync`.
        #[allow(clippy::redundant_closure)]
        let out = (0..n).into_par_iter().map(|i| f(i)).collect();
        out
    }
    #[cfg(not(feature = "parallel"))]
    (0..n).map(f).collect()
}

/// Call `f(index, chunk)` for each `size` items of `data`, on all cores.
pub(crate) fn chunks<T: Send>(data: &mut [T], size: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(size).enumerate().for_each(|(i, chunk)| f(i, chunk));
    }
    #[cfg(not(feature = "parallel"))]
    data.chunks_mut(size).enumerate().for_each(|(i, chunk)| f(i, chunk));
}
