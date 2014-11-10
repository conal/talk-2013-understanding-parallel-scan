# Understanding efficient parallel scan

I first gave this talk [at the Hacker Dojo in Mountain View, California on October 24, 2013](http://www.meetup.com/haskellhackersathackerdojo/events/132372202/).

You can find [the slides (PDF)](http://conal.net/talks/understanding-parallel-scan.pdf) in [my talks folder](http://conal.net/talks/).

## Abstract

Abstract:

This talk will demonstrate high-level algorithm design for parallel functional programming, by means of the example of (parallel) prefix computation---known as "scan" to functional programmers. We'll start with a simple and precise scan specification that leads directly to an algorithm performing quadratic work and hence quadratic time in a sequential implementation. An easy observation leads to a familiar sequential algorithm performing linear work. This version, however, does not lend itself to parallel execution. A conventional divide-and-conquer idea leads to a functional algorithm that can execute in parallel O (log n) time (given sufficient computational resources) and O (n log n) work. Playing with variations and poking at asymmetries, a beautiful generalization reveals itself. An inversion of representation then leads to a linear work algorithm while retaining logarithmic time, without changing the scan source code. This inversion yields an example of non-regular (or "nested") algebraic data types, which are relatively obscure even in the functional programming community. (I suspect that these non-regular types are often more parallel-friendly than their familiar counterparts, as well as naturally capturing some shape/size constraints.) Yet another modest variation leads to O (n log log n) work and O (log log n) time. These three parallel algorithms can be decomposed into five simple pieces, which can be systematically recomposed to automatically generate parallel scan algorithms for a broad range of data structures.
