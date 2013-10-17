%% -*- latex -*-

%% %let atwork = True

% Presentation
\documentclass{beamer}

%% % Printed, 2-up
%% \documentclass[serif,handout]{beamer}
%% \usepackage{pgfpages}
%% \pgfpagesuselayout{2 on 1}[border shrink=1mm]

%% % Printed, 4-up
%% \documentclass[serif,handout,landscape]{beamer}
%% \usepackage{pgfpages}
%% \pgfpagesuselayout{4 on 1}[border shrink=1mm]

\usefonttheme{serif}

\usepackage{beamerthemesplit}

%% % http://www.latex-community.org/forum/viewtopic.php?f=44&t=16603
%% \makeatletter
%% \def\verbatim{\small\@verbatim \frenchspacing\@vobeyspaces \@xverbatim}
%% \makeatother

\usepackage{graphicx}
\usepackage{color}
\DeclareGraphicsExtensions{.pdf,.png,.jpg}

\useinnertheme[shadow]{rounded}
% \useoutertheme{default}
\useoutertheme{shadow}
\useoutertheme{infolines}
% Suppress navigation arrows
\setbeamertemplate{navigation symbols}{}

\input{macros}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include mine.fmt

%if atwork
% \title{High-level algorithm design for reschedulable computation, Part 1}
\title{High-level algorithm design\\ for reschedulable computation}
\subtitle{Part 1: Efficient parallel scan} % Understanding 
%else
\title{Understanding efficient parallel scan} % 
%endif
\author{\href{http://conal.net}{Conal Elliott}}
\institute{\href{http://tabula.com/}{Tabula}}
% Abbreviate date/venue to fit in infolines space
%% \date{\href{http://www.meetup.com/haskellhackersathackerdojo/events/105583982/}{March 21, 2013}}
\date{October, 2013}

\setlength{\itemsep}{2ex}
\setlength{\parskip}{1ex}

\setlength{\blanklineskip}{1.5ex}

% \nc\half[1]{\frac{#1}{2}}
\nc\half[1]{{#1}/2}
\nc\cl{c_l}
\nc\ch{c_h}

% \nc\tboxed[2]{{\boxed{#1}\,}^{#2}}
% \nc\tvox[2]{\tboxed{\rule{0pt}{2ex}#1}{#2}}
% \nc\vox[1]{\tvox{#1}{}}

\nc\bboxed[1]{\boxed{\rule[-0.9ex]{0pt}{2.8ex}#1}}
\nc\vox[1]{\bboxed{#1}}
\nc\tvox[2]{\vox{#1}\vox{#2}}

\nc\lscan{\Varid{lscan}}

\nc\trans[1]{\\[1.3ex] #1 \\[0.75ex]}
\nc\ptrans[1]{\pause\trans{#1}}
\nc\ptransp[1]{\ptrans{#1}\pause}

\nc\pitem{\pause \item}

%%%%

% \setbeameroption{show notes} % un-comment to see the notes

\begin{document}

\frame{\titlepage}

\title{Efficient parallel scan}

\framet{Prefix sum (left scan)}{
\begin{center}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n}
\trans{\lscan\Downarrow}
\tvox{b_1, \ldots, b_n}{b_{n+1}}
\end{array}
\]
\end{minipage}
where
\begin{minipage}[c]{0.3\textwidth}
\[ b_k = \sum\limits_{1 \le i < k}{a_i} \]
\end{minipage}
\end{center}
}

\framet{In CUDA C}{
\begin{minipage}[c]{0.7\textwidth}
\tiny
\begin{verbatim}
__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    // load input into shared memory
    temp[2*thid] = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];
    // build sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai]; }
        offset *= 2; }
    // clear the last element
    if (thid == 0) { temp[n - 1] = 0; }
    // traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; } }
    __syncthreads();
    // write results to device memory
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1]; }
\end{verbatim}
\vspace{-6ex}
\href{http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html}{\emph{Source: GPU Gems 3, Chapter 39}}
\normalsize
\end{minipage}
\hspace{-1in}
\begin{minipage}[c]{0.25\textwidth}
\pause
\begin{figure}
\wpicture{2in}{ShadowedPictures/beaker-looks-left}

\pause\hspace{0.75in}\emph{WAT}
\end{figure}
\end{minipage}
}

\framet{Prefix sum (left scan)}{
\begin{center}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n}
\trans{\lscan\Downarrow}
\tvox{b_1, \ldots, b_n}{b_{n+1}}
\end{array}
\]
\end{minipage}
where
\begin{minipage}[c]{0.3\textwidth}
\[ b_k = \sum\limits_{1 \le i < k}{a_i} \]
\end{minipage}

\end{center}

\vspace{2ex}\pause
\emph{Work:} $O(n^2)$.

\pause
\emph{Time:} \pause $O(n^2)$, $O(n)$, $O(\log n)$.

\vspace{7.7ex}
}

\framet{As a recurrence}{
\begin{center}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n}
\trans{\lscan\Downarrow}
\tvox{b_1, \ldots, b_n}{b_{n+1}}
\end{array}
\]
\end{minipage}
where
\begin{minipage}[c]{0.3\textwidth}
\begin{align*}
b_1 &= 0 \\
b_{k+1} &= b_k + a_k
\end{align*}
\end{minipage}
\end{center}

\vspace{2ex} \pause
\emph{Work:} $O(n)$.

\pause
\emph{Depth} (ideal parallel ``time''): $O(n)$.

\ 

\pause
Linear \emph{dependency chain} thwarts parallelism (depth $<$ work).
}

\nc\arr[1]{\Downarrow _{\text{\makebox[0pt][l]{\emph{#1}}}}}

\framet{Divide and conquer}{
\pause
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n, a'_1, \ldots, a'_n}
\ptransp{\arr{split}}
\vox{a_1, \ldots, a_n}
\ 
\vox{a'_1, \ldots, a'_n}
\ptransp{\arr{lscan} \hspace{12ex} \arr{lscan}}
\tvox{b_1, \ldots, b_n}{b_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
\ptransp{\arr{merge}}
\tvox{b_1, \ldots, b_n, b_{n+1} + b'_1, \ldots, b_{n+1} + b'_n}{b_{n+1}+b'_{n+1}}
\end{array}
\]

\begin{itemize}
\pitem Equivalent? Why?
       \pause (Associativity.)
\pitem No more linear dependency chain.
\pitem Work and depth analysis?
\end{itemize}
}

\framet{Depth analysis}{
Depends on cost of splitting and merging.
\begin{itemize}
\pitem Constant:
 \begin{align*}
 D(n) &= D(n/2) + O(1) \\
 D(n) &= O(\log n)
 \end{align*}

\pitem Linear:
 \begin{align*}
  D(n) &= D(n/2) + O(n) \\
  D(2^k) &= O (1 + 2 + 4 + \cdots + 2^k) = O(2^k) \\
  D(n) &= O(n)
 \end{align*}

\pitem Logarithmic:
 \begin{align*}
  D(n) &= D(n/2) + O(\log n) \\
  D(2^k) &= O(0 + 1 + 2 + \cdots + k) = O(k^2) \\
  D(n) &= O(\log^2 n)
 \end{align*}
\end{itemize}
}

\framet{Work analysis}{
Work recurrence:

\[ W(n) = 2 \, W(n/2) + O(n) \]

\vspace{4ex}

\pause
By the \href{http://en.wikipedia.org/wiki/Master_theorem}{\emph{Master Theorem}},
\[ W(n) = O(n \, \log n) \]
}

\framet{Analysis summary}{
Sequential:
\begin{align*}
 D(n) &= O(n) \\
 W(n) &= O(n)
\end{align*}

\ \pause

Divide and conquer:
\begin{align*}
 D(n) &= O(\log n) \\
 W(n) &= O(n \, \log n)
\end{align*}

\vspace{3ex}\pause Challenge: can we get $O(n)$ work and $O(\log n)$ depth?
}

\nc\case[2]{#2 & \text{if~} #1 \\}
\nc\mtCase[2]{\case{a #1 b^d}{O(#2)}}

\framet{Master Theorem}{
Given a recurrence:
\[ f(n) = a \, f(n/b) + O(n^d) \]
We have the following closed form bound:
\[ 
f(n) = \begin{cases}
 \mtCase{<}{n^d}
 \mtCase{=}{n^d \, \log n}
 \mtCase{>}{n^{\log_b a}}
\end{cases}
\]
\ 
\vspace{15.8ex} % to align with next slide
}

\nc\mtCaseo[2]{\case{a #1 b}{O(#2)}}

\framet{Master Theorem ($d=1$)}{
Given a recurrence:
\[ f(n) = a \, f(n/b) + O(n) \]
We have the following closed form bound:
\[ 
f(n) = \begin{cases}
 \mtCaseo{<}{n}
 \mtCaseo{=}{n \, \log n}
 \mtCaseo{>}{n^{\log_b a}}
\end{cases}
\]

\ \pause

\emph{Puzzle:} how to get $a < b$ for our recurrence?
\[ W(n) = 2 \, W(n/2) + O(n) \]
Return to this question later.
}

\hidden{
\framet{Aesthetic quibble}{
The divide-and-conquer algorithm:
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n, b_1, \ldots, b_n}
\trans{\arr{split}}
\vox{a_1, \ldots, a_n}
\ 
\vox{b_1, \ldots, b_n}
\trans{\arr{lscan} \hspace{12ex} \arr{lscan}}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
\trans{\arr{merge}}
\tvox{a'_1, \ldots, a'_n, b''_1, \ldots, b''_n}{b''_{n+1}}
\end{array}
\]
where
\[ b''_i = a'_{n+1}+b'_i \]

\ \pause
Note the asymmetry: adjust the $b'_i$ but not the $a'_i$.
}
}

\framet{Variation: $3$-way split/merge}{
\vspace{-1ex}
\[
\begin{array}{c}
\vox{a_{1,1}, \ldots, a_{1,m}, a_{2,1}, \ldots, a_{2,m}, a_{3,1}, \ldots, a_{3,m}}
\ptrans{\arr{split}}
\vox{a_{1,1}, \ldots, a_{1,m}}
\ 
\vox{a_{2,1}, \ldots, a_{2,m}}
\ 
\vox{a_{3,1}, \ldots, a_{3,m}}
\ptrans{\arr{lscan} \hspace{12ex} \arr{lscan} \hspace{12ex} \arr{lscan}}
\tvox{b_{1,1}, \ldots, b_{1,m}}{b_{1,m+1}}
\ 
\tvox{b_{2,1}, \ldots, b_{2,m}}{b_{2,m+1}}
\ 
\tvox{b_{3,1}, \ldots, b_{3,m}}{b_{3,m+1}}
\ptrans{\arr{merge}}
\tvox{d_{1,1}, \ldots, d_{1,m} , d_{2,1}, \ldots, d_{2,m}, d_{3,1}, \ldots, d_{3,m}}{d_{3,m+1}}
\end{array}
\]
where
\begin{align*}
d_{1,j}&= b_{1,j} \\
d_{2,j}&= b_{1,m+1}+b_{2,j} \\
d_{3,j}&= b_{1,m+1}+b_{2,m+1}+b_{3,j} \\
\end{align*}
}

\hidden{
\framet{Multi-way merge}{
\[
\begin{array}{c}
\tvox{b_{1,1}, \ldots, b_{1,n}}{b_{1,n+1}}
\ 
\tvox{b_{2,1}, \ldots, b_{2,n}}{b_{2,n+1}}
\ 
\tvox{b_{3,1}, \ldots, b_{3,n}}{b_{3,n+1}}
\ptrans{\arr{merge}}
\tvox{c_{1,1}, \ldots, c_{1,n} , c_{2,1}, \ldots, c_{2,n}, c_{3,1}, \ldots, c_{3,n}}{c_{3,n+1}}
\end{array}
\]
where
\begin{align*}
c^1_i&= b^1_i \\
c^2_i&= b^1_{n+1}+b^2_i \\
c^3_i&= b^1_{n+1}+b^2_{n+1}+b^3_i \\
\end{align*}

\pause
\vspace{-3ex} % Why??
\emph{Where have we seen this pattern of shifts?}
}
}

\nc\sdots[1]{\hspace{#1} \ldots \hspace{#1}}

\framet{Variation: $k$-way split/merge}{
\[
\begin{array}{c}
\vox{a_{1,1}, \ldots, a_{1,m},  \ldots,  a_{k,1}, \ldots, a_{k,m}}
\ptrans{\arr{split}}
\vox{a_{1,1}, \ldots, a_{1,m}} \sdots{3ex} \vox{a_{k,1}, \ldots, a_{k,m}}
\ptrans{\arr{lscan} \sdots{10ex} \arr{lscan}}
\tvox{b_{1,1}, \ldots, b_{1,m}}{b_{1,m+1}} \sdots{3ex} \tvox{b_{k,1}, \ldots, b_{k,m}}{b_{k,m+1}}
\ptrans{\arr{merge}}
\tvox{d_{1,1}, \ldots, d_{1,m}, \ldots, d_{k,1}, \ldots, d_{k,m}}{c_{k+1}} % {d_{k,m+1}}
\end{array}
\]
where
\begin{align*}
d_{i,j} &= c_i + b_{i,j} \\
c_i &= \sum_{1 \le l < i} b_{l,m+1} \\
\end{align*}
}

\framet{$k$-way split/merge}{
\[
\begin{array}{c}
\vox{a_{1,1}, \ldots, a_{1,m},  \ldots,  a_{k,1}, \ldots, a_{k,m}}
\trans{\arr{split}}
\vox{a_{1,1}, \ldots, a_{1,m}} \sdots{3ex} \vox{a_{k,1}, \ldots, a_{k,m}}
\trans{\arr{lscan} \sdots{10ex} \arr{lscan}}
\tvox{b_{1,1}, \ldots, b_{1,m}}{b_{1,m+1}} \sdots{3ex} \tvox{b_{k,1}, \ldots, b_{k,m}}{b_{k,m+1}}
\trans{\arr{merge}}
\tvox{d_{1,1}, \ldots, d_{1,m}, \ldots, d_{k,1}, \ldots, d_{k,m}}{c_{k+1}} %{d_{k,m+1}}
\end{array}
\]
where
\begin{align*}
d_{i,j} &= c_j + b_{i,j} \\
\tvox{c_1,\ldots,c_k}{c_{k+1}} &= \lscan\left({\vox{b_{1,m+1},\ldots,b_{k,m+1}}}\right) \\
\end{align*}
}

\framet{$k$-way split/merge}{
\[
\begin{array}{c}
\vox{a_{1,1}, \ldots, a_{1,m},  \ldots,  a_{k,1}, \ldots, a_{k,m}}
\trans{\arr{split}}
\vox{a_{1,1}, \ldots, a_{1,m}} \sdots{3ex} \vox{a_{k,1}, \ldots, a_{k,m}}
\trans{\arr{lscan} \sdots{10ex} \arr{lscan}}
\tvox{b_{1,1}, \ldots, b_{1,m}}{b_{1,m+1}} \sdots{3ex} \tvox{b_{k,1}, \ldots, b_{k,m}}{b_{k,m+1}}
\trans{\arr{merge}}
\tvox{d_{1,1}, \ldots, d_{1,m}, \ldots, d_{k,1}, \ldots, d_{k,m}}{c_{k+1}} %{d_{k,m+1}}
\end{array}
\]
where

\vspace{-3ex}\hspace{15ex}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{b_{1,m+1},\ldots,b_{k,m+1}}
\trans{\arr{lscan}}
\tvox{c_1,\ldots,c_k}{c_{k+1}}
\end{array}
\]
\end{minipage}
\begin{minipage}[c]{0.3\textwidth}
\[ d_{i,j} = c_j + b_{i,j} \]
\end{minipage}
}

\framet{Work analysis}{

Master Theorem ($d = 1$):

\[ W(n) = a \, W(n/b) + O(n) \]

\[ 
W(n) = \begin{cases}
 \mtCaseo{<}{n}
 \mtCaseo{=}{n \, \log n}
 \mtCaseo{>}{n^{\log_b a}}
\end{cases}
\]

\ \pause

Scan with $k$-way split:
\begin{align*}
W(n) &= k \, W(n/k) + W(k) + O(n) \\
     &= k \, W (n/k) + O(n)
\end{align*}
Still $O(n \, \log n)$.

\vspace{1ex} \pause
If $k$ is \emph{fixed}.
}

\framet{Split inversion}{

Two kinds of split:
\begin{itemize}
\item \emph{Top-down} --- $k$ pieces of size $n/k$ each
\begin{align*}
W(n) &= k \, W(n/k) + W(k) + O(n) \\
     &= k \, W (n/k) + O(n) \\
     &= O(n \, \log n)
\end{align*}
\pause
\item 
\emph{Bottom-up} --- $n/k$ pieces of size $k$ each:
\pause
\begin{align*}
W(n) &= (n/k) \, W(k) + W (n/k) + O(n) \\
     &= W (n/k) + O(n) \\
     &= O(n)
\end{align*}
\pause
Mission accomplished: $O(n)$ work and $O(\log n)$ depth!
\end{itemize}
}

\framet{Root split} {

Another idea: split into $\sqrt{n}$ pieces of size $\sqrt{n}$ each.

\pause

\begin{align*}
W(n) &= \sqrt{n} \cdot W (\sqrt{n}) + W (\sqrt{n}) + O(n) \\
     &= \sqrt{n} \cdot W (\sqrt{n}) + O(n) \\
     &= O(n \, \log \log n) \\
\\[1ex]
D(n) &= O(\log \log n)
\end{align*}

Nearly constant depth and nearly linear work.
Useful in practice?
}

\framet{In CUDA C -- bottom-up binary}{
\begin{minipage}[c]{0.7\textwidth}
\tiny
\begin{verbatim}
__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    // load input into shared memory
    temp[2*thid] = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];
    // build sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai]; }
        offset *= 2; }
    // clear the last element
    if (thid == 0) { temp[n - 1] = 0; }
    // traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; } }
    __syncthreads();
    // write results to device memory
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1]; }
\end{verbatim}
\vspace{-6ex}
\href{http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html}{\emph{Source: GPU Gems 3, Chapter 39}}
\normalsize
\end{minipage}
\hspace{-1in}
\begin{minipage}[c]{0.25\textwidth}
\pause
\begin{figure}
\wpicture{2in}{ShadowedPictures/picard-facepalm}
\end{figure}
\end{minipage}
}

\framet{In Haskell --- generalized left scan}{

> class LScan f where
>   lscan :: Monoid a => f a -> (f a, a)

\vspace{2ex}

Parametrized over container and associative operation.

}

\framet{In Haskell --- top-down}{

> data T f a = L a | B (f (T f a))
>
> SPACE
>
> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L a)   = (L mempty, a)
>   lscan (B ts)  = (B (adjust <$> zip (tots',ts')), tot)
>    where
>      (ts' ,tots)   = unzip (lscan <$> ts)
>      (tots',tot)   = lscan tots
>      adjust (p,t)  = (p `mappend`) <$> t

}

\framet{In Haskell --- bottom-up}{

> data T f a = L a | B (T f (f a))
>
> SPACE
>
> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L a)   = (L mempty, a)
>   lscan (B ts)  = (B (adjust <$> zip (tots',ts')), tot)
>    where
>      (ts' ,tots)   = unzip (lscan <$> ts)
>      (tots',tot)   = lscan tots
>      adjust (p,t)  = (p `mappend`) <$> t

}

\framet{In Haskell --- root split}{

> data T f a = L (f a) | B (T f (T f a))
> 
> SPACE
> 
> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L as)  = first L (lscan as)
>   lscan (B ts)  = (B (adjust <$> zip (tots',ts')), tot)
>    where
>      (ts' ,tots)   = unzip (lscan <$> ts)
>      (tots',tot)   = lscan tots
>      adjust (p,t)  = (p `mappend`) <$> t

}

%if atwork

\framet{Reflections}{
\begin{itemize} \itemsep 1.5em
\item Reschedulable computing needs new languages and techniques.
\vspace{1ex}
\begin{itemize} \itemsep 1em
\pitem \emph{Out:} sequencing, threads, mutation.
\pitem \emph{In:} math, functional programming.
\end{itemize}
\pitem Reduce other dependencies via equational reasoning.
\pitem Associativity matters.
\end{itemize}
}

%else
\framet{The commonality -- type composition}{

> lscanGF ::  (Functor f, Zippy g, LScan g, LScan f, Monoid a) =>
>             g (f a) -> (g (f a), a)
> lscanGF gfa  = (adjust <$> zip (tots',gfa'), tot)
>  where
>    (gfa' ,tots)  = unzip (lscan <$> gfa)
>    (tots',tot)   = lscan tots
>    adjust (p,t)  = (p `mappend`) <$> t

}


\framet{Top-down trees}{

> data T f a = L a | B (f (T f a))
>
> SPACE
>
> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L a)  = (L mempty, a)
>   lscan (B w)  = first B (lscanGF w)

}


\framet{Bottom-up trees}{

> data T f a = L a | B (T f (f a))
>
> SPACE
>
> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L a)  = (L mempty, a)
>   lscan (B w)  = first B (lscanGF w)

}

\framet{Root split -- top-down}{

> data T f a = L (f a) | B (T f (T f a))
>
> SPACE
>
> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L as)  = first L  (lscan as)
>   lscan (B w)   = first B  (lscanGF w)

}

\framet{Type composition, explicitly}{

> newtype (g :. f) a = O (g (f a))

\pause

> instance  (Functor f, Zippy g, LScan g, LScan f) =>
>           LScan (g :. f) where
>   lscan (O ts)  = (O (adjust <$> zip (tots',gfa')), tot)
>    where
>      (gfa' ,tots)  = unzip (lscan <$> gfa)
>      (tots',tot)   = lscan tots
>      adjust (p,t)  = (p `mappend`) <$> t

}

\framet{Trees with explicit composition}{

> data T f a = L   a    | B ((f :. T f)    a)  -- top-down f-tree
>
> data T f a = L   a    | B ((T f :. f)    a)  -- bottom-up f-tree
>
> data T f a = L (f a)  | B ((T f :. T f)  a)  -- top-down root f-tree
>
> data T f a = L (f a)  | B (T (f :. f)    a)  -- bottom-up root f-tree

\pause
|f|-trees:

> instance (Zippy f, LScan f) => LScan (RT f) where
>   lscan (L a)  = (L mempty, a)
>   lscan (B w)  = first B  (lscan w)

\vspace{0.7ex}

}

\framet{Trees with explicit composition}{

> data T f a = L   a    | B ((f :. T f)    a)  -- top-down f-tree
>
> data T f a = L   a    | B ((T f :. f)    a)  -- bottom-up f-tree
>
> data T f a = L (f a)  | B ((T f :. T f)  a)  -- top-down root f-tree
>
> data T f a = L (f a)  | B (T (f :. f)    a)  -- bottom-up root f-tree

Root |f|-trees:

> instance (Zippy f, LScan f) => LScan (RT f) where
>   lscan (L as)  = first L  (lscan as)
>   lscan (B w)   = first B  (lscan w)

\pause
The bottom-up trees are \emph{perfect} -- $f^n$ and $f^{f^n}$.

}

%endif

\framet{Data structure tinker toys}{

\begin{minipage}[c]{0.68\textwidth}

> newtype  Const b      a  = Const b
> newtype  Id           a  = Id a
> data     (f  :*:  g)  a  = f a :*: g a
> data     (f  :+:  g)  a  = InL (f a) | InR (g a)
> newtype  (g  :.   f)  a  = O (g (f a))

%% \vspace{-15ex}

\ 

\pause
Each has an |LScan| instance.

\ 

Parallel scan for many data structures.

\ 

See post: \href{http://conal.net/blog/posts/composable-parallel-scanning}{\emph{Composable
parallel scanning}}.

\end{minipage}
\begin{minipage}[c]{0.3\textwidth}
\wpicture{1.5in}{Pictures/tinker-toy-bird}
\end{minipage}

\pause\ 

General approach to algorithm development?

}

\end{document}
