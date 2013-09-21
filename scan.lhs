%% -*- latex -*-
\documentclass[serif]{beamer}

\usepackage{beamerthemesplit}

\usepackage{graphicx}
\usepackage{color}
\DeclareGraphicsExtensions{.pdf,.png,.jpg}

\useinnertheme[shadow]{rounded}
% \useoutertheme{default}
\useoutertheme{shadow}
\useoutertheme{infolines}

\input{macros}

%include polycode.fmt
%include forall.fmt
%include greek.fmt
%include mine.fmt

% \title{High-level algorithm design for reschedulable computation, Part 1}
\title{Understanding efficient parallel scan}
\author{\href{http://conal.net}{Conal Elliott}}
\institute{\href{http://tabula.com/}{Tabula}}
% Abbreviate date/venue to fit in infolines space
%% \date{\href{http://www.meetup.com/haskellhackersathackerdojo/events/105583982/}{March 21, 2013}}
\date{Fall, 2013}

%% Do I use any of this picture stuff?

\nc\wpicture[2]{\includegraphics[width=#1]{pictures/#2}}

\nc\wfig[2]{
\begin{center}
\wpicture{#1}{#2}
\end{center}
}
\nc\fig[1]{\wfig{4in}{#1}}

\setlength{\itemsep}{2ex}
\setlength{\parskip}{1ex}

\setlength{\blanklineskip}{1.5ex}

\nc\usebg[1]{\usebackgroundtemplate{\wpicture{1.2\textwidth}{#1}}}

\begin{document}

\frame{\titlepage}

\nc\framet[2]{\frame{\frametitle{#1}#2}}

\nc\hidden[1]{}

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

\nc\sums{\Varid{sums}}

\nc\trans[1]{\\[1.3ex] #1 \\[0.75ex]}
\nc\ptrans[1]{\pause\trans{#1}}
\nc\ptransp[1]{\ptrans{#1}\pause}

\nc\pitem{\pause \item}

%%%%

\framet{Prefix sum (scan)}{
\begin{center}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n}
\trans{\sums\Downarrow}
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

\vspace{8ex}
}

\framet{As a recurrence}{
\begin{center}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n}
\trans{\sums\Downarrow}
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
\ptransp{\arr{sums} \hspace{12ex} \arr{sums}}
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

\vspace{3ex}\pause Can we get $O(n)$ work and $O(\log n)$ depth?
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
\trans{\arr{sums} \hspace{12ex} \arr{sums}}
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
\ptrans{\arr{sums} \hspace{12ex} \arr{sums} \hspace{12ex} \arr{sums}}
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
\ptrans{\arr{sums} \sdots{10ex} \arr{sums}}
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
\trans{\arr{sums} \sdots{10ex} \arr{sums}}
\tvox{b_{1,1}, \ldots, b_{1,m}}{b_{1,m+1}} \sdots{3ex} \tvox{b_{k,1}, \ldots, b_{k,m}}{b_{k,m+1}}
\trans{\arr{merge}}
\tvox{d_{1,1}, \ldots, d_{1,m}, \ldots, d_{k,1}, \ldots, d_{k,m}}{c_{k+1}} %{d_{k,m+1}}
\end{array}
\]
where
\begin{align*}
d_{i,j} &= c_j + b_{i,j} \\
\tvox{c_1,\ldots,c_k}{c_{k+1}} &= \sums\left({\vox{b_{1,m+1},\ldots,b_{k,m+1}}}\right) \\
\end{align*}
}

\framet{$k$-way split/merge}{
\[
\begin{array}{c}
\vox{a_{1,1}, \ldots, a_{1,m},  \ldots,  a_{k,1}, \ldots, a_{k,m}}
\trans{\arr{split}}
\vox{a_{1,1}, \ldots, a_{1,m}} \sdots{3ex} \vox{a_{k,1}, \ldots, a_{k,m}}
\trans{\arr{sums} \sdots{10ex} \arr{sums}}
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
\trans{\arr{sums}}
\tvox{c_1,\ldots,c_k}{c_{k+1}}
\end{array}
\]
\end{minipage}
\begin{minipage}[c]{0.3\textwidth}
\[ d_{i,j} = c_j + b_{i,j} \]
\end{minipage}
}

\framet{Work analysis}{

Master Theorem:

\[ W(n) = a \, W(n/b) + O(n) \]

\[ 
W(n) = \begin{cases}
 \mtCaseo{<}{n}
 \mtCaseo{=}{n \, \log n}
 \mtCaseo{>}{n^{\log_b a}}
\end{cases}
\]

\ \pause

% $k$-way split:
$k$ pieces of size $n/k$ each:
% \[ W(n) = k \, W(n/k) + W(k) + O(n) \]
\begin{align*}
W(n) &= k \, W(n/k) + W(k) + O(n) \\
     &= k \, W (n/k) + O(n)
\end{align*}
Still $O(n \, \log n)$.

\vspace{1ex} \pause
If $k$ is \emph{fixed}.
}

\framet{Split inversion}{

$k$-way split:
$k$ pieces of size $n/k$ each.

\ \pause

Idea: \emph{Invert split} --- $n/k$ pieces of size $k$ each.

\pause
\begin{align*}
W(n) &= (n/k) \, W(k) + W (n/k) + O(n)\\
     &= W (n/k) + O(n)
\end{align*}

Now we get $O(n)$ work and depth!
}

\framet{Root split} {

Another idea: split into $\sqrt{n}$ pieces of size $\sqrt{n}$ each.

\[ W(n) = \sqrt{n} \cdot W (\sqrt{n}) + W (\sqrt{n}) + O(n) \]

\ \pause

Solution:

\begin{align*}
 D(n) &= O(\log \log n) \\
 W(n) &= O(n \, \log \log n) \\
\end{align*}
}

\end{document}
