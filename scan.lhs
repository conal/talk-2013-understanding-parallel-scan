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

\nc\bboxed[1]{\boxed{\rule[-0.9ex]{0pt}{2.6ex}#1}}
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
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\end{array}
\]
\end{minipage}
where
\begin{minipage}[c]{0.3\textwidth}
\[ a'_i = \sum\limits_{1 \le k < i}{a_k} \]
\end{minipage}
\end{center}

\vspace{2ex}\pause
Work: quadratic.

\pause
Time: quadratic, linear, logarithmic.
}

\framet{As a recurrence}{
\begin{center}
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n}
\trans{\sums\Downarrow}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\end{array}
\]
\end{minipage}
where
\begin{minipage}[c]{0.3\textwidth}
\[
\begin{array}{ll}
a'_1 = 0 \\
a'_{i+1} = a'_i + a_i
\end{array}
\]
\end{minipage}
\end{center}

\vspace{2ex} \pause
Work: linear.

\pause
\emph{Depth} (ideal ``time''): linear.

Linear \emph{dependency chain} thwarts parallelism (depth $<$ work).
}

\nc\arr[1]{\Downarrow _{\text{\makebox[0pt][l]{\emph{#1}}}}}

\framet{Divide and conquer}{
\pause
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n, b_1, \ldots, b_n}
\ptransp{\arr{split}}
\vox{a_1, \ldots, a_n}
\ 
\vox{b_1, \ldots, b_n}
\ptransp{\arr{sums} \hspace{12ex} \arr{sums}}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
\ptransp{\arr{merge}}
\tvox{a'_1, \ldots, a'_n, a'_{n+1} + b'_1, \ldots, a'_{n+1} + b'_n}{a'_{n+1}+b'_{n+1}}
\end{array}
\]

\begin{itemize}
\pitem Why is this definition equivalent?
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
 D(n) &= D(n/2) + c \\
 D(n) &= O(\log n)
 \end{align*}

\pitem Linear:
 \begin{align*}
  D(n) &= D(n/2) + c \, n \\
  D(2^k) &= (1 + 2 + 4 + \cdots + 2^{k-1}) \cdot c = O(2^k) \\
  D(n) &= O(n)
 \end{align*}

\pitem Logarithmic:
 \begin{align*}
  D(n) &= D(n/2) + c \, \log n \\
  D(2^k) &= (0 + 1 + 2 + \cdots + k-1) \cdot c = O(k^2) \\
  D(n) &= O(\log^2 n)
 \end{align*}
\end{itemize}
}

\framet{Work analysis}{
Work recurrence:
\[ W(n) = 2 \, W(n/2) + c' \, n \]

\pause
By the \href{http://en.wikipedia.org/wiki/Master_theorem}{\emph{Master Theorem}},
\[ W(n) = O(n \, \log n) \]
}

\framet{Analysis summary}{

\begin{align*}
 D(n) &= O(\log n) \hidden{\text{\hspace{4ex} (or \ldots)}} \\[2ex]
 W(n) &= O(n \, \log n)
\end{align*}

\ 

\pause Note:
\begin{itemize}
\item The sequential version does $O(n)$ work in $O(n)$ depth.
\pitem Can we get $O(n)$ work and $O(\log n)$ depth?
\end{itemize}
}

\nc\case[2]{#2 & \text{if~} #1 \\}
\nc\mtCase[2]{\case{a #1 b^d}{O(#2)}}

\framet{Master Theorem}{
Given a recurrence:
\[ f(n) = a \, f(n/b) + c \, n^d \]
We have the following closed form bound:
\[ 
f(n) = \begin{cases}
 \mtCase{<}{n^d}
 \mtCase{=}{n^d \, \log n}
 \mtCase{>}{n^{\log_b a}}
\end{cases}
\]
\vspace{5.4ex} % to align with next slide
}

\nc\mtCaseo[2]{\case{a #1 b}{O(#2)}}

\framet{Simplified Master Theorem ($d=1$)}{
Given a recurrence:
\[ f(n) = a \, f(n/b) + c \, n \]
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
\[ W(n) = 2 \, W(n/2) + c' \, n \]
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

\framet{Variation: three-way split/merge}{
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n, b_1, \ldots, b_n, c_1, \ldots, c_n}
\ptrans{\arr{split}}
\vox{a_1, \ldots, a_n}
\ 
\vox{b_1, \ldots, b_n}
\ 
\vox{c_1, \ldots, c_n}
\ptrans{\arr{sums} \hspace{12ex} \arr{sums} \hspace{12ex} \arr{sums}}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
\ 
\tvox{c'_1, \ldots, c'_n}{c'_{n+1}}
\ptrans{\arr{merge}}
\tvox{a'_1, \ldots, a'_n , b''_1, \ldots, b''_n, c''_1, \ldots, c''_n}{c''_{n+1}}
\end{array}
\]
where
\begin{align*}
b''_i&= a'_{n+1}+b'_i  \\
c''_i&= a'_{n+1}+b'_{n+1}+c'_i 
\end{align*}
}

\framet{Variation: four-way split/merge}{
\[
\begin{array}{c}
\vox{a_1, \ldots, a_n, b_1, \ldots, b_n, c_1, \ldots, c_n, d_1, \ldots, d_n}
\ptrans{\arr{split}}
\vox{a_1, \ldots, a_n}
\ 
\vox{b_1, \ldots, b_n}
\ 
\vox{c_1, \ldots, c_n}
\ 
\vox{d_1, \ldots, d_n}
\ptrans{\arr{sums} \hspace{12ex} \arr{sums} \hspace{12ex} \arr{sums} \hspace{12ex} \arr{sums}}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
\ 
\tvox{c'_1, \ldots, c'_n}{c'_{n+1}}
\ 
\tvox{d'_1, \ldots, d'_n}{d'_{n+1}}
\ptrans{\arr{merge}}
\tvox{a'_1, \ldots, a'_n , b''_1, \ldots, b''_n, c''_1, \ldots, c''_n, d''_1, \ldots, d''_n}{d''_{n+1}}
\end{array}
\]
where
\begin{align*}
b''_i&= a'_{n+1}+b'_i \\
c''_i&= a'_{n+1}+b'_{n+1}+c'_i \\
d''_i&= a'_{n+1}+b'_{n+1}+c'_{n+1}+d'_i
\end{align*}
}

\framet{Multi-way merge}{
\[
\begin{array}{c}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
\ 
\tvox{c'_1, \ldots, c'_n}{c'_{n+1}}
\ 
\tvox{d'_1, \ldots, d'_n}{d'_{n+1}}
\ptrans{\arr{merge}}
\tvox{a''_1, \ldots, a''_n , b''_1, \ldots, b''_n, c''_1, \ldots, c''_n, d''_1, \ldots, d''_n}{d''_{n+1}}
\end{array}
\]
\pause where
\begin{align*}
a''_i &= 0 + a'_i \\
b''_i &= a'_{n+1} + b'_i \\
c''_i &= (a'_{n+1}+b'_{n+1}) + c'_i \\
d''_i &= (a'_{n+1}+b'_{n+1}+c'_{n+1}) + d'_i \\
\end{align*}

\pause
\vspace{-3ex} % Why??
\emph{Where have we seen this pattern of shifts?}
}

\nc\sdots[1]{\hspace{#1} \ldots \hspace{#1}}

\framet{$k$-way split/merge}{
\[
\begin{array}{c}
\vox{a_{1,1}, \ldots, a_{1,m},  \ldots,  a_{k,1}, \ldots, a_{k,m}}
\ptrans{\arr{split}}
\vox{a_{1,1}, \ldots, a_{1,m}} \sdots{3ex} \vox{a_{k,1}, \ldots, a_{k,m}}
\ptrans{\arr{sums} \sdots{6ex} \arr{sums}}
\tvox{b_{1,1}, \ldots, b_{1,m}}{b_{1,m+1}} \sdots{3ex} \tvox{b_{k,1}, \ldots, b_{k,m}}{b_{k,m+1}}
\ptrans{\arr{merge}}
\tvox{d_{1,1}, \ldots, d_{1,m}, \ldots, d_{k,1}, \ldots, d_{k,m}}{c_{k+1}}
\end{array}
\]
where
\begin{align*}
\tvox{c_1,\ldots,c_k}{c_{k+1}} &= \sums\left({\vox{b_{1,m+1},\ldots,b_{k,m+1}}}\right) \\
d_{i,j} &= c_j + b_{i,j}
\end{align*}

}


\end{document}
