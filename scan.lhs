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

\title{High-level algorithm design for reschedulable computation, Part 1}
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

\nc\bboxed[1]{\boxed{\rule[-0.75ex]{0pt}{2.4ex}#1}}
\nc\vox[1]{\bboxed{#1}}
\nc\tvox[2]{\vox{#1}\vox{#2}}

\nc\sums{\Varid{sums}}

\nc\trans[1]{\\[1.3ex] #1 \\[0.75ex]}
\nc\ptrans[1]{\pause\trans{#1}\pause}

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
\ptrans{\arr{split}}
\vox{a_1, \ldots, a_n}
\ 
\vox{b_1, \ldots, b_n}
% \ptrans{\Downarrow \sums \hspace{3ex} \sums \Downarrow}
\ptrans{\arr{sums} \hspace{12ex} \arr{sums}}
\tvox{a'_1, \ldots, a'_n}{a'_{n+1}}
\ 
\tvox{b'_1, \ldots, b'_n}{b'_{n+1}}
% \ptrans{\Downarrow}
\ptrans{\arr{merge}}
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
 D(2 \, m) &= D(m) + c \\
 D(n) &= O(\log n)
 \end{align*}

\pitem Linear:
 \begin{align*}
  D(2 \, m) &= D(m) + c \, m \\
  D(2^k) &= (1 + 2 + 4 + \cdots + 2^{k-1}) \cdot c = O(2^k) \\
  D(n) &= O(n)
 \end{align*}

\pitem Logarithmic:
 \begin{align*}
  D(2 \, m) &= D(m) + c \, \log m \\
  D(2^k) &= (0 + 1 + 2 + \cdots + k-1) \cdot c = O(k^2) \\
  D(n) &= O(\log^2 n)
 \end{align*}
\end{itemize}
}

\framet{Work analysis}{
Work recurrence:
\[ W(2 \, n) = 2 \, W(n) + c' \, n \]

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

\end{document}
