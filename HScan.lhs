> {-# LANGUAGE DeriveFunctor #-}
> {-# OPTIONS_GHC -Wall #-}

> {-# OPTIONS_GHC -fno-warn-unused-imports #-}

> module HScan where

> import Prelude hiding (zip,unzip)
> import Data.Monoid

Sequential prefix sum on lists:

> lsums :: [Int] -> ([Int], Int)
> lsums l = lsums' (0,l)
> 
> lsums' :: (Int,[Int]) -> ([Int], Int)
> lsums' (p, [] ) = ([],p)
> lsums' (p, x:l) = (p:l',q)
>  where
>    (l',q) = lsums' (p+x,l)


Sequential prefix sum on trees:

< data T a = L a | B (T a) (T a) deriving Functor

< tscan :: Monoid a => T a -> (T a, a)
< tscan w = tscan' (mempty, w)
<
< tscan' :: Monoid a => (a, T a) -> (T a, a)
< tscan' (p, L a)   = (L p, p `mappend` a)
< tscan' (p, B u v) = (B u' v', r)
<  where
<    (u', q) = tscan' (p, u)
<    (v', r) = tscan' (q, v)

Top-down divide-and-conquer:

< tscan :: Monoid a => T a -> (T a, a)
< tscan (L n) = (L mempty, n)
< tscan (B u v) = (B u' (fmap (q `mappend`) v'), q `mappend` r)
<  where
<    (u', q) = tscan u
<    (v', r) = tscan v

> class Functor f => Zippy f where
>   zip   :: (f a, f b) -> f (a,b)
>   unzip :: f (a,b) -> (f a, f b)

> zipWith2 :: Zippy f => (a -> b -> c) -> f a -> f b -> f c
> zipWith2 h as bs = fmap (uncurry h) (zip (as,bs))

> data T f a = L a | B (f (T f a)) deriving Functor

> instance Zippy f => Zippy (T f) where
>   zip (L  a, L  b) = L (a,b)
>   zip (B us, B vs) = B (fmap zip (zip (us,vs)))
>   zip _ = undefined
>   unzip (L (a,b)) = (L a, L b)
>   unzip (B ts) = (B us', B vs')
>    where
>      (us',vs') = unzip (fmap unzip ts)

< ts :: f (T f (a,b))
< fmap unzip ts :: f (T f a, T f b)
< unzip (fmap unzip ts) :: (f (T f a), f (T f b))

< tscan :: T a -> (T a, a)
< tscan (L n) = (L 0, n)
< tscan (B u v) = (B u' (fmap (q+) v'), q+r)
<  where
<    (u', q) = tscan u
<    (v', r) = tscan v

Uniform pairs:

> data P a = a :# a deriving Functor

Bottom-up binary trees:

> type BT = T P

Top-down trees:

> data T' f a = L' a | B' (f (T f a)) deriving Functor

> type TT = T' P

Left scan class: 

> class LScan f where
>   lscan :: Monoid a => f a -> (f a, a)

Scan for top-down trees:

> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L n)  = (L mempty, n)
>   lscan (B ts) = (B (zipWith2 adjust tots' ts'), tot)
>    where
>      (ts' ,tots)  = unzip (fmap lscan ts)
>      (tots',tot)  = lscan tots
>      adjust p t   = fmap (p `mappend`) t

Same definition for bottom-up trees (modulo type and constructor names):

> instance (Zippy f, LScan f) => LScan (T' f) where
>   lscan (L' n)  = (L' mempty, n)
>   lscan (B' ts) = (B' (zipWith2 adjust tots' ts'), tot)
>    where
>      (ts' ,tots)  = unzip (fmap lscan ts)
>      (tots',tot)  = lscan tots
>      adjust p t   = fmap (p `mappend`) t


