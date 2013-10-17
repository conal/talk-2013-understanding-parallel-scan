> {-# LANGUAGE DeriveFunctor, TypeOperators #-}
> {-# OPTIONS_GHC -Wall #-}

> {-# OPTIONS_GHC -fno-warn-unused-imports #-}

> module HScan where

> import Prelude hiding (zip,unzip)
> import Data.Monoid
> import Data.Functor ((<$>))
> import Control.Arrow (first,(***))
> import Control.Compose ((:.)(..))

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
>
> unzipWith2 :: Zippy f => (a -> (b, c)) -> f a -> (f b, f c)
> unzipWith2 f xs = unzip (fmap f xs)

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

> instance Zippy P where
>   zip (a :# a', b :# b') = (a,b) :# (a',b')
>   unzip ((a,b) :# (a',b')) = (a :# a', b :# b')

> instance (Zippy g, Zippy f) => Zippy (g :. f) where
>   zip (O gfa, O gfb) = O (zip <$> zip (gfa,gfb))
>   unzip (O gfab) = (O *** O) (unzip (unzip <$> gfab))

< gfa :: g (f a)
< gfb :: g (f b)
< zip (gfa,gfb) :: g (f a, f b)
< zip <$> zip (gfa,gfb) :: g (f (a,b))
< O (zip <$> zip (gfa,gfb)) :: (g :. f) (a,b)

< gfab :: g (f (a,b))
< unzip <$> gfab :: g (f a, f b)
< unzip (unzip <$> gfab :: g (f a, f b)) :: (g (f a), g (f b))

Top-down binary trees:

> type TT = T P

Bottom-up:

> data T' f a = L' a | B' (T f (f a)) deriving Functor

> type BT = T' P

Left scan class: 

> class LScan f where
>   lscan :: Monoid a => f a -> (f a, a)

Uniform pairs:

> instance LScan P where
>   lscan (a :# b) = (mempty :# a, a `mappend` b)

Composition scan:

> lscanGF ::  (Functor f, Zippy g, LScan g, LScan f, Monoid a) =>
>             g (f a) -> (g (f a), a)
> lscanGF gfa  = (adjust <$> zip (tots',gfa'), tot)
>  where
>    (gfa' ,tots)  = unzip (lscan <$> gfa)
>    (tots',tot)   = lscan tots
>    adjust (p,t)  = (p `mappend`) <$> t

Scan for top-down trees:

< instance (Zippy f, LScan f) => LScan (T f) where
<   lscan (L a)   = (L mempty, a)
<   lscan (B ts)  = (B (adjust <$> zip (tots',ts')), tot)
<    where
<      (ts' ,tots)   = unzip (lscan <$> ts)
<      (tots',tot)   = lscan tots
<      adjust (p,t)  = (p `mappend`) <$> t

> instance (Zippy f, LScan f) => LScan (T f) where
>   lscan (L a)  = (L mempty, a)
>   lscan (B w)  = first B (lscanGF w)

Same definition for bottom-up trees (modulo type and constructor names):

< instance (Zippy f, LScan f) => LScan (T' f) where
<   lscan (L' a)   = (L' mempty, a)
<   lscan (B' ts)  = (B' (adjust <$> zip (tots',ts')), tot)
<    where
<      (ts' ,tots)   = unzip (lscan <$> ts)
<      (tots',tot)   = lscan tots
<      adjust (p,t)  = (p `mappend`) <$> t

> instance (Zippy f, LScan f) => LScan (T' f) where
>   lscan (L' a)  = (L' mempty, a)
>   lscan (B' w)  = first B' (lscanGF w)

Root split, top-down:

> data RT f a = L'' (f a) | B'' (T f (T f a)) deriving Functor

< instance (Zippy f, LScan f) => LScan (RT f) where
<   lscan (L'' as)  = first L'' (lscan as)
<   lscan (B'' ts)  = (B'' (adjust <$> zip (tots',ts')), tot)
<    where
<      (ts' ,tots)   = unzip (lscan <$> ts)
<      (tots',tot)   = lscan tots
<      adjust (p,t)  = (p `mappend`) <$> t

> instance (Zippy f, LScan f) => LScan (RT f) where
>   lscan (L'' as)  = first L'' (lscan as)
>   lscan (B'' w)   = first B'' (lscanGF w)

Generalize to functor composition:

< instance (Functor f, Zippy g, LScan g, LScan f) => LScan (g :. f) where
<   lscan (O ts)  = (O (adjust <$> zip (tots',ts')), tot)
<    where
<      (ts' ,tots)   = unzip (lscan <$> ts)
<      (tots',tot)   = lscan tots
<      adjust (p,t)  = (p `mappend`) <$> t

> instance (Functor f, Zippy g, LScan g, LScan f) => LScan (g :. f) where
>   lscan (O ts)  = first O (lscanGF ts)

Bottom-up:

> data RT' f a = L''' (f a) | B''' (T (f :. f) a) deriving Functor
>
> instance (Zippy f, LScan f) => LScan (RT' f) where
>   lscan (L''' as)  = first L''' (lscan as)
>   lscan (B''' w)   = first B''' (lscan w)
