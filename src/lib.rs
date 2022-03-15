//! A simple crate for iterating over the bits in bytes.
//! 
//! The crate provides 2 functionalities, exposed as traits:
//!  - Iterating over the 1-bits in unsigned integer types and sequences thereof -- see [`Biterate`]
//!  - The reverse: Constructing integers from indicies of the 1-bits -- see [`CompressIndices`]
#![deny(missing_docs)]

use std::ops::{Shr, ShlAssign, BitOrAssign};


#[inline(always)]
const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

/// An iterator of bit-indices from a multi-word sequence.
pub struct BitIndicesSeq<I: Iterator> {
    words: I,
    offset: usize,
    current_word: Option<BitIndices<I::Item>>,
}

impl<I> Iterator for BitIndicesSeq<I>
    where 
        I: Iterator,
        I::Item : BitBlock,
{
    type Item = usize; 

    fn next(&mut self) -> Option<usize> {
        match self.current_word.as_mut().map(Iterator::next) {
            Some(Some(i)) => return Some(i + self.offset),
            Some(None) => { self.offset += num_bits::<I::Item>(); },
            None => {},
        }

        while let Some(word) = self.words.next() {
            let mut current_word = word.biterate();
            match current_word.next() {
                Some(i) => {
                    self.current_word = Some(current_word);
                    return Some(i + self.offset)
                },
                None => {
                    self.offset += num_bits::<I::Item>();
                    continue   
                }
                
            }
        }
        None
    }
}

/// An iterator over the bits in a word (such as `u8`, `u16` etc).
#[derive(Clone, Copy)]
pub struct BitIndices<T> {
    offset: u8, // biggest word we support is u128 which has less than 255 bits
    word: T,
}

/// A trait abstracting over types which can be used as a block of bits.  
/// 
/// At the moment this is unsigned integer primitive types. 
///  
/// The trait is not sealed, so you may implement it, but it unstable and may change.
pub trait BitBlock : ShlAssign<u8> + Shr + BitOrAssign + Copy {
    /// The zero value for this type
    fn zero() -> Self;

    /// An array of zero values for this type
    fn zero_array<const N: usize>() -> [Self; N];
    
    /// Set the single bit at `idx` places from the left to 1.  
    /// 
    /// # Safety
    /// The caller guarantees that `idx` is less than the number of bits in the type.
    unsafe fn set_bit_unchecked(&mut self, idx: usize);

    /// The checked version of `set_bit_unchecked`
    fn set_bit(&mut self, idx: usize) {
        assert!(idx < num_bits::<Self>(), "{} is not large enough to hold {}", std::any::type_name::<Self>(), idx);
        unsafe { self.set_bit_unchecked(idx) }
    }

    /// Is this value zero?
    fn is_zero(self) -> bool;

    /// Is this value one?
    fn is_one(self) -> bool;

    /// The number of leading zeros in the bit-representation.
    fn leading_zeros(self) -> u32;
}

impl<T: BitBlock> Iterator for BitIndices<T> {
    type Item = usize; 

    fn next(&mut self) -> Option<usize> {
        if self.word.is_zero() { 
            None
        } else if self.word.is_one() { // Has to be special-cased because shift overflows otherwise
            self.word = T::zero();
            Some(num_bits::<T>() - 1)
        } else {
            let shift = self.word.leading_zeros() as u8 + 1;
            self.word <<= shift;
            self.offset += shift;
            Some(self.offset as usize - 1)
        }
    }
}

/// The main trait for iterating over the one bits in an integer or sequence of integers.
pub trait Biterate {
    /// The iterator type produced by `.biterate()`
    type Iterator;

    /// Iterate over the 1-bits in the bit-representation, starting from 0.  Indices are unique.
    ///
    /// Primitive unsigned integer types, the 0 index is the left-most bit.
    /// 
    /// For multi-word sequences, such as arrays, slices and [`Vec`]s, the 0 index is the 
    /// left-most bit of the first element in the sequence.
    /// 
    /// # Examples 
    /// ```
    /// use biterate::Biterate;
    /// 
    /// let bits = [0b1000_0000u8, 0b0110_0000];
    /// let indices : Vec<_> = bits.biterate().collect();
    /// assert_eq!(indices, vec![0, 9, 10]);
    /// 
    /// let bits = 0b0000_1001_0110_0001u16;
    /// let indices : Vec<_> = bits.biterate().collect();
    /// assert_eq!(indices, vec![4, 7, 9, 10, 15]);
    /// ```
    fn biterate(self) -> Self::Iterator;
}


/// Adaptor trait for working with generic iterators.
/// 
/// Because of Rust's rules against overlapping impls, this is has to be a separate trait 
/// from [`Biterate`] in order to be implemented for all iterators of appropriate types. 
pub trait BiterateAdaptor {
    /// The iterator type produced by `.biterate()`
    type Iterator;

    /// Iterate over the bits in this sequence.  
    /// 
    /// # Examples 
    /// ```
    /// use biterate::BiterateAdaptor;
    /// 
    /// let bits = std::iter::once(0b1000_0000u8).chain([0b0110_0000]);
    /// let indices : Vec<_> = bits.biterate().collect();
    /// assert_eq!(indices, vec![0, 9, 10]);
    /// ```
    fn biterate(self) -> Self::Iterator;
}

impl<T: BitBlock> Biterate for T {
    type Iterator = BitIndices<T>;
    fn biterate(self) -> Self::Iterator {
        BitIndices {
            word: self,
            offset: 0,
        }
    }
}

impl<I> BiterateAdaptor for I
    where
        I: Iterator,
        I::Item: BitBlock 
{
    type Iterator = BitIndicesSeq<I>;
    fn biterate(self) -> Self::Iterator {
        BitIndicesSeq{ words: self, offset: 0, current_word: None }
    }
}

impl<'a, T: BitBlock> Biterate for &'a [T] {
    type Iterator = BitIndicesSeq<std::iter::Copied<std::slice::Iter<'a, T>>>;
    fn biterate(self) -> Self::Iterator {
        self.iter().copied().biterate()
    }
}

impl<T: BitBlock, const N: usize> Biterate for [T; N] {
    type Iterator = BitIndicesSeq<std::array::IntoIter<T, N>>;

    fn biterate(self) -> Self::Iterator {
        self.into_iter().biterate()
    }
}

impl<T: BitBlock> Biterate for Vec<T> {
    type Iterator = BitIndicesSeq<std::vec::IntoIter<T>>;

    fn biterate(self) -> Self::Iterator {
        self.into_iter().biterate()
    }
}


macro_rules! impl_word_for_primitive {
    ($t:ty) => {
        impl BitBlock for $t {
            #[inline(always)]
            fn zero() -> Self { 0 }
            
            #[inline(always)]
            fn zero_array<const N: usize>() -> [Self; N] { [0; N] }

            #[inline(always)]
            unsafe fn set_bit_unchecked(&mut self, idx: usize) {
                const FIRST_BIT : $t = <$t>::rotate_right(1, 1);
                *self |= FIRST_BIT >> idx;
            }

            #[inline(always)]
            fn is_zero(self) -> bool { self == 0 }
        
            #[inline(always)]
            fn is_one(self) -> bool { self == 1 }
        
            #[inline(always)]
            fn leading_zeros(self) -> u32 { <$t>::leading_zeros(self) }
        }        
    };
}

impl_word_for_primitive!(u8);
impl_word_for_primitive!(u16);
impl_word_for_primitive!(u32);
impl_word_for_primitive!(u64);
impl_word_for_primitive!(u128);
impl_word_for_primitive!(usize);

/// Converts an iterator of bit-indices to a compressed bit-set form.
/// 
/// This is the reverse operation of [`Biterate`].
pub trait CompressIndices<Target> {
    /// The trait is parameterised over the output type, like [`Iterator::collect()`], and therefore
    /// requires the return type to be annotated. 
    /// 
    /// # Panics
    /// For fixed-size types, like uints and arrays, this will panic if the return type 
    /// is not big enough to hold the largest index.
    /// 
    /// ```should_panic
    /// # use biterate::CompressIndices;
    /// let bits : u8 = [8].into_iter().compress_indices();
    /// ```
    /// 
    /// ```should_panic
    /// # use biterate::CompressIndices;
    /// let bits : [u16; 2] = [32].into_iter().compress_indices();
    /// ```
    /// 
    /// 
    /// # Examples
    /// ```
    /// use biterate::CompressIndices;
    /// 
    /// let bits : u8 = [0, 1, 5, 7].into_iter().compress_indices();
    /// assert_eq!(bits, 0b1100_0101);
    /// 
    /// let bits : [u8; 3] = [0, 7, 8, 23].into_iter().compress_indices();
    /// assert_eq!(bits, [0b1000_0001, 0b1000_0000, 0b0000_00001]);
    /// 
    /// let bits : Vec<u16> = [0, 7, 8, 23, 31].into_iter().compress_indices();
    /// assert_eq!(bits, vec![0b1000_0001_1000_0000, 0b0000_0001_0000_0001]);
    /// ```
    fn compress_indices(self) -> Target;
}

impl<I, T> CompressIndices<T> for I 
    where 
        I: Iterator<Item=usize>,
        T: BitBlock,
{
    fn compress_indices(self) -> T {
        let mut i = T::zero();
        for k in self {
            i.set_bit(k);
        }
        i
    }
}


impl<I, T, const N: usize> CompressIndices<[T; N]> for I 
    where 
        I: Iterator<Item=usize>,
        T: BitBlock,
{
    fn compress_indices(self) -> [T; N] {
        let max_size = num_bits::<T>() * N;
        let mut words = T::zero_array::<N>();

        for k in self {
            assert!(k < max_size, "{} is not large enough to hold {}", std::any::type_name::<[T; N]>(), k);
            let word_idx = k / num_bits::<T>();
            let bit_idx = k % num_bits::<T>();
            debug_assert!(word_idx < N);
            unsafe {
                words.get_unchecked_mut(word_idx).set_bit_unchecked(bit_idx)
            }
        }

        words
    }
}

impl<I, T> CompressIndices<Vec<T>> for I 
    where 
        I: Iterator<Item=usize>,
        T: BitBlock,
{
    fn compress_indices(mut self) -> Vec<T> {
        let mut words = match self.next() {
            None => return Vec::new(),
            Some(k) => {
                let word_idx = k / num_bits::<T>();
                let bit_idx = k % num_bits::<T>();
                let mut v = vec![T::zero(); word_idx + 1];
                unsafe { v[word_idx].set_bit_unchecked(bit_idx); };
                v
            }
        };
        for k in self {
            let word_idx = k / num_bits::<T>();
            let bit_idx = k % num_bits::<T>();
            if word_idx >= words.len() {
                words.extend(std::iter::repeat(T::zero()).take(word_idx - words.len() + 1));
            }
            debug_assert!(word_idx < words.len());
            unsafe {
                words.get_unchecked_mut(word_idx).set_bit_unchecked(bit_idx)
            }
        }

        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! compress_word {
        ($t:ty, $correct:expr, $index:expr) => {
            let word : $t = $index.into_iter().compress_indices();
            assert_eq!(word, $correct);
        };
    }

    macro_rules! round_trip {
        (($($t:ty,)+), $tokens:tt) => {
            $(
                round_trip!(@EI $t, $tokens);
            )*
        };

        (@EI $t:ty, ($($indices:expr,)+)) => {
            $(
                round_trip!(@BASE $t, $indices);
            )*
        };

        (@BASE $t:ty, $indices:expr) => {
            let inds = $indices;
            let compressed : $t = inds.into_iter().compress_indices();
            let expanded : Vec<_> = compressed.biterate().collect();
            if &*expanded != &inds {
                eprintln!("Compressed ({}) = {:?}",stringify!($t), &compressed);
                eprintln!(" Correct: {:?}", &inds);
                eprintln!("Biterate: {:?}", &expanded);
                panic!("round trip mismatch");
            }
        };
    }


    #[test]
    fn compress_to_u8() {
        compress_word!(u8, 0b1100_0100, [1, 0, 5]);
    }

    #[should_panic]
    #[test]
    fn out_of_bounds_u8() {
        compress_word!(u8, 0, [8]);
    }

    #[should_panic]
    #[test]
    fn out_of_bounds_u16x4() {
        compress_word!([u16; 4], [0; 4], [64]);
    }

    #[test]
    fn compress_to_u16() {
        compress_word!(u16, 0b1000_1000_0000_1001, [15, 0, 12, 0, 4]);
    }

    #[test]
    fn compress_to_u8x2() {
        compress_word!([u8; 2], [0b1000_1000, 0b0000_1001], [15, 0, 12, 0, 4]);
    }

    #[test]
    fn compress_to_vec_u8() {
        compress_word!(Vec<u8>, vec![0b1000_1000, 0b0000_1001], [15, 0, 12, 0, 4]);
    }

    #[test]
    fn compress_to_vec_16() {
        compress_word!(Vec<u16>, vec![0b1000_1000_0000_1001], [15, 0, 12, 0, 4]);
    }

    #[test]
    fn round_trip_u8() {
        round_trip!(
            (
                u8,
                u16,
                [u8; 2],
                u32,
                u64,
                u128,
                usize,
            ), 
            (
                [0],
                [7],
                [0, 1, 7],
                [1, 2, 3, 4],
            )
        );
    }

    #[test]
    fn round_trip_u16() {
        round_trip!(
            (
                u16,
                [u8; 2],
                u32,
                u64,
                u128,
                usize,
            ), 
            (
                [0, 8],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
            )
        );
    }

    #[test]
    fn round_trip_u64() {
        round_trip!(
            (
                [u8; 8],
                [u16; 4],
                [u32; 2],
                u64,
                u128,
                usize,
            ), 
            (
                [0, 12, 28, 29, 53, 63],
                [0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 63],
            )
        );

    }

    #[test]
    fn round_trip_u128() {
        round_trip!(
            (
                [u32; 4],
                [u64; 2],
                u128,
            ), 
            (
                [0, 12, 28, 29, 53, 63, 76, 82, 85, 99, 101, 102, 103,111, 120, 127],
                [0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 63],
            )
        );
    }
}
