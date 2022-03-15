# `biterate`

A simple crate for iterating over the bits in bytes.

The crate provides 2 functionalities, exposed as traits:

- Iterating over the 1-bits in unsigned integer types and sequences thereof
- The reverse: Constructing integers from indices of the 1-bits

It seems stupid to publish something so simple as a crate, but I've ended up writing this code so many times (and getting it wrong) I figured I should put it a crate with actual tests.
