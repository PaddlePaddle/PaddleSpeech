// matrix/kaldi-vector.h

// Copyright 2009-2012   Ondrej Glembek;  Microsoft Corporation;  Lukas Burget;
//                       Saarland University (Author: Arnab Ghoshal);
//                       Ariya Rastrow;  Petr Schwarz;  Yanmin Qian;
//                       Karel Vesely;  Go Vivace Inc.;  Arnab Ghoshal
//                       Wei Shi;
//                2015   Guoguo Chen
//                2017   Daniel Galvez
//                2019   Yiwen Shao

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_MATRIX_KALDI_VECTOR_H_
#define KALDI_MATRIX_KALDI_VECTOR_H_ 1

#include "matrix/matrix-common.h"

namespace kaldi {

/// \addtogroup matrix_group
/// @{

///  Provides a vector abstraction class.
///  This class provides a way to work with vectors in kaldi.
///  It encapsulates basic operations and memory optimizations.
template <typename Real>
class VectorBase {
  public:
    /// Set vector to all zeros.
    void SetZero();

    /// Returns true if matrix is all zeros.
    bool IsZero(Real cutoff = 1.0e-06) const;  // replace magic number

    /// Set all members of a vector to a specified value.
    void Set(Real f);

    /// Returns the  dimension of the vector.
    inline MatrixIndexT Dim() const { return dim_; }

    /// Returns the size in memory of the vector, in bytes.
    inline MatrixIndexT SizeInBytes() const { return (dim_ * sizeof(Real)); }

    /// Returns a pointer to the start of the vector's data.
    inline Real *Data() { return data_; }

    /// Returns a pointer to the start of the vector's data (const).
    inline const Real *Data() const { return data_; }

    /// Indexing  operator (const).
    inline Real operator()(MatrixIndexT i) const {
        KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                              static_cast<UnsignedMatrixIndexT>(dim_));
        return *(data_ + i);
    }

    /// Indexing operator (non-const).
    inline Real &operator()(MatrixIndexT i) {
        KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                              static_cast<UnsignedMatrixIndexT>(dim_));
        return *(data_ + i);
    }

    /** @brief Returns a sub-vector of a vector (a range of elements).
     *  @param o [in] Origin, 0 < o < Dim()
     *  @param l [in] Length 0 < l < Dim()-o
     *  @return A SubVector object that aliases the data of the Vector object.
     *  See @c SubVector class for details   */
    SubVector<Real> Range(const MatrixIndexT o, const MatrixIndexT l) {
        return SubVector<Real>(*this, o, l);
    }

    /** @brief Returns a const sub-vector of a vector (a range of elements).
     *  @param o [in] Origin, 0 < o < Dim()
     *  @param l [in] Length 0 < l < Dim()-o
     *  @return A SubVector object that aliases the data of the Vector object.
     *  See @c SubVector class for details   */
    const SubVector<Real> Range(const MatrixIndexT o,
                                const MatrixIndexT l) const {
        return SubVector<Real>(*this, o, l);
    }

    /// Copy data from another vector (must match own size).
    void CopyFromVec(const VectorBase<Real> &v);

    /// Copy data from another vector of different type (double vs. float)
    template <typename OtherReal>
    void CopyFromVec(const VectorBase<OtherReal> &v);

    /// Performs a row stack of the matrix M
    void CopyRowsFromMat(const MatrixBase<Real> &M);
    template <typename OtherReal>
    void CopyRowsFromMat(const MatrixBase<OtherReal> &M);

    /// Performs a column stack of the matrix M
    void CopyColsFromMat(const MatrixBase<Real> &M);

    /// Extracts a row of the matrix M.  Could also do this with
    /// this->Copy(M[row]).
    void CopyRowFromMat(const MatrixBase<Real> &M, MatrixIndexT row);
    /// Extracts a row of the matrix M with type conversion.
    template <typename OtherReal>
    void CopyRowFromMat(const MatrixBase<OtherReal> &M, MatrixIndexT row);

    /// Extracts a column of the matrix M.
    template <typename OtherReal>
    void CopyColFromMat(const MatrixBase<OtherReal> &M, MatrixIndexT col);

    /// Reads from C++ stream (option to add to existing contents).
    /// Throws exception on failure
    void Read(std::istream &in, bool binary);

    /// Writes to C++ stream (option to write in binary).
    void Write(std::ostream &Out, bool binary) const;

    friend class VectorBase<double>;
    friend class VectorBase<float>;

  protected:
    /// Destructor;  does not deallocate memory, this is handled by child
    /// classes.
    /// This destructor is protected so this object can only be
    /// deleted via a child.
    ~VectorBase() {}

    /// Empty initializer, corresponds to vector of zero size.
    explicit VectorBase() : data_(NULL), dim_(0) {
        KALDI_ASSERT_IS_FLOATING_TYPE(Real);
    }

    /// data memory area
    Real *data_;
    /// dimension of vector
    MatrixIndexT dim_;
    KALDI_DISALLOW_COPY_AND_ASSIGN(VectorBase);
};  // class VectorBase

/** @brief A class representing a vector.
 *
 *  This class provides a way to work with vectors in kaldi.
 *  It encapsulates basic operations and memory optimizations.  */
template <typename Real>
class Vector : public VectorBase<Real> {
  public:
    /// Constructor that takes no arguments.  Initializes to empty.
    Vector() : VectorBase<Real>() {}

    /// Constructor with specific size.  Sets to all-zero by default
    /// if set_zero == false, memory contents are undefined.
    explicit Vector(const MatrixIndexT s,
                    MatrixResizeType resize_type = kSetZero)
        : VectorBase<Real>() {
        Resize(s, resize_type);
    }

    /// Copy constructor from CUDA vector
    /// This is defined in ../cudamatrix/cu-vector.h
    // template<typename OtherReal>
    // explicit Vector(const CuVectorBase<OtherReal> &cu);

    /// Copy constructor.  The need for this is controversial.
    Vector(const Vector<Real> &v)
        : VectorBase<Real>() {  //  (cannot be explicit)
        Resize(v.Dim(), kUndefined);
        this->CopyFromVec(v);
    }

    /// Copy-constructor from base-class, needed to copy from SubVector.
    explicit Vector(const VectorBase<Real> &v) : VectorBase<Real>() {
        Resize(v.Dim(), kUndefined);
        this->CopyFromVec(v);
    }

    /// Type conversion constructor.
    template <typename OtherReal>
    explicit Vector(const VectorBase<OtherReal> &v) : VectorBase<Real>() {
        Resize(v.Dim(), kUndefined);
        this->CopyFromVec(v);
    }

    // Took this out since it is unsafe : Arnab
    //  /// Constructor from a pointer and a size; copies the data to a location
    //  /// it owns.
    //  Vector(const Real* Data, const MatrixIndexT s): VectorBase<Real>() {
    //    Resize(s);
    //    CopyFromPtr(Data, s);
    //  }


    /// Swaps the contents of *this and *other.  Shallow swap.
    void Swap(Vector<Real> *other);

    /// Destructor.  Deallocates memory.
    ~Vector() { Destroy(); }

    /// Read function using C++ streams.  Can also add to existing contents
    /// of matrix.
    void Read(std::istream &in, bool binary);

    /// Set vector to a specified size (can be zero).
    /// The value of the new data depends on resize_type:
    ///   -if kSetZero, the new data will be zero
    ///   -if kUndefined, the new data will be undefined
    ///   -if kCopyData, the new data will be the same as the old data in any
    ///      shared positions, and zero elsewhere.
    /// This function takes time proportional to the number of data elements.
    void Resize(MatrixIndexT length, MatrixResizeType resize_type = kSetZero);

    /// Remove one element and shifts later elements down.
    void RemoveElement(MatrixIndexT i);

    /// Assignment operator.
    Vector<Real> &operator=(const Vector<Real> &other) {
        Resize(other.Dim(), kUndefined);
        this->CopyFromVec(other);
        return *this;
    }

    /// Assignment operator that takes VectorBase.
    Vector<Real> &operator=(const VectorBase<Real> &other) {
        Resize(other.Dim(), kUndefined);
        this->CopyFromVec(other);
        return *this;
    }

  private:
    /// Init assumes the current contents of the class are invalid (i.e. junk or
    /// has already been freed), and it sets the vector to newly allocated
    /// memory
    /// with the specified dimension.  dim == 0 is acceptable.  The memory
    /// contents
    /// pointed to by data_ will be undefined.
    void Init(const MatrixIndexT dim);

    /// Destroy function, called internally.
    void Destroy();
};


/// Represents a non-allocating general vector which can be defined
/// as a sub-vector of higher-level vector [or as the row of a matrix].
template <typename Real>
class SubVector : public VectorBase<Real> {
  public:
    /// Constructor from a Vector or SubVector.
    /// SubVectors are not const-safe and it's very hard to make them
    /// so for now we just give up.  This function contains const_cast.
    SubVector(const VectorBase<Real> &t,
              const MatrixIndexT origin,
              const MatrixIndexT length)
        : VectorBase<Real>() {
        // following assert equiv to origin>=0 && length>=0 &&
        // origin+length <= rt.dim_
        KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(origin) +
                         static_cast<UnsignedMatrixIndexT>(length) <=
                     static_cast<UnsignedMatrixIndexT>(t.Dim()));
        VectorBase<Real>::data_ = const_cast<Real *>(t.Data() + origin);
        VectorBase<Real>::dim_ = length;
    }

    /// This constructor initializes the vector to point at the contents
    /// of this packed matrix (SpMatrix or TpMatrix).
    // SubVector(const PackedMatrix<Real> &M) {
    // VectorBase<Real>::data_ = const_cast<Real*> (M.Data());
    // VectorBase<Real>::dim_   = (M.NumRows()*(M.NumRows()+1))/2;
    //}

    /// Copy constructor
    SubVector(const SubVector &other) : VectorBase<Real>() {
        // this copy constructor needed for Range() to work in base class.
        VectorBase<Real>::data_ = other.data_;
        VectorBase<Real>::dim_ = other.dim_;
    }

    /// Constructor from a pointer to memory and a length.  Keeps a pointer
    /// to the data but does not take ownership (will never delete).
    /// Caution: this constructor enables you to evade const constraints.
    SubVector(const Real *data, MatrixIndexT length) : VectorBase<Real>() {
        VectorBase<Real>::data_ = const_cast<Real *>(data);
        VectorBase<Real>::dim_ = length;
    }

    /// This operation does not preserve const-ness, so be careful.
    SubVector(const MatrixBase<Real> &matrix, MatrixIndexT row) {
        VectorBase<Real>::data_ = const_cast<Real *>(matrix.RowData(row));
        VectorBase<Real>::dim_ = matrix.NumCols();
    }

    ~SubVector() {}  ///< Destructor (does nothing; no pointers are owned here).

  private:
    /// Disallow assignment operator.
    SubVector &operator=(const SubVector &other) {}
};

/// @} end of "addtogroup matrix_group"
/// \addtogroup matrix_funcs_io
/// @{
/// Output to a C++ stream.  Non-binary by default (use Write for
/// binary output).
template <typename Real>
std::ostream &operator<<(std::ostream &out, const VectorBase<Real> &v);

/// Input from a C++ stream.  Will automatically read text or
/// binary data from the stream.
template <typename Real>
std::istream &operator>>(std::istream &in, VectorBase<Real> &v);

/// Input from a C++ stream. Will automatically read text or
/// binary data from the stream.
template <typename Real>
std::istream &operator>>(std::istream &in, Vector<Real> &v);
/// @} end of \addtogroup matrix_funcs_io

/// \addtogroup matrix_funcs_scalar
/// @{


// template<typename Real>
// bool ApproxEqual(const VectorBase<Real> &a,
// const VectorBase<Real> &b, Real tol = 0.01) {
// return a.ApproxEqual(b, tol);
//}

// template<typename Real>
// inline void AssertEqual(VectorBase<Real> &a, VectorBase<Real> &b,
// float tol = 0.01) {
// KALDI_ASSERT(a.ApproxEqual(b, tol));
//}


}  // namespace kaldi

// we need to include the implementation
#include "matrix/kaldi-vector-inl.h"


#endif  // KALDI_MATRIX_KALDI_VECTOR_H_
