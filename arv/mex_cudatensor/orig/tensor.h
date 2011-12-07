/*
 * tensor.h
 *
 *  Created on: 10 Aug 2010
 *      Author: cemgil
 */

#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <valarray>
#include <algorithm>

using namespace std;
/* flips a valarray
 *
 */
template<class T> void reverse(valarray<T>& va)
{
	for (size_t b=0, e=va.size()-1 ; e>b; e--, b++)
	{
		swap(va[b], va[e]);
	};
};


/*
 * rgslice_iter : (fastest index first gslice iterator)
 *
 * Converts a linear sequential index to a strided array index.
 *
 * size_t sz = {3,2,5};
 * valarray<size_t>  sizes(sz, 3);
 * valarray<size_t>  strides(3); strides[0]=1; strides[1]=sz[0]; strides[2]=sz[1];
 *
 * rgslice gi(0, sizes, strides);  // initialize to (0,0,0)
 * gi++;  // Returns current value than increments by one.
 * gi.Reset();
 *
 * for (int i=0; i<sz; i++ ) C[giC++] += A[giB++]*B[giB++];
 *
 * Note:
 * The standard library object gslice is constructed via gslice(size_t offset, valarray<> sizes, valarray<> strides).
 * The parameters 'sizes' and 'strides' store the index information. The convention is that the fastest index
 * is the last element. rgslice_iterator assumes the reverse and assumes that the fastest index is stored first.
 *
 *
 */
class rgslice_iter {
	valarray<size_t> *psizes;
	valarray<size_t> *pstrides;
	size_t offset;
	size_t dim;
	size_t cardinality;
	public:
	valarray<size_t> *pcur;

	rgslice_iter(const size_t ofs, valarray<size_t> sizes, valarray<size_t> strides) {
		psizes = new valarray<size_t>(sizes);
		pstrides = new valarray<size_t>(strides);
		offset = ofs;
		dim = sizes.size();
		pcur = new valarray<size_t>(dim);
		cardinality = 1;
		for (size_t i=0; i<dim;i++) cardinality *= (*psizes)[i];
	}

	~rgslice_iter() {
		delete pcur;
		delete psizes;
		delete pstrides;
	};

	void Print() {
		for (size_t i=0; i< (*pcur).size(); i++ ) cout << (*pcur)[i] << " ";
		cout << endl;
	}

	void Reset() {
		*pcur = 0;
	}

	size_t Cardinality() {
		return cardinality;
	}

	size_t Index() {
		return ((*pcur)*(*pstrides)).sum() + offset;
	}

//	operator unsigned int() {return Index();}

	operator gslice() {
		valarray<size_t> sz = *psizes; reverse(sz);
		valarray<size_t> st = *pstrides; reverse(st);
		return gslice(offset, sz, st);
	}

	size_t operator++(int)
	{
		size_t past_index = Index();
		size_t rem = 0;
		(*pcur)[0] += 1;
		for (unsigned int i=0; i<dim; i++) {
			rem = (*pcur)[i] / (*psizes)[i];
			(*pcur)[i] %= (*psizes)[i];
			if (rem == 0) {break;}
			else {
				if (i<dim-1) (*pcur)[i+1]+=1;
			}
		}
		return past_index;
	}

};

/*
 *  A slice iterator specialised to rectangular tensor arrays
 */

class tensor_iter {
	valarray<size_t> *pcur;
	valarray<size_t> *psizes;
	size_t dim;
	size_t cardinality;
	public:

	tensor_iter(valarray<size_t> sizes) {
		psizes = new valarray<size_t>(sizes);
		dim = sizes.size();
		pcur = new valarray<size_t>(dim);
		cardinality = 1;
		for (size_t i=0; i<dim;i++) cardinality *= (*psizes)[i];
	}

	~tensor_iter() {
		delete pcur;
		delete psizes;
	};

	void Print() {
		for (size_t i=0; i< (*pcur).size(); i++ ) cout << (*pcur)[i] << " ";
		cout << endl;
	}

	void Reset() {
		*pcur = 0;
	}

	valarray<size_t>* Current() {
		return pcur;
	};

	size_t Cardinality() {
		return cardinality;
	}

	valarray<size_t>* operator++()
	{
		size_t rem = 0;
		(*pcur)[0] += 1;
		for (unsigned int i=0; i<dim; i++) {
			rem = (*pcur)[i] / (*psizes)[i];
			(*pcur)[i] %= (*psizes)[i];
			if (rem == 0) {break;}
			else {
				if (i<dim-1) (*pcur)[i+1]+=1;
			}
		}
		return pcur;
	}

};


template <class T> ostream& operator<<(ostream& co, const valarray<T>& va ) {
	for (size_t i=0; i<va.size(); i++) {
		co << va[i] << " ";
	}
	co << endl;
	return co;
}


ostream& operator<<(ostream& co, const gslice& gs ) {
	for (size_t i=0; i< gs.size().size(); i++) {
		 co << gs.size()[i] << " " << gs.stride()[i] << endl;
	}
	co << endl;
	return co;
}


/*
 *
 *
 */
template <typename T> class Tensor {
	unsigned int dim;
	size_t total_size;
	valarray<T> * va;
	valarray<size_t> * sizes;
	valarray<size_t> * strides;

public:
	valarray<size_t>* Sizes_ptr() {return sizes;};
	valarray<size_t>* Strides_ptr() {return strides;};
	valarray<T>* Data_ptr() {return va;};

	size_t Dim() {return dim;};
	size_t Size(unsigned int d){
		if (d>=dim) return 1;
		else return (*sizes)[d];
	};

	size_t Stride(unsigned int d){
		if (d>=dim) return 0;
		else return (*strides)[dim-1];  //???
	};


	void Init(size_t nDims, size_t* szs) {
		sizes = new valarray<size_t>(nDims);
		strides = new valarray<size_t>(nDims);
		size_t sz = 1;
		for (size_t i=0; i< nDims; i++) {
			(*strides)[i] = szs[i]>1 ? sz : 0;
			(*sizes)[i] = szs[i];
			sz *= szs[i];
		};
		total_size = sz;

		va = new valarray<T>(sz);

		for (size_t i=0; i< sz; i++) {
		  (*va)[i] = T(i) + 1; //???
		};
		dim = nDims;
	};


	Tensor(size_t nDims, size_t* szs) {
		Init(nDims, szs);
	};

	// Specify all sizes but only member indices i_d where 0<=d<nDims,  d>nDims are ignored
	// all the remaining sizes are set to 1
	Tensor(size_t nDims, size_t* szs, size_t idx1, size_t idx2 = size_t(-1), size_t idx3 = size_t(-1), size_t idx4 = size_t(-1) ) {
		size_t* sz = new size_t[nDims];
		for (size_t i=0; i<nDims; i++) sz[i] = 1;

#define TEST_AND_SET(par)	if (idx##par>=0 && idx##par<nDims) sz[idx##par] = szs[idx##par]
		TEST_AND_SET(1);
		TEST_AND_SET(2);
		TEST_AND_SET(3);
		TEST_AND_SET(4);

		Init(nDims, sz);
		delete [] sz;
	};


	Tensor(size_t m, size_t n = 1){
		size_t dm = 2;
		size_t sizes[2]; sizes[0] = m; sizes[1] = n;
		Init(dm, sizes);
	};

	Tensor(size_t m, size_t n, size_t k) {
		size_t dm = 3;
		size_t sizes[3]; sizes[0] = m; sizes[1] = n; sizes[2] = k;
		Init(dm, sizes);
	};

	Tensor(size_t m, size_t n, size_t k, size_t l) {
		size_t dm = 4;
		size_t sizes[4]; sizes[0] = m; sizes[1] = n;  sizes[2] = k; sizes[3] = l;
		Init(dm, sizes);
	};

	Tensor(size_t m, size_t n, size_t k, size_t l, size_t o) {
		size_t dm = 5;
		size_t sizes[5]; sizes[0] = m; sizes[1] = n;  sizes[2] = k; sizes[3] = l; sizes[4] = o;
		Init(dm, sizes);
	};

	Tensor(size_t m, size_t n, size_t k, size_t l, size_t o, size_t p) {
		size_t dm = 6;
		size_t sizes[6]; sizes[0] = m; sizes[1] = n;  sizes[2] = k; sizes[3] = l; sizes[4] = o; sizes[5] = p;
		Init(dm, sizes);
	};


	~Tensor() {
		delete va;
		delete sizes;
		delete strides;
	};

	inline T& operator[](valarray<size_t>* pcur) {
//		size_t idx = ((*pcur)*(*strides)).sum();
//		cout << idx << endl;
		return (*va)[((*pcur)*(*strides)).sum()];
	};

	T& operator[](tensor_iter& ti) {
//		size_t idx = ((*ti.Current())*(*strides)).sum();
//		cout << idx << endl;
		return (*va)[   (      (*ti.Current())    *      (*strides)     ).sum()     ];
	};

	T operator=(T val) {
		*va = val;
		return val;
	}

	void Randomize() {
		for (size_t i=0; i< (*va).size(); i++ )
			(*va)[i] = T(rand()/double(RAND_MAX));
	}

	void Print(const char* str = "") {
		cout << str << endl << *va;
	};

	void oldPrint(const char* str = "") {
		size_t M = (*sizes)[0];
		size_t N = (*sizes)[1];

		size_t rest = total_size/(M*N);
		cout << str << endl;
		for (size_t k=0; k< rest; k++) {
			cout << "-- Face " << k << " ------ " << endl;
			for (size_t i=0; i< M; i++) {
				for (size_t j=0; j< N; j++) {
					cout << (*va)[k*(*strides)[2]+j*(*strides)[1]+i] << " ";
				};
				cout << endl;
			};
		};
		cout << endl;

	};
};

/* Multiplies two tensors and contracts on the indicies of the target
 */
template <class T> void Tmult_contract(Tensor<T>& C, Tensor<T>& A, Tensor<T>& B, bool reset_target = true) {
	size_t dimA = A.Dim();
	size_t dimB = B.Dim();
	size_t dimC = C.Dim();

	if ((dimA != dimB) || (dimA != dimC) || (dimC != dimB)) {
		cout << "Dimensions mismatch!" << endl;
		return;
	}

	valarray<size_t> sizes(dimA);
	valarray<size_t>* sA = A.Sizes_ptr();
	valarray<size_t>* sB = B.Sizes_ptr();
	valarray<size_t>* sC = C.Sizes_ptr();

	for (int i=0; i< dimA; i++) {
		sizes[i] = max(max((*sA)[i], (*sB)[i]), (*sC)[i]);
	}
	 tensor_iter ti(sizes);

	 ti.Reset();

	 if (reset_target) *(C.Data_ptr()) = 0;

	 for (unsigned int i=0; i<ti.Cardinality(); i++) {
		 C[ti] += A[ti]*B[ti];
		 ++ti;
	 }
}

template <class T> Tensor<T>& operator /=(Tensor<T>& C, Tensor<T>& A) {
	(*C.Data_ptr()) /= (*A.Data_ptr());
	return C;
}

template <class T> Tensor<T>& operator *=(Tensor<T>& C, Tensor<T>& A) {
	(*C.Data_ptr()) *= (*A.Data_ptr());
	return C;
}

template <class T> Tensor<T>& operator +=(Tensor<T>& C, Tensor<T>& A) {
	(*C.Data_ptr()) += (*A.Data_ptr());
	return C;
}

template <class T> Tensor<T>& operator -=(Tensor<T>& C, Tensor<T>& A) {
	(*C.Data_ptr()) += (*A.Data_ptr());
	return C;
}

template <class T> void prod_reciprocal(Tensor<T>& C, Tensor<T>& A) {
// C = A/C
	valarray<T>* pC = C.Data_ptr();
	valarray<T>* pA = A.Data_ptr();

	for (unsigned int i=0; i<(*pC).size(); i++)
	{
		(*pC)[i] = (*pA)[i]/(*pC)[i];
	};

}


#endif /* TENSOR_H_ */
