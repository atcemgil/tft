//============================================================================
// Name        : test_valarray.cpp
// Author      : atc
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================


#include "tensor.h"

using namespace std;

int test_nmf();
int test_parafac();
int test_tucker();
int test_matrix_product();


int main(void) {
	//  V = V.*(T'*(MX./(T*V)))./(T'*M);
	//  T = T.*((MX./(T*V))*V')./(M*V');
	test_matrix_product();
	// test_parafac();
	//test_tucker();

}

int test_matrix_product()
{
  Tensor<double> C(3, 3, 2, 2);
  Tensor<double> B(3, 3, 2, 2);
  Tensor<double> A(3, 3, 2, 2);

  cout << "A";
  A.Print();
  cout << "B";
  B.Print();
  cout << "C";
  C.Print();
  Tmult_contract(C, A, B, true);
  //X *= X;
  cout << "C";
  C.Print();
  //return 0;
};

int test_tucker(void) {
	size_t sizes[] = {10, 10, 10, 2, 2, 2}; size_t nDims = 6;

	bool verbose = false;

	Tensor<double> X(nDims, sizes, 0, 1, 2);
	Tensor<double> X_TEMP(nDims, sizes, 0, 1, 2);
	Tensor<double> M(nDims, sizes, 0, 1, 2);
	Tensor<double> MX(nDims, sizes, 0, 1, 2);

//	Tensor<double> Z1(sizes[0], 1, 1, sizes[3], 1, 1);
	Tensor<double> Z1(nDims, sizes, 0, 3);
	Tensor<double> Z2(nDims, sizes, 1, 4);
	Tensor<double> Z3(nDims, sizes, 2, 5);
	Tensor<double> Z4(nDims, sizes, 3, 4, 5);

//	Tensor<double> Z2(1, sizes[1], 1, 1, sizes[4], 1);
//	Tensor<double> Z3(1, 1, sizes[2], 1, 1, sizes[5]);
//	Tensor<double> Z4(1, 1, 1, sizes[3], sizes[4], sizes[5]);

//	Tensor<double> TZ1(sizes[0], 1, 1, sizes[3], 1, 1);
//	Tensor<double> TZ2(1, sizes[1], 1, 1, sizes[4], 1);
//	Tensor<double> TZ3(1, 1, sizes[2], 1, 1, sizes[5]);
//	Tensor<double> TZ4(1, 1, 1, sizes[3], sizes[4], sizes[5]);
	Tensor<double> TZ1(nDims, sizes, 0, 3);
	Tensor<double> TZ2(nDims, sizes, 1, 4);
	Tensor<double> TZ3(nDims, sizes, 2, 5);
	Tensor<double> TZ4(nDims, sizes, 3, 4, 5);

	Tensor<double> DZ1(nDims, sizes, 0, 3);
	Tensor<double> DZ2(nDims, sizes, 1, 4);
	Tensor<double> DZ3(nDims, sizes, 2, 5);
	Tensor<double> DZ4(nDims, sizes, 3, 4, 5);

	Tensor<double> Z1_true(nDims, sizes, 0, 3);
	Tensor<double> Z2_true(nDims, sizes, 1, 4);
	Tensor<double> Z3_true(nDims, sizes, 2, 5);
	Tensor<double> Z4_true(nDims, sizes, 3, 4, 5);

	//	Tensor<double> T345(1, 1, sizes[2], sizes[3], sizes[4], 1);
	//	Tensor<double> T234(1, sizes[1], sizes[2], sizes[3], 1, 1 );
	Tensor<double> T345(nDims, sizes, 2, 3, 4);
	Tensor<double> T234(nDims, sizes, 1, 2, 3);

	Tensor<double> T156(nDims, sizes, 0, 4, 5);
	Tensor<double> T135(nDims, sizes, 0, 2, 4);


//	Tensor<double> T246(1, sizes[1], 1, sizes[3], 1, sizes[5]);
//	Tensor<double> T126(sizes[0], sizes[1], 1, 1, 1, sizes[5]);
	Tensor<double> T246(nDims, sizes, 1, 3, 5);
	Tensor<double> T126(nDims, sizes, 0, 1, 5);


	 M = 1;
	 Z1_true.Randomize();
	 Z2_true.Randomize();
	 Z3_true.Randomize();
	 Z4_true.Randomize();

	 Z1.Randomize();
	 Z2.Randomize();
	 Z3.Randomize();
	 Z4.Randomize();

	 Tmult_contract<double>(T156, Z1_true, Z4_true);
	 Tmult_contract<double>(T126, T156, Z2_true);
	 Tmult_contract<double>(X, T126, Z3_true);
	 Tmult_contract<double>(MX, X, M);

	 for (int e=0;e<5000;e++)
	 {

		// Z1 ***************
		 Tmult_contract<double>(T156, Z1, Z4);
		 Tmult_contract<double>(T126, T156, Z2);
		 Tmult_contract<double>(X_TEMP, T126, Z3);
		 // X_TEMP = MX/X_TEMP;
		 prod_reciprocal(X_TEMP, MX);

		 Tmult_contract<double>(T345, Z3, Z4);
		 Tmult_contract<double>(T234, T345, Z2);
		 Tmult_contract<double>(TZ1, T234, X_TEMP);
		 Tmult_contract<double>(DZ1, T234, M);

		 TZ1 /= DZ1;
		 Z1 *= TZ1;

		 // Z2 ***************
		 Tmult_contract<double>(T156, Z1, Z4);
		 Tmult_contract<double>(T126, T156, Z2);
		 Tmult_contract<double>(X_TEMP, T126, Z3);
		 // X_TEMP = MX/X_TEMP;
		 prod_reciprocal(X_TEMP, MX);

		 Tmult_contract<double>(T156, Z1, Z4);
		 Tmult_contract<double>(T135, T156, Z3);
		 Tmult_contract<double>(TZ2, T135, X_TEMP);
		 Tmult_contract<double>(DZ2, T135, M);

		 TZ2 /= DZ2;
		 Z2 *= TZ2;

		 // Z3 ***************
		 Tmult_contract<double>(T156, Z1, Z4);
		 Tmult_contract<double>(T126, T156, Z2);
		 Tmult_contract<double>(X_TEMP, T126, Z3);
		 // X_TEMP = MX/X_TEMP;
		 prod_reciprocal(X_TEMP, MX);

		 Tmult_contract<double>(T246, Z2, Z4);
		 Tmult_contract<double>(T126, T246, Z1);
		 Tmult_contract<double>(TZ3, T126, X_TEMP);
		 Tmult_contract<double>(DZ3, T126, M);

		 TZ3 /= DZ3;
		 Z3 *= TZ3;

		 // Z4 ***************
		 Tmult_contract<double>(T156, Z1, Z4);
		 Tmult_contract<double>(T126, T156, Z2);
		 Tmult_contract<double>(X_TEMP, T126, Z3);
		 // X_TEMP = MX/X_TEMP;
		 prod_reciprocal(X_TEMP, MX);

		 Tmult_contract<double>(T126, X_TEMP, Z3);
		 Tmult_contract<double>(T156, T126, Z2);
		 Tmult_contract<double>(TZ4, T156, Z1);

		 Tmult_contract<double>(T126, M, Z3);
		 Tmult_contract<double>(T156, T126, Z2);
		 Tmult_contract<double>(DZ4, T156, Z1);

		 TZ4 /= DZ4;
		 Z4 *= TZ4;

		 // Compute output --------------------------------
		 if (e%100==0)
		 cout << " Epoch " << e << endl;

		 if (verbose)
		 {
			 Tmult_contract<double>(T156, Z1, Z4);
			 Tmult_contract<double>(T126, T156, Z2);
			 Tmult_contract<double>(X_TEMP, T126, Z3);

			 cout << *MX.Data_ptr();
			 cout << *X_TEMP.Data_ptr();
		 }
	 }

	 cout << *Z1_true.Data_ptr();
	 cout << *Z1.Data_ptr();

	 cout << *Z4_true.Data_ptr();
	 cout << *Z4.Data_ptr();

	 Z4.Print();

	 return 0;
};

int test_parafac(void) {
	size_t sizes[] = {2, 2, 2, 3};
	Tensor<double> X(sizes[0], sizes[1], sizes[2], 1);
	Tensor<double> X_TEMP(sizes[0], sizes[1], sizes[2], 1);
	Tensor<double> M(sizes[0], sizes[1], sizes[2], 1);
	Tensor<double> MX(sizes[0], sizes[1], sizes[2], 1);

	Tensor<double> Z1(sizes[0], 1, 1, sizes[3]);
	Tensor<double> Z2(1, sizes[1], 1, sizes[3]);
	Tensor<double> Z3(1, 1, sizes[2], sizes[3]);

	Tensor<double> TZ1(sizes[0], 1, 1, sizes[3]);
	Tensor<double> TZ2(1, sizes[1], 1, sizes[3]);
	Tensor<double> TZ3(1, 1, sizes[2], sizes[3]);
	Tensor<double> DZ1(sizes[0], 1, 1, sizes[3]);
	Tensor<double> DZ2(1, sizes[1], 1, sizes[3]);
	Tensor<double> DZ3(1, 1, sizes[2], sizes[3]);

	Tensor<double> Z1_true(sizes[0], 1, 1, sizes[3]);
	Tensor<double> Z2_true(1, sizes[1], 1, sizes[3]);
	Tensor<double> Z3_true(1, 1, sizes[2], sizes[3]);

	Tensor<double> TZ12(sizes[0], sizes[1], 1, sizes[3]);
	Tensor<double> TZ23(1, sizes[1], sizes[2], sizes[3]);
	Tensor<double> TZ31(sizes[0], 1, sizes[2], sizes[3]);

	 M = 1;
	 Z1_true.Randomize();
	 Z2_true.Randomize();
	 Z3_true.Randomize();

	 Z1.Randomize();
	 Z2.Randomize();
	 Z3.Randomize();

	 // Compute hat(X), the model prediction
	 Tmult_contract<double>(TZ12, Z1_true, Z2_true);
	 Tmult_contract<double>(X, TZ12, Z3_true);
	 Tmult_contract<double>(MX, X, M);

	 for (int e=0;e<50;e++)
	 {

		// Z1 ***************
		 Tmult_contract<double>(TZ12, Z1, Z2);
		 Tmult_contract<double>(X_TEMP, TZ12, Z3);
		 // X_TEMP = MX/X_TEMP;
		 prod_reciprocal(X_TEMP, MX);

		 Tmult_contract<double>(TZ12, X_TEMP, Z3);
		 Tmult_contract<double>(TZ1, TZ12, Z2);

		 Tmult_contract<double>(TZ12, M, Z3);
		 Tmult_contract<double>(DZ1, TZ12, Z2);

		 TZ1 /= DZ1;
		 Z1 *= TZ1;

		// Z2 ***************
		 Tmult_contract<double>(TZ23, Z2, Z3);
		 Tmult_contract<double>(X_TEMP, TZ23, Z1);
		 // X_TEMP = MX/X_TEMP;
		 prod_reciprocal(X_TEMP, MX);

		 Tmult_contract<double>(TZ23, X_TEMP, Z1);
		 Tmult_contract<double>(TZ2, TZ23, Z3);

		 Tmult_contract<double>(TZ23, M, Z1);
		 Tmult_contract<double>(DZ2, TZ23, Z3);

		 TZ2 /= DZ2;
		 Z2 *= TZ2;

		// Z3 ***************
		Tmult_contract<double>(TZ31, Z3, Z1);
		Tmult_contract<double>(X_TEMP, TZ31, Z2);
		// X_TEMP = MX/X_TEMP;
		prod_reciprocal(X_TEMP, MX);

		Tmult_contract<double>(TZ31, X_TEMP, Z2);
		Tmult_contract<double>(TZ3, TZ31, Z1);

		Tmult_contract<double>(TZ31, M, Z2);
		Tmult_contract<double>(DZ3, TZ31, Z1);

		TZ3 /= DZ3;
		Z3 *= TZ3;

		// --------------------------------
		Tmult_contract<double>(TZ31, Z3, Z1);
		Tmult_contract<double>(X_TEMP, TZ31, Z2);
		cout << " Epoch " << e << endl;
		cout << *MX.Data_ptr();
		cout << *X_TEMP.Data_ptr();

	 }

	 return 0;

}

int test_nmf(void) {
	size_t sizes[] = {5, 4, 2};
//	int nDims = 3;	//size_t strides[nDims];

//	 Tensor<double> C(sizes[0], sizes[1], sizes[2]);
	 Tensor<double> X(sizes[0], 1, sizes[2]);
	 Tensor<double> X_TEMP(sizes[0], 1, sizes[2]);
	 Tensor<double> M(sizes[0], 1, sizes[2]);
	 Tensor<double> V(1, sizes[1], sizes[2]);
	 Tensor<double> V_true(1, sizes[1], sizes[2]);
	 Tensor<double> D_V(1, sizes[1], sizes[2]);
	 Tensor<double> D_V2(1, sizes[1], sizes[2]);
	 Tensor<double> T(sizes[0], sizes[1], 1);
	 Tensor<double> T_true(sizes[0], sizes[1], 1);
	 Tensor<double> D_T(sizes[0], sizes[1], 1);
	 Tensor<double> D_T2(sizes[0], sizes[1], 1);

	 Tensor<double> MX(sizes[0], 1, sizes[2]);

	 // T'*M

	 M = 1;
	 T_true.Randomize();
	 V_true.Randomize();
	 Tmult_contract<double>(X, T_true, V_true);

	 T.Randomize();
	 V.Randomize();


	 Tmult_contract<double>(MX, X, M);

	 //---------------------------

	 for (int e=0;e<100;e++)
	 {
	 Tmult_contract<double>(X_TEMP, T, V);
	 // X_TEMP = MX/X_TEMP;
	 prod_reciprocal(X_TEMP, MX);

	 Tmult_contract<double>(D_V, T, M);
	 Tmult_contract<double>(D_V2, T, X_TEMP);

	 D_V2 /= D_V;
	 V *= D_V2;

	 Tmult_contract<double>(X_TEMP, T, V);
	 // X_TEMP = MX/X_TEMP;
	 prod_reciprocal(X_TEMP, MX);

	 Tmult_contract<double>(D_T, V, M);
	 Tmult_contract<double>(D_T2, V, X_TEMP);

	 D_T2 /= D_T;
	 T *= D_T2;

	 Tmult_contract<double>(X_TEMP, T, V);

	 cout << " Epoch " << e << endl;
	 cout << *X.Data_ptr();
	 cout << *X_TEMP.Data_ptr();

	 }


	 return 0;

}



int test_kronecker_product(void) {
	size_t sizes[] = {3, 5, 4, 2};
	int nDims = 4;	size_t strides[nDims];

	size_t sz = 1;
	 for (int i=0; i< nDims; i++) {
		strides[i] = sz;
		sz *= sizes[i];
	};


//	 Tensor<double> C(sizes[0], sizes[1], sizes[2]);
	 Tensor<double> C(sizes[0], sizes[1], sizes[2], sizes[3]);
	 Tensor<double> A(sizes[0], sizes[1], 1, 1);
	 Tensor<double> B(1, 1, sizes[2], sizes[3]);

	 C.Print("C");

	 valarray<size_t>* ps = C.Strides_ptr();

	 (*ps)[0] = sizes[2];
	 (*ps)[1] = sizes[0]*sizes[2]*sizes[3];
	 (*ps)[2] = 1;
	 (*ps)[3] = sizes[0]*sizes[2];

	 A.Print("A");
	 cout << *A.Strides_ptr();
	 B.Print("B");
	 cout << *B.Strides_ptr();
	 C.Print();
	 cout << *C.Strides_ptr();

	 cout << "---------------" << endl;

	 Tmult_contract<double>(C, A, B);

	 C.Print();

	 cout << *C.Data_ptr();


	 return 0;

}

int old_main(void) {

	size_t sizes[] = {3, 2, 4};
	int nDims = 3;	size_t strides[3];

	size_t sz = 1;
	 for (int i=0; i< nDims; i++) {
		strides[i] = sz;
		sz *= sizes[i];
	};

	 rgslice_iter gi(0, valarray<size_t>(sizes, nDims), valarray<size_t>(strides, nDims) );

	 for (int i=0; i<gi.Cardinality(); i++ ) {
		 gi.Print();
		 gi++;
	 }


	valarray<double> tensor(sz);

	for (int i=0; i< sz; i++) {
		tensor[i] = double(i)+1.0;
	};

	for (int i=0; i< sz; i++) {
		cout << tensor[i] << " ";
	};
	cout << endl;

//	slice row(0, sizes[1], strides[1]);
	slice row(0, 3, 1);
	valarray<double> fiber = tensor[row];
	slice_array<double> fbr = tensor[row];
	slice_array<double> fb = tensor[slice(3, 3, 1)];

	for (int i=0; i< sz; i++) {
		cout << tensor[i] << " ";
	};
	cout << endl;
	for (size_t i=0; i< fiber.size(); i++) {
		cout << fiber[i] << " ";
	};
	cout << endl;

	valarray<size_t> tsizes(3);
	tsizes[0] = sizes[2];
	tsizes[1] = sizes[1];
	tsizes[2] = sizes[0];

	valarray<size_t> tstrides(3);
	tstrides[0] = strides[2];
	tstrides[1] = strides[1];
	tstrides[2] = 0;

	gslice face(0, tsizes, tstrides);

	valarray<double> pl = tensor[face];
	gslice_array<double> repface = tensor[face];

	gslice_array<double> plane = tensor[face];



	for (size_t i=0; i< pl.size(); i++) {
		cout << pl[i] << " ";
	};
	cout << endl;



	tensor += repface;

	for (size_t i=0; i< tensor.size(); i++) {
		cout << tensor[i] << " ";
	};
	cout << endl;

	Tensor<double> ts(1, 12, 2, 1);

	// ts.Print();

	return EXIT_SUCCESS;
}
