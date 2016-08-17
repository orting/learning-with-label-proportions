/*
  Test matrix operations
 */

#include <algorithm>
#include <random>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "Eigen/Dense"

#include "gtest/gtest.h"

#include "Util/MatrixOperations.h"

class CoOccurenceMatrixTest : public ::testing::Test {
public:
  typedef Eigen::Matrix< double,
		       Eigen::Dynamic,
		       Eigen::Dynamic,
		       Eigen::RowMajor > MatrixType;  

protected:
  virtual void SetUp() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> disSize(3, 500);
    std::uniform_int_distribution<size_t> disMax(3, 50);

    A.resize( disSize( gen ) );
    B.resize( A.size() );
    C.resize( disSize( gen ) );

    maxA = disMax( gen );
    maxB = disMax( gen );
    maxC = disMax( gen );
    std::uniform_int_distribution<size_t> disA(0, maxA);
    std::uniform_int_distribution<size_t> disB(0, maxB);
    std::uniform_int_distribution<size_t> disC(0, maxC);
    std::generate( A.begin(), A.end(), [&gen,&disA]{ return disA( gen ) ; });
    std::generate( B.begin(), B.end(), [&gen,&disB]{ return disB( gen ) ; });
    std::generate( C.begin(), C.end(), [&gen,&disC]{ return disC( gen ) ; });
    
  }
  std::vector< size_t > A, B, C;
  size_t maxA, maxB, maxC;
};
    

TEST_F( CoOccurenceMatrixTest, CoOccurenceWithSelfIsDiagonalMatrix ) {
  MatrixType out = MatrixType::Zero( maxA+1, maxA+1 );
  coOccurenceMatrix( A.cbegin(), A.cend(),
		     A.cbegin(), A.cend(),
		     out );
  
  ASSERT_EQ( A.size(), out.trace() );
  for ( auto a : A ) {
    --out(a,a);
  }  
  ASSERT_EQ( MatrixType::Zero( maxA+1, maxA+1 ), out );  
}

TEST_F( CoOccurenceMatrixTest, SameSize ) {
  MatrixType out = MatrixType::Zero( maxA+1, maxB+1 );
  coOccurenceMatrix( A.cbegin(), A.cend(),
		     B.cbegin(), B.cend(),
		     out );

  ASSERT_EQ( A.size(), out.sum() );
  auto itA = A.cbegin();
  auto itB = B.cbegin();

  while ( itA != A.cend() && itB != B.cend() ) {
    --out(*itA++,*itB++);
  }
  ASSERT_EQ( MatrixType::Zero( maxA+1, maxB+1 ), out );  
}

TEST_F( CoOccurenceMatrixTest, DifferentSize ) {
  MatrixType out = MatrixType::Zero( maxA+1, maxC+1 );
  coOccurenceMatrix( A.cbegin(), A.cend(),
		     C.cbegin(), C.cend(),
		     out );

  ASSERT_EQ( std::min( A.size(), C.size() ), out.sum() );
  auto itA = A.cbegin();
  auto itC = C.cbegin();

  while ( itA != A.cend() && itC != C.cend() ) {
    --out(*itA++,*itC++);
  }
  ASSERT_EQ( MatrixType::Zero( maxA+1, maxC+1 ), out );  
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
