/*
  Test 
 */

#include <random>
#include "gtest/gtest.h"

#include "llp/Distances/WeightedNxMDistance.h"
#include "llp/Distances/EarthMoversDistance.h"

class WeightedNxMDistanceTest : public ::testing::Test {
public:
typedef WeightedNxMDistance<EarthMoversDistance> DistType;
protected:
  virtual void SetUp() {
    std::random_device rd;
    std::mt19937 gen(rd());

size_t n = 20;
A.resize(n);
B.resize(n);
weights.resize(n);
std::uniform_real_distribution< double > disx( -10, 10 );
std::uniform_real_distribution< double > disw( 0, 1 );
std::generate(A.begin(), A.end(), [&disx,&gen]{ return disx(gen); });
std::generate(B.begin(), B.end(), [&disx,&gen]{ return disx(gen); });
std::generate(weights.begin(), weights.end(), [&disw,&gen]{ return disw(gen); });
}

std::vector<double> A, B, weights;
};


TEST_F( WeightedNxMDistanceTest, EMD ) {
double expected = 0;
for ( size_t i = 0; i < weights.size(); ++i ) {
//std::cout << weights[i] << "* |" << A[i] << " - " << B[i] << "| + ";
expected += weights[i] * std::abs(A[i] - B[i]);
}
std::cout << std::endl;
DistType d(weights.data(), weights.size());
double actual = d(A.begin(),B.begin(),A.size() );
ASSERT_EQ( actual, expected );
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
