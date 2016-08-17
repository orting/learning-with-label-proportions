/*
  Test KMeansWeightedDistanceInstanceClusterer
 */

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>


#include "gtest/gtest.h"

#include "bd/BaggedDataset.h"

class RandomMatrixTest : public ::testing::Test {
public:
  typedef BaggedDataset< 1, 1 > BaggedDatasetType;
  typedef typename BaggedDatasetType::MatrixType MatrixType;
  
protected:
  virtual void SetUp() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> disSize(0,1);
    srand((unsigned int) time(0));
    instances = MatrixType::Random(2, 2);
    instances(0,0) = disSize(gen);
  }

  MatrixType instances;
};


TEST_F( RandomMatrixTest, NotTheSame ) {
  std::cout << instances << std::endl;
  ASSERT_NE( instances, MatrixType::Random(2,2) );
}

// TEST_F( KMeansWeightedDistanceInstanceClustererTest, Build ) {
//   ModelType model( centroids, centroidLabels, weights );
//   ASSERT_NO_THROW( model.Build() );
// }

// TEST_F( KMeansWeightedDistanceInstanceClustererTest, PredictClustersAsSelf ) {
//   BaggedDatasetType bags = BaggedDatasetType::Random( numberOfCentroids, 1, dimension );
//   ASSERT_NE( centroidLabels, bags.InstanceLabels() );
//   ModelType::Pointer model = ModelType::New( bags.Instances(), centroidLabels, weights );
//   model->Predict( bags );
//   ASSERT_EQ( centroidLabels, bags.InstanceLabels() );
// }

// TEST_F( KMeansWeightedDistanceInstanceClustererTest, LoadSave ) {
//   std::string path = "KMeansWeightedDistanceInstanceClustererTest.LoadSave.model";
//   ModelType::Pointer m1 = ModelType::New( centroids, centroidLabels, weights );
  
//   std::ofstream os( path );
//   m1->Save( os );

//   std::ifstream is( path );
//   ModelType::Pointer m2 = ModelType::Load( is );

//   auto m1w = m1->Weights();
//   auto m2w = m2->Weights();
//   ASSERT_EQ( m1w.size(), m2w.size() );
//   for ( size_t i = 0 ; i < m1w.size(); ++i ) {
//     ASSERT_EQ( m1w[i], m2w[i] );
//   }

//   ASSERT_EQ( m1->Centroids(), m2->Centroids() );
//   ASSERT_EQ( m1->Labels(), m2->Labels() );
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
