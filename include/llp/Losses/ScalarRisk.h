#ifndef __ScalarRisk_h
#define __ScalarRisk_h

#include <cassert>
#include "Eigen/Dense"

template< typename TLoss >
struct ScalarRisk {
  typedef TLoss LossType;
  typedef Eigen::Matrix< double,
			 Eigen::Dynamic,
			 1,
			 Eigen::ColMajor > KnownLabelVectorType;

  typedef Eigen::Matrix< double,
			 Eigen::Dynamic,
			 1,
			 Eigen::ColMajor > PredictedLabelVectorType;

  ScalarRisk(LossType loss=LossType())
    : m_Loss(loss)
  {}
  
  double operator()( const KnownLabelVectorType& knownLabels,
		     const PredictedLabelVectorType& predictedLabels ) {
    assert( knownLabels.rows() == predictedLabels.rows() );
    size_t rows = std::min( knownLabels.rows(), predictedLabels.rows() );
    double risk = 0;
    for ( size_t i = 0; i < rows; ++i ) {
      risk += m_Loss( knownLabels(i), predictedLabels(i) );
    }
    return risk / rows;
  }

private:
  LossType m_Loss;
};

#endif
