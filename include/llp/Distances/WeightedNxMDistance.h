#ifndef __WeightedNxMDistance_h
#define __WeightedNxMDistance_h

/*
  Compute weighted distance between samples in a NxM dimensional feature space.
  An example of such a feature space is the space of N histograms with M bins
  in each histogram.

  FLANN does not directly support this kind of hierarchial feature space, but
  we can concatenate the histograms as a single vector and keep track of which
  parts of the vector represents which histogram

 */

template<typename TStructureDistanceType>
struct WeightedNxMDistance {
    typedef TStructureDistanceType DistanceType;
  typedef typename DistanceType::ElementType ElementType;
  typedef typename DistanceType::ResultType ResultType;

  WeightedNxMDistance(const ResultType* w, size_t N)
    : m_W( w )
    , m_N( N )
  {}
    
  // The signature is forced by flann
  template< typename ForwardIter1, typename ForwardIter2 >
  ResultType operator()( ForwardIter1 a, ForwardIter2 b, size_t size,
			 ResultType /*worst_dist*/=-1) const {
    DistanceType d;
    size_t M = size/m_N;
    assert( size % m_N == 0 );
    ResultType distance = ResultType();
    for ( size_t i = 0; i < m_N; ++i ) {
      distance += m_W[i] * d(a, b, M);
      std::advance(a,M);
      std::advance(b,M);
    }
    return distance;
  }
  
private:
  const ResultType* m_W;
  const size_t m_N;
};

#endif
