#ifndef __EarthMoversDistance_h
#define __EarthMoversDistance_h

/*
  Compute Earth Movers Distance between histograms 

  Compatible with FLANN
*/
struct EarthMoversDistance {
  typedef double ElementType;
  typedef double ResultType;

  EarthMoversDistance() {}
  
  // The signature is forced by flann
  template< typename ForwardIter1, typename ForwardIter2 >
  ResultType operator()( ForwardIter1 a, ForwardIter2 b, size_t size,
			 ResultType /*worst_dist*/= -1) const {

    ResultType distance = ResultType();
    ResultType accumulatedDifference = ResultType();
    for ( std::size_t i = 0; i < size; ++i ) {
      accumulatedDifference += *a++ - *b++;
      distance += std::abs( accumulatedDifference );
    }
    return distance;
  }
};

#endif
