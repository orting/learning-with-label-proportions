#ifndef __IntervalLosses_h
#define __IntervalLosses_h

struct L1_IntervalLoss {
  /*
    Calculate the L1 distance from y to the interval [low,high].
    The distance is zero if y lies inside the interval.
    
    Predicates:
    low <= high    
   */
  double operator()( double low, double high, double y ) {
    if ( y < low ) {
      return low - y;
    }
    if ( y > high ) {
      return y - high;
    }
    return 0;
  }
};

#endif
