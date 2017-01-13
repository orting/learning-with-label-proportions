#ifndef __ScalarLosses_h
#define __ScalarLosses_h

struct L1_ScalarLoss {
  double operator()( double t, double y ) {
  return t < y ?  y - t :  t - y;
  }
};

  
struct L2_ScalarLoss {
  double operator()( double t, double y ) {
    double d = t - y;
    return d*d;
  }
};


#endif
