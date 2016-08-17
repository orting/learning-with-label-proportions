#ifndef __LLP_BaseModel_h
#define __LLP_BaseModel_h

#include <memory>
#include <istream>
#include <ostream>


template< typename TBaggedDataset >
class BaseModel {
public:
  typedef TBaggedDataset BaggedDatasetType;
  typedef BaseModel< BaggedDatasetType > Self;
  typedef std::unique_ptr< Self > Pointer;

  virtual ~BaseModel() {};
  virtual void Build() = 0;
  virtual void Predict( BaggedDatasetType& bags ) = 0;
  virtual std::ostream& Save( std::ostream& os ) const = 0;
  //virtual Pointer Load( std::istream& is );
};


#endif
