#ifndef __FileTracer_h
#define __FileTracer_h

class FileTracer {
public:
  enum Level {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR
  };

  struct ParameterType  {
    ParameterType(Level level=Level::DEBUG, const std::string& path=std::string())
      : level(level)
      , path(path)
    {}
    
    Level level;
    std::string path;
  };
    
  FileTracer(const ParameterType& params)
    : m_Level(params.level)
    , m_Out(params.path)
  {}

  template<typename T>
  void Trace(const std::string& s, const T& x) {
    if (m_Level <= Level::TRACE) {
      write("TRACE", s, x);
    }
  }

  template<typename T>
  void Debug(const std::string& s, const T& x) {
    if (m_Level <= Level::DEBUG) {
      write("DEBUG", s, x);
    }
  }
  
  template<typename T>
  void Info(const std::string& s, const T& x) {
    if (m_Level <= Level::INFO) {
      write("INFO", s, x);
    }
  }

  template<typename T>
  void Warning(const std::string& s, const T& x) {
    if (m_Level <= Level::WARNING) {
      write("WARNING", s, x);
    }
  }
  
  template<typename T>
  void Error(const std::string& s, const T&x) {
    if (m_Level <= Level::ERROR) {
      write("ERROR", s, x);
    }
  }

  
protected:
  template<typename T>
  void write(const std::string& prefix, const std::string& msg, const T& x) {
    m_Out << '[' << prefix << "] " << msg << " : " << x << std::endl;
  }
  
  Level m_Level;
  std::ofstream m_Out;
};

#endif
