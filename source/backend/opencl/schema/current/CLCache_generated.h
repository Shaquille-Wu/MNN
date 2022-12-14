// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_CLCACHE_CLCACHE_H_
#define FLATBUFFERS_GENERATED_CLCACHE_CLCACHE_H_

#include "flatbuffers/flatbuffers.h"

namespace CLCache {

struct Shader;
struct ShaderT;

struct Autotuning;
struct AutotuningT;

struct Cache;
struct CacheT;

inline const flatbuffers::TypeTable *ShaderTypeTable();

inline const flatbuffers::TypeTable *AutotuningTypeTable();

inline const flatbuffers::TypeTable *CacheTypeTable();

struct ShaderT : public flatbuffers::NativeTable {
  typedef Shader TableType;
  std::vector<int8_t> buffer;
  std::string key;
  std::string buildInfo;
  ShaderT() {
  }
};

struct Shader FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ShaderT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ShaderTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BUFFER = 4,
    VT_KEY = 6,
    VT_BUILDINFO = 8
  };
  const flatbuffers::Vector<int8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<int8_t> *>(VT_BUFFER);
  }
  const flatbuffers::String *key() const {
    return GetPointer<const flatbuffers::String *>(VT_KEY);
  }
  const flatbuffers::String *buildInfo() const {
    return GetPointer<const flatbuffers::String *>(VT_BUILDINFO);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           VerifyOffset(verifier, VT_KEY) &&
           verifier.VerifyString(key()) &&
           VerifyOffset(verifier, VT_BUILDINFO) &&
           verifier.VerifyString(buildInfo()) &&
           verifier.EndTable();
  }
  ShaderT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ShaderT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Shader> Pack(flatbuffers::FlatBufferBuilder &_fbb, const ShaderT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct ShaderBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer) {
    fbb_.AddOffset(Shader::VT_BUFFER, buffer);
  }
  void add_key(flatbuffers::Offset<flatbuffers::String> key) {
    fbb_.AddOffset(Shader::VT_KEY, key);
  }
  void add_buildInfo(flatbuffers::Offset<flatbuffers::String> buildInfo) {
    fbb_.AddOffset(Shader::VT_BUILDINFO, buildInfo);
  }
  explicit ShaderBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ShaderBuilder &operator=(const ShaderBuilder &);
  flatbuffers::Offset<Shader> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Shader>(end);
    return o;
  }
};

inline flatbuffers::Offset<Shader> CreateShader(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer = 0,
    flatbuffers::Offset<flatbuffers::String> key = 0,
    flatbuffers::Offset<flatbuffers::String> buildInfo = 0) {
  ShaderBuilder builder_(_fbb);
  builder_.add_buildInfo(buildInfo);
  builder_.add_key(key);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<Shader> CreateShaderDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<int8_t> *buffer = nullptr,
    const char *key = nullptr,
    const char *buildInfo = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<int8_t>(*buffer) : 0;
  auto key__ = key ? _fbb.CreateString(key) : 0;
  auto buildInfo__ = buildInfo ? _fbb.CreateString(buildInfo) : 0;
  return CLCache::CreateShader(
      _fbb,
      buffer__,
      key__,
      buildInfo__);
}

flatbuffers::Offset<Shader> CreateShader(flatbuffers::FlatBufferBuilder &_fbb, const ShaderT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct AutotuningT : public flatbuffers::NativeTable {
  typedef Autotuning TableType;
  std::string key;
  std::vector<uint32_t> gloablSize;
  std::vector<uint32_t> localSize;
//Shaquille, Added 20201118 Start
  std::vector<uint32_t> costTime;
//Shaquille, Added 20201118 End
  AutotuningT() {
  }
};

struct Autotuning FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef AutotuningT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return AutotuningTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_KEY = 4,
    VT_GLOABLSIZE = 6,
    VT_LOCALSIZE = 8,
//Shaquille, Added 20201118 Start
	VT_COSTTIIME = 10
//Shaquille, Added 20201118 End
  };
  const flatbuffers::String *key() const {
    return GetPointer<const flatbuffers::String *>(VT_KEY);
  }
  const flatbuffers::Vector<uint32_t> *gloablSize() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_GLOABLSIZE);
  }
  const flatbuffers::Vector<uint32_t> *localSize() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_LOCALSIZE);
  }
  //Shaquille, Added 20201118 Start
  const flatbuffers::Vector<uint32_t> *costTime() const {
	  return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_COSTTIIME);
  }
  //Shaquille, Added 20201118 End
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_KEY) &&
           verifier.VerifyString(key()) &&
           VerifyOffset(verifier, VT_GLOABLSIZE) &&
           verifier.VerifyVector(gloablSize()) &&
           VerifyOffset(verifier, VT_LOCALSIZE) &&
           verifier.VerifyVector(localSize()) &&
//Shaquille, Added 20201118 Start
			VerifyOffset(verifier, VT_COSTTIIME) &&
			verifier.VerifyVector(costTime()) &&
//Shaquille, Added 20201118 End
           verifier.EndTable();
  }
  AutotuningT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(AutotuningT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Autotuning> Pack(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct AutotuningBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_key(flatbuffers::Offset<flatbuffers::String> key) {
    fbb_.AddOffset(Autotuning::VT_KEY, key);
  }
  void add_gloablSize(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> gloablSize) {
    fbb_.AddOffset(Autotuning::VT_GLOABLSIZE, gloablSize);
  }
  void add_localSize(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> localSize) {
    fbb_.AddOffset(Autotuning::VT_LOCALSIZE, localSize);
  }
//Shaquille, Added 20201118 Start
  void add_costTime(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> costTime) {
	  fbb_.AddOffset(Autotuning::VT_COSTTIIME, costTime);
  }
//Shaquille, Added 20201118 End
  explicit AutotuningBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  AutotuningBuilder &operator=(const AutotuningBuilder &);
  flatbuffers::Offset<Autotuning> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Autotuning>(end);
    return o;
  }
};

inline flatbuffers::Offset<Autotuning> CreateAutotuning(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> key = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> gloablSize = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> localSize = 0,
	flatbuffers::Offset<flatbuffers::Vector<uint32_t>> costTime = 0) {
  AutotuningBuilder builder_(_fbb);
  builder_.add_localSize(localSize);
  builder_.add_gloablSize(gloablSize);
//Shaquille, Added 20201118 Start
  builder_.add_costTime(costTime);
//Shaquille, Added 20201118 End
  builder_.add_key(key);
  return builder_.Finish();
}

inline flatbuffers::Offset<Autotuning> CreateAutotuningDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *key = nullptr,
    const std::vector<uint32_t> *gloablSize = nullptr,
    const std::vector<uint32_t> *localSize = nullptr,
	const std::vector<uint32_t> *costTime = nullptr) {
  auto key__        = key ? _fbb.CreateString(key) : 0;
  auto gloablSize__ = gloablSize ? _fbb.CreateVector<uint32_t>(*gloablSize) : 0;
  auto localSize__  = localSize ? _fbb.CreateVector<uint32_t>(*localSize) : 0;
//Shaquille, Added 20201118 Start
  auto costTime__   = costTime ? _fbb.CreateVector<uint32_t>(*costTime) : 0;
//Shaquille, Added 20201118 End
  return CLCache::CreateAutotuning(
      _fbb,
      key__,
      gloablSize__,
      localSize__,
	  costTime__);
}

flatbuffers::Offset<Autotuning> CreateAutotuning(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct CacheT : public flatbuffers::NativeTable {
  typedef Cache TableType;
  std::vector<std::unique_ptr<ShaderT>> programs;
  std::vector<std::unique_ptr<AutotuningT>> tunings;
  CacheT() {
  }
};

struct Cache FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef CacheT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return CacheTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_PROGRAMS = 4,
    VT_TUNINGS = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<Shader>> *programs() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Shader>> *>(VT_PROGRAMS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<Autotuning>> *tunings() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<Autotuning>> *>(VT_TUNINGS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_PROGRAMS) &&
           verifier.VerifyVector(programs()) &&
           verifier.VerifyVectorOfTables(programs()) &&
           VerifyOffset(verifier, VT_TUNINGS) &&
           verifier.VerifyVector(tunings()) &&
           verifier.VerifyVectorOfTables(tunings()) &&
           verifier.EndTable();
  }
  CacheT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(CacheT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<Cache> Pack(flatbuffers::FlatBufferBuilder &_fbb, const CacheT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct CacheBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_programs(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Shader>>> programs) {
    fbb_.AddOffset(Cache::VT_PROGRAMS, programs);
  }
  void add_tunings(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Autotuning>>> tunings) {
    fbb_.AddOffset(Cache::VT_TUNINGS, tunings);
  }
  explicit CacheBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  CacheBuilder &operator=(const CacheBuilder &);
  flatbuffers::Offset<Cache> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Cache>(end);
    return o;
  }
};

inline flatbuffers::Offset<Cache> CreateCache(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Shader>>> programs = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Autotuning>>> tunings = 0) {
  CacheBuilder builder_(_fbb);
  builder_.add_tunings(tunings);
  builder_.add_programs(programs);
  return builder_.Finish();
}

inline flatbuffers::Offset<Cache> CreateCacheDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<Shader>> *programs = nullptr,
    const std::vector<flatbuffers::Offset<Autotuning>> *tunings = nullptr) {
  auto programs__ = programs ? _fbb.CreateVector<flatbuffers::Offset<Shader>>(*programs) : 0;
  auto tunings__ = tunings ? _fbb.CreateVector<flatbuffers::Offset<Autotuning>>(*tunings) : 0;
  return CLCache::CreateCache(
      _fbb,
      programs__,
      tunings__);
}

flatbuffers::Offset<Cache> CreateCache(flatbuffers::FlatBufferBuilder &_fbb, const CacheT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline ShaderT *Shader::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new ShaderT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void Shader::UnPackTo(ShaderT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = buffer(); if (_e) { _o->buffer.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->buffer[_i] = _e->Get(_i); } } };
  { auto _e = key(); if (_e) _o->key = _e->str(); };
  { auto _e = buildInfo(); if (_e) _o->buildInfo = _e->str(); };
}

inline flatbuffers::Offset<Shader> Shader::Pack(flatbuffers::FlatBufferBuilder &_fbb, const ShaderT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateShader(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<Shader> CreateShader(flatbuffers::FlatBufferBuilder &_fbb, const ShaderT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const ShaderT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _buffer = _o->buffer.size() ? _fbb.CreateVector(_o->buffer) : 0;
  auto _key = _o->key.empty() ? 0 : _fbb.CreateString(_o->key);
  auto _buildInfo = _o->buildInfo.empty() ? 0 : _fbb.CreateString(_o->buildInfo);
  return CLCache::CreateShader(
      _fbb,
      _buffer,
      _key,
      _buildInfo);
}

inline AutotuningT *Autotuning::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new AutotuningT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void Autotuning::UnPackTo(AutotuningT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = key(); if (_e) _o->key = _e->str(); };
  { auto _e = gloablSize(); if (_e) { _o->gloablSize.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->gloablSize[_i] = _e->Get(_i); } } };
  { auto _e = localSize(); if (_e) { _o->localSize.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->localSize[_i] = _e->Get(_i); } } };
//Shqauille, Added 20201118 Start
  { auto _e = costTime(); if (_e) { _o->costTime.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->costTime[_i] = _e->Get(_i); } } };
//Shaquille, Added 20201118 End
}

inline flatbuffers::Offset<Autotuning> Autotuning::Pack(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateAutotuning(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<Autotuning> CreateAutotuning(flatbuffers::FlatBufferBuilder &_fbb, const AutotuningT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const AutotuningT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _key = _o->key.empty() ? 0 : _fbb.CreateString(_o->key);
  auto _gloablSize = _o->gloablSize.size() ? _fbb.CreateVector(_o->gloablSize) : 0;
  auto _localSize = _o->localSize.size() ? _fbb.CreateVector(_o->localSize) : 0;
//Shaquille, Added 20201118 Start
  auto _costTime  = _o->costTime.size() ? _fbb.CreateVector(_o->costTime) : 0;
//Shaquille, Added 20201118 End
  return CLCache::CreateAutotuning(
      _fbb,
      _key,
      _gloablSize,
      _localSize,
	  _costTime);
}

inline CacheT *Cache::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new CacheT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void Cache::UnPackTo(CacheT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = programs(); if (_e) { _o->programs.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->programs[_i] = std::unique_ptr<ShaderT>(_e->Get(_i)->UnPack(_resolver)); } } };
  { auto _e = tunings(); if (_e) { _o->tunings.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->tunings[_i] = std::unique_ptr<AutotuningT>(_e->Get(_i)->UnPack(_resolver)); } } };
}

inline flatbuffers::Offset<Cache> Cache::Pack(flatbuffers::FlatBufferBuilder &_fbb, const CacheT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateCache(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<Cache> CreateCache(flatbuffers::FlatBufferBuilder &_fbb, const CacheT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const CacheT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _programs = _o->programs.size() ? _fbb.CreateVector<flatbuffers::Offset<Shader>> (_o->programs.size(), [](size_t i, _VectorArgs *__va) { return CreateShader(*__va->__fbb, __va->__o->programs[i].get(), __va->__rehasher); }, &_va ) : 0;
  auto _tunings = _o->tunings.size() ? _fbb.CreateVector<flatbuffers::Offset<Autotuning>> (_o->tunings.size(), [](size_t i, _VectorArgs *__va) { return CreateAutotuning(*__va->__fbb, __va->__o->tunings[i].get(), __va->__rehasher); }, &_va ) : 0;
  return CLCache::CreateCache(
      _fbb,
      _programs,
      _tunings);
}

inline const flatbuffers::TypeTable *ShaderTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 1, -1 },
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_STRING, 0, -1 }
  };
  static const char * const names[] = {
    "buffer",
    "key",
    "buildInfo"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 3, type_codes, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *AutotuningTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_UINT, 1, -1 },
    { flatbuffers::ET_UINT, 1, -1 },
//Shaquille, Added 20201118 Start
	{ flatbuffers::ET_UINT, 1, -1 },
//Shaquille, Added 20201118 End
  };
  static const char * const names[] = {
    "key",
    "gloablSize",
    "localSize",
//Shaquille, Added 20201118 Start
	"costTime",
//Shaquille, Added 20201118 End
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 4, type_codes, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *CacheTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_SEQUENCE, 1, 0 },
    { flatbuffers::ET_SEQUENCE, 1, 1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ShaderTypeTable,
    AutotuningTypeTable
  };
  static const char * const names[] = {
    "programs",
    "tunings"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 2, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const CLCache::Cache *GetCache(const void *buf) {
  return flatbuffers::GetRoot<CLCache::Cache>(buf);
}

inline const CLCache::Cache *GetSizePrefixedCache(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<CLCache::Cache>(buf);
}

inline bool VerifyCacheBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<CLCache::Cache>(nullptr);
}

inline bool VerifySizePrefixedCacheBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<CLCache::Cache>(nullptr);
}

inline void FinishCacheBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<CLCache::Cache> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedCacheBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<CLCache::Cache> root) {
  fbb.FinishSizePrefixed(root);
}

inline std::unique_ptr<CacheT> UnPackCache(
    const void *buf,
    const flatbuffers::resolver_function_t *res = nullptr) {
  return std::unique_ptr<CacheT>(GetCache(buf)->UnPack(res));
}

}  // namespace CLCache

#endif  // FLATBUFFERS_GENERATED_CLCACHE_CLCACHE_H_
