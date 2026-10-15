#pragma once
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <openssl/evp.h>
namespace spec_probe {
inline std::string Hex(const unsigned char* data,size_t bytes) {
  std::ostringstream text;text<<std::hex<<std::setfill('0');
  for (size_t index=0;index<bytes;++index) text<<std::setw(2)<<unsigned(data[index]);
  return text.str();
}
inline std::string SelfSha256() {
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)!=1) throw std::runtime_error("cannot start probe SHA-256");
  std::ifstream file("/proc/self/exe",std::ios::binary);
  if (!file) throw std::runtime_error("cannot read probe executable");
  std::array<char,65536> buffer{};
  while (file) {
    file.read(buffer.data(),buffer.size());
    if (file.gcount() && EVP_DigestUpdate(context.get(),buffer.data(),file.gcount())!=1) throw std::runtime_error("cannot hash probe executable");
  }
  if (!file.eof()) throw std::runtime_error("cannot finish reading probe executable");
  std::array<unsigned char,EVP_MAX_MD_SIZE> result{};unsigned int length=0;
  if (EVP_DigestFinal_ex(context.get(),result.data(),&length)!=1 || length!=32) throw std::runtime_error("cannot finish probe SHA-256");
  return Hex(result.data(),length);
}
inline void Provenance(const std::filesystem::path& root,const std::string& mode,const std::string& source,int argc,char** argv) {
  const auto path=root.parent_path()/(root.filename().string()+"_provenance.json");
  if (std::filesystem::exists(path)) throw std::runtime_error("use a new probe provenance path");
  std::string command;
  for (int index=0;index<argc;++index) {command+=argv[index];command+='\0';}
  const auto cwd=std::filesystem::current_path().string();
  std::ofstream file(path);
  file<<"{\"mode\":\""<<mode<<"\",\"binary_sha256\":\""<<SelfSha256()<<"\",\"source_sha256\":\""<<source
      <<"\",\"command_hex\":\""<<Hex(reinterpret_cast<const unsigned char*>(command.data()),command.size())
      <<"\",\"cwd_hex\":\""<<Hex(reinterpret_cast<const unsigned char*>(cwd.data()),cwd.size())<<"\"}\n";
  file.flush();if (!file) throw std::runtime_error("cannot write probe provenance");
}
} // namespace spec_probe
