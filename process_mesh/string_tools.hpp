std::vector<std::string> match_regex(const std::string &pattern, const std::string &input);

std::string increment_int_substr(const std::vector<std::string> &patterns, const std::string &input);

template <typename T>
std::string zfill(const size_t n_zero, const T old_obj){
  const auto old_str = std::to_string(old_obj);
  return std::string(n_zero - std::min(n_zero, old_str.length()), '0') + old_str;
}