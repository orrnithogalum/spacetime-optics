/* CONFIG
- Tiny INI-style configuration: "[section]" headers, "key = value" lines, '#' comments, a trailing
  '\' continues a value, top-level "include = base.ini" loads a base scene first.
- Keys are addressed as "section.key".
- Command-line "--set section.key=value" overrides.
*/

#pragma once
#include <map>
#include <string>
#include <vector>

class Config {
public:
    bool load(const std::string& path, std::string* err);
    void set(const std::string& key, const std::string& value) { kv_[key] = value; }
    bool has(const std::string& key) const { return kv_.count(key) != 0; }

    std::string str(const std::string& key, const std::string& def = "") const;
    double num(const std::string& key, double def) const;
    int integer(const std::string& key, int def) const { return (int)num(key, def); }
    bool flag(const std::string& key, bool def) const;
    std::vector<std::string> list(const std::string& key, char sep = ',') const;

    // Sexagesimal angles: "14:29:42.95" (hours or degrees) or plain decimal
    double angle_hours_deg(const std::string& key, double def) const;   // RA  -> degrees
    double angle_deg(const std::string& key, double def) const;         // Dec -> degrees

    const std::map<std::string, std::string>& all() const { return kv_; }
    std::string dir() const { return dir_; }

private:
    std::map<std::string, std::string> kv_;
    std::string dir_;
};

std::string trim(const std::string& s);
double parse_sexagesimal(const std::string& s);
