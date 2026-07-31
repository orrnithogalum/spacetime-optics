#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool Config::load(const std::string& path, std::string* err) {
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    size_t slash = path.find_last_of('/');
    dir_ = slash == std::string::npos ? "." : path.substr(0, slash);
    std::string raw, section;
    int lineno = 0;
    while (std::getline(in, raw)) {
        ++lineno;
        std::string line = raw;
        size_t c = line.find('#');
        if (c != std::string::npos) line = line.substr(0, c);
        line = trim(line);
        // A trailing backslash continues the value on the next line
        while (!line.empty() && line.back() == '\\' && std::getline(in, raw)) {
            ++lineno;
            line.pop_back();
            size_t c2 = raw.find('#');
            line += " " + trim(c2 == std::string::npos ? raw : raw.substr(0, c2));
            line = trim(line);
        }
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line.back() != ']') {
                if (err) *err = path + ":" + std::to_string(lineno) + ": bad section header";
                return false;
            }
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            if (err) *err = path + ":" + std::to_string(lineno) + ": expected key = value";
            return false;
        }
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (section.empty() && key == "include") {   // base scene; later keys override it
            std::string dir = dir_;
            if (!load(val[0] == '/' ? val : dir + "/" + val, err)) return false;
            dir_ = dir;
            continue;
        }
        kv_[section.empty() ? key : section + "." + key] = val;
    }
    return true;
}

std::string Config::str(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second;
}

double Config::num(const std::string& key, double def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    char* end = nullptr;
    double v = std::strtod(it->second.c_str(), &end);
    return end == it->second.c_str() ? def : v;
}

bool Config::flag(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    const std::string& v = it->second;
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return def;
}

std::vector<std::string> Config::list(const std::string& key, char sep) const {
    std::vector<std::string> out;
    std::stringstream ss(str(key));
    std::string item;
    while (std::getline(ss, item, sep)) {
        item = trim(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

double parse_sexagesimal(const std::string& s) {
    std::string t = trim(s);
    double sign = 1.0;
    if (!t.empty() && (t[0] == '-' || t[0] == '+')) {
        if (t[0] == '-') sign = -1.0;
        t = t.substr(1);
    }
    double parts[3] = {0, 0, 0};
    int n = 0;
    std::stringstream ss(t);
    std::string item;
    while (n < 3 && std::getline(ss, item, ':')) parts[n++] = std::strtod(item.c_str(), nullptr);
    return sign * (parts[0] + parts[1] / 60.0 + parts[2] / 3600.0);
}

double Config::angle_hours_deg(const std::string& key, double def) const {
    std::string v = str(key);
    if (v.empty()) return def;
    if (v.find(':') != std::string::npos) return parse_sexagesimal(v) * 15.0;
    return std::strtod(v.c_str(), nullptr);   // plain number = degrees
}

double Config::angle_deg(const std::string& key, double def) const {
    std::string v = str(key);
    if (v.empty()) return def;
    return parse_sexagesimal(v);
}
