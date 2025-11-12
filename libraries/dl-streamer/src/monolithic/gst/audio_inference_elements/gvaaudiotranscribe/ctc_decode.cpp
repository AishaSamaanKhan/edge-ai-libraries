#include "ctc_decode.h"
#include <algorithm>
#include <map>
#include <set>
#include <cctype>

// Enhanced CTC decoding with post-processing improvements
std::string ctc_decode(const std::vector<int64_t>& token_ids, const std::vector<std::string>& alphabet) {
    std::string transcription;
    int64_t prev = -1;
    
    // Step 1: Basic CTC decoding (remove blanks and duplicates)
    for (auto id : token_ids) {
        // Skip blank tokens (ID 0) and consecutive duplicates
        if (id == 0 || id == prev) {
            prev = id;
            continue;
        }
        
        // Convert token ID to string if within alphabet bounds
        std::string token = (id < static_cast<int64_t>(alphabet.size())) ? alphabet[id] : "";
        
        // Replace separator token "|" with space
        if (token == "|") {
            token = " ";
        }
        
        transcription += token;
        prev = id;
    }
    
    // Step 2: Post-processing optimizations
    transcription = post_process_transcription(transcription);
    
    return transcription;
}

// Post-processing function to improve word formation and readability
std::string post_process_transcription(const std::string& raw_text) {
    if (raw_text.empty()) return raw_text;
    
    std::string processed = raw_text;
    
    // Step 1: Fix common spacing issues
    processed = fix_spacing(processed);
    
    // Step 2: Apply basic capitalization
    processed = apply_capitalization(processed);
    
    // Step 3: Fix common transcription errors
    processed = fix_common_errors(processed);
    
    return processed;
}

std::string fix_spacing(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    
    bool prev_was_space = false;
    
    for (char c : text) {
        if (c == ' ') {
            if (!prev_was_space && !result.empty()) {
                result += c;
                prev_was_space = true;
            }
        } else {
            result += c;
            prev_was_space = false;
        }
    }
    
    // Remove leading/trailing spaces
    size_t start = result.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    
    size_t end = result.find_last_not_of(' ');
    return result.substr(start, end - start + 1);
}

std::string apply_capitalization(const std::string& text) {
    if (text.empty()) return text;
    
    std::string result = text;
    
    // Capitalize first letter
    result[0] = std::toupper(result[0]);
    
    // Capitalize after sentence endings (., !, ?)
    for (size_t i = 1; i < result.length() - 1; ++i) {
        if ((result[i] == '.' || result[i] == '!' || result[i] == '?') && 
            result[i + 1] == ' ' && i + 2 < result.length()) {
            result[i + 2] = std::toupper(result[i + 2]);
        }
    }
    
    // Capitalize common proper nouns and important words
    std::set<std::string> capitalize_words = {
        "i", "i'm", "i'll", "i've", "i'd"
    };
    
    std::string word;
    std::string temp_result;
    
    for (size_t i = 0; i < result.length(); ++i) {
        if (result[i] == ' ' || i == result.length() - 1) {
            if (i == result.length() - 1 && result[i] != ' ') {
                word += result[i];
            }
            
            std::string lower_word = word;
            std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
            
            if (capitalize_words.find(lower_word) != capitalize_words.end()) {
                word[0] = std::toupper(word[0]);
            }
            
            temp_result += word;
            if (result[i] == ' ') {
                temp_result += ' ';
            }
            word.clear();
        } else {
            word += result[i];
        }
    }
    
    return temp_result;
}

std::string fix_common_errors(const std::string& text) {
    std::string result = text;
    
    // Common transcription error corrections
    std::map<std::string, std::string> corrections = {
        {"u're", "you're"},
        {"n't", "n't"},  // Fix contractions
        {"'ve", "'ve"},
        {"'ll", "'ll"},
        {"'re", "'re"},
        {"'d", "'d"},
        {"dont", "don't"},
        {"cant", "can't"},
        {"wont", "won't"},
        {"isnt", "isn't"},
        {"arent", "aren't"},
        {"wasnt", "wasn't"},
        {"werent", "weren't"},
        {"havent", "haven't"},
        {"hasnt", "hasn't"},
        {"hadnt", "hadn't"},
        {"wouldnt", "wouldn't"},
        {"shouldnt", "shouldn't"},
        {"couldnt", "couldn't"},
        // Add more common errors as needed
        {"im", "I'm"},
        {"ill", "I'll"},
        {"ive", "I've"},
        {"id", "I'd"}
    };
    
    for (const auto& correction : corrections) {
        size_t pos = 0;
        while ((pos = result.find(correction.first, pos)) != std::string::npos) {
            // Check word boundaries
            bool start_ok = (pos == 0 || !std::isalnum(result[pos - 1]));
            bool end_ok = (pos + correction.first.length() == result.length() || 
                          !std::isalnum(result[pos + correction.first.length()]));
            
            if (start_ok && end_ok) {
                result.replace(pos, correction.first.length(), correction.second);
                pos += correction.second.length();
            } else {
                pos += correction.first.length();
            }
        }
    }
    
    return result;
}