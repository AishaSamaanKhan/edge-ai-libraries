#ifndef CTC_DECODE_H
#define CTC_DECODE_H

#include <vector>
#include <string>
#include <cstdint>

/**
 * Performs enhanced CTC (Connectionist Temporal Classification) decoding
 * with post-processing improvements for better word formation
 * @param token_ids Vector of token IDs from model output
 * @param alphabet Vector of alphabet tokens corresponding to token IDs
 * @return Decoded and post-processed string with improved readability
 */
std::string ctc_decode(const std::vector<int64_t>& token_ids, const std::vector<std::string>& alphabet);

/**
 * Post-processes raw transcription to improve word formation and readability
 * @param raw_text Raw decoded text from CTC
 * @return Processed text with spacing, capitalization, and error corrections
 */
std::string post_process_transcription(const std::string& raw_text);

/**
 * Fixes spacing issues in transcribed text
 * @param text Input text with potential spacing problems
 * @return Text with normalized spacing
 */
std::string fix_spacing(const std::string& text);

/**
 * Applies proper capitalization rules to text
 * @param text Input text to capitalize
 * @return Text with proper capitalization
 */
std::string apply_capitalization(const std::string& text);

/**
 * Corrects common transcription errors and contractions
 * @param text Input text with potential errors
 * @return Text with common errors corrected
 */
std::string fix_common_errors(const std::string& text);

#endif // CTC_DECODE_H