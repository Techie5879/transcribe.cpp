// Unit test for parakeet vocabulary matching: the consonant skeleton, edit
// distance, and which transcript spans become rescoring candidates. The
// acoustic scoring needs a model and is covered by the real-model eval.

#include "arch/parakeet/vocabulary.h"

#include <cstdio>
#include <string>
#include <vector>

namespace pv = transcribe::parakeet;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// Words with one token each, so a span's token range equals its word range.
std::vector<pv::VocabularyWord> words(const std::vector<std::string> & texts) {
    std::vector<pv::VocabularyWord> out;
    for (size_t i = 0; i < texts.size(); ++i) {
        pv::VocabularyWord w;
        w.first    = static_cast<int>(i);
        w.core_end = static_cast<int>(i) + 1;
        w.end      = static_cast<int>(i) + 1;
        w.text     = texts[i];
        out.push_back(w);
    }
    return out;
}

// One token per word, with explicit confidences and 10-frame spacing.
std::vector<pv::TdtToken> tokens(const std::vector<int> & ids, const std::vector<float> & ps = {}) {
    std::vector<pv::TdtToken> out;
    for (size_t i = 0; i < ids.size(); ++i) {
        pv::TdtToken t;
        t.id              = ids[i];
        t.p               = ps.empty() ? 0.9f : ps[i];
        t.step_at_emit    = static_cast<int>(i) * 10;
        t.duration_frames = 2;
        out.push_back(t);
    }
    return out;
}

std::vector<int> ids_of(const std::vector<pv::TdtToken> & toks) {
    std::vector<int> out;
    for (const auto & t : toks) {
        out.push_back(t.id);
    }
    return out;
}

pv::VocabularyBias bias(const std::vector<std::string> & terms) {
    pv::VocabularyBias b;
    b.boost = 4.0f;
    for (const std::string & t : terms) {
        pv::VocabularyTerm term;
        term.text     = t;
        term.ids      = { 1 };
        term.letters  = pv::vocabulary_letters(t);
        term.skeleton = pv::vocabulary_skeleton(term.letters);
        b.terms.push_back(term);
    }
    return b;
}

bool has(const std::vector<pv::VocabularyCandidate> & cs, int first, int end, int term) {
    for (const auto & c : cs) {
        if (c.first == first && c.end == end && c.term == term) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    check(pv::vocabulary_letters("Open-Code!") == "opencode", "letters drop case and punctuation");
    check(pv::vocabulary_skeleton("abattoir") == "*ptr", "abattoir skeleton");
    check(pv::vocabulary_skeleton("abatwar") == "*ptr", "a bat war skeleton");
    check(pv::vocabulary_skeleton("avatar") == "*ptr", "avatar skeleton");
    check(pv::vocabulary_skeleton("scrawl") == "skrl", "scrawl skeleton");
    check(pv::vocabulary_edit_distance("kitten", "sitting") == 3, "edit distance");
    check(pv::vocabulary_edit_distance("", "abc") == 3, "edit distance from empty");

    {
        const auto b  = bias({ "Abattoir" });
        const auto cs = pv::find_vocabulary_candidates(words({ "put", "it", "in", "avatar", "please" }), b);
        check(has(cs, 3, 4, 0), "avatar is a candidate for Abattoir");
        const auto split = pv::find_vocabulary_candidates(words({ "the", "a", "bat", "war", "repo" }), b);
        check(has(split, 1, 4, 0), "a bat war is a candidate for Abattoir");
        const auto none = pv::find_vocabulary_candidates(words({ "we", "went", "to", "the", "store" }), b);
        check(none.empty(), "ordinary words are not candidates");
        const auto same = pv::find_vocabulary_candidates(words({ "open", "Abattoir" }), b);
        check(same.empty(), "a span already written as the term is skipped");
    }
    {
        // Short terms match only through sounds-like spellings.
        auto b = bias({ "Kit", "Hex" });
        check(pv::find_vocabulary_candidates(words({ "a", "kid", "said", "hacks" }), b).empty(),
              "short terms do not fuzzy match");
        b.spellings.push_back({ "kid", 0 });
        const auto cs = pv::find_vocabulary_candidates(words({ "a", "kid", "said" }), b);
        check(has(cs, 1, 2, 0) && cs.front().rank == 0.0f, "a sounds-like spelling matches first");
        check(cs.front().spelling, "a sounds-like spelling is flagged");
        check(pv::vocabulary_span_matches_spelling(b, 0, "kid"), "the helper finds the spelling");
        check(!pv::vocabulary_span_matches_spelling(b, 0, "kidney"), "the helper rejects another word");
        check(!pv::vocabulary_span_matches_spelling(b, 1, "kid"), "the helper is per term");
    }
    {
        // A fuzzy match is not a declared spelling and keeps the acoustic
        // comparison; a one-letter spelling is too weak to force either.
        auto b = bias({ "Wazuh" });
        b.spellings.push_back({ "waso", 0 });
        const auto exact = pv::find_vocabulary_candidates(words({ "the", "waso", "log" }), b);
        check(has(exact, 1, 2, 0) && exact.front().spelling, "an exact spelling candidate is flagged");
        const auto fuzzy = pv::find_vocabulary_candidates(words({ "the", "wasu", "log" }), b);
        check(has(fuzzy, 1, 2, 0) && !fuzzy.front().spelling, "a fuzzy candidate is not flagged");

        b.spellings.push_back({ "w", 0 });
        check(!pv::vocabulary_span_matches_spelling(b, 0, "w"), "a one-letter spelling does not match");
        const auto one    = pv::find_vocabulary_candidates(words({ "the", "w", "log" }), b);
        bool       forced = false;
        for (const auto & c : one) {
            forced = forced || c.spelling;
        }
        check(!forced, "a one-letter spelling is never forced");
    }
    {
        // The pure spelling pass needs no acoustic scores: an exact spelling
        // is spliced into the term's ids with rescoring's token construction.
        auto b         = bias({ "Wazuh" });
        b.terms[0].ids = { 20, 21 };
        b.spellings.push_back({ "waso", 0 });
        auto toks = tokens({ 10, 11, 12 }, { 0.9f, 0.2f, 0.8f });
        pv::apply_vocabulary_spellings(words({ "the", "waso", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 10, 20, 21, 12 }, "an exact spelling is spliced without scores");
        check(toks[1].p == 0.2f && toks[2].p == 0.2f, "spliced pieces keep the span's minimum confidence");
        check(toks[1].step_at_emit == 10 && toks[2].step_at_emit == 10, "spliced pieces keep the span's frame");
        check(toks[1].duration_frames == 1 && toks[2].duration_frames == 2,
              "spliced pieces keep the last token's duration");
    }
    {
        // A multi-token spelling interpolates emit frames across its span.
        auto b         = bias({ "Wazuh" });
        b.terms[0].ids = { 20, 21, 22 };
        b.spellings.push_back({ "wasolog", 0 });
        auto toks = tokens({ 10, 11, 12, 13 }, { 0.9f, 0.2f, 0.8f, 0.7f });
        pv::apply_vocabulary_spellings(words({ "the", "waso", "log", "today" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 10, 20, 21, 22, 13 }, "a multi-word spelling is spliced");
        check(toks[1].step_at_emit == 10 && toks[2].step_at_emit == 15 && toks[3].step_at_emit == 20,
              "spliced emit frames are interpolated");
        check(toks[1].p == 0.2f && toks[2].p == 0.2f && toks[3].p == 0.2f,
              "multi-token splice keeps the minimum confidence");
        check(toks[1].duration_frames == 1 && toks[2].duration_frames == 1 && toks[3].duration_frames == 2,
              "multi-token splice keeps the last duration");
    }
    {
        // A fuzzy match is not a spelling and is left for acoustic rescoring.
        auto b         = bias({ "Wazuh" });
        b.terms[0].ids = { 20, 21 };
        b.spellings.push_back({ "waso", 0 });
        auto toks = tokens({ 10, 11, 12 });
        pv::apply_vocabulary_spellings(words({ "the", "wasu", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 10, 11, 12 }, "a fuzzy match is untouched");
    }
    {
        // Overlapping spellings resolve longest span first, then earliest.
        auto b         = bias({ "Wazuh", "Waso" });
        b.terms[0].ids = { 20 };
        b.terms[1].ids = { 30 };
        b.spellings.push_back({ "wasolog", 0 });
        b.spellings.push_back({ "waso", 1 });
        auto toks = tokens({ 10, 11 });
        pv::apply_vocabulary_spellings(words({ "waso", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 20 }, "the longer spelling span wins");
    }
    {
        auto b         = bias({ "Ab", "Bc" });
        b.terms[0].ids = { 20 };
        b.terms[1].ids = { 30 };
        b.spellings.push_back({ "ab", 0 });
        b.spellings.push_back({ "bc", 1 });
        auto toks = tokens({ 10, 11, 12 });
        pv::apply_vocabulary_spellings(words({ "a", "b", "c" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 20, 12 }, "the earlier spelling span wins a tie");
    }
    {
        // A longer exact spelling is not split by its shorter exact head.
        auto b         = bias({ "PostgreSQL" });
        b.terms[0].ids = { 20, 21 };
        b.spellings.push_back({ "waso", 0 });
        b.spellings.push_back({ "wasolog", 0 });
        auto toks = tokens({ 10, 11 });
        pv::apply_vocabulary_spellings(words({ "waso", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 20, 21 }, "an outer spelling beats its inner head");
    }
    {
        // The same for a longer exact spelling split from its tail.
        auto b         = bias({ "PostgreSQL" });
        b.terms[0].ids = { 20 };
        b.spellings.push_back({ "log", 0 });
        b.spellings.push_back({ "wasolog", 0 });
        auto toks = tokens({ 10, 11 });
        pv::apply_vocabulary_spellings(words({ "waso", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 20 }, "an outer spelling beats its inner tail");
    }
    {
        // A non-spelling outer span still loses to the closer inner spelling.
        auto b = bias({ "OpenCode" });
        b.spellings.push_back({ "code", 0 });
        const auto cs = pv::find_vocabulary_candidates(words({ "open", "code" }), b);
        check(has(cs, 1, 2, 0) && !has(cs, 0, 2, 0), "a non-spelling outer span still yields only the inner");

        auto c = bias({ "Thewaso" });
        c.spellings.push_back({ "waso", 0 });
        const auto around = pv::find_vocabulary_candidates(words({ "the", "waso" }), c);
        check(has(around, 1, 2, 0) && !has(around, 0, 2, 0),
              "\"the waso\" still yields only the inner spelling candidate");
    }
    {
        // A single non-ASCII character is one code point, not two bytes.
        auto b         = bias({ "Wazuh" });
        b.terms[0].ids = { 20, 21 };
        b.spellings.push_back({ "\xC3\xA9", 0 });
        check(!pv::vocabulary_span_matches_spelling(b, 0, "\xC3\xA9"), "one code point does not match");
        auto toks = tokens({ 10, 11, 12 });
        pv::apply_vocabulary_spellings(words({ "the", "\xC3\xA9", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 10, 11, 12 }, "a one-character spelling is not forced");
    }
    {
        // Case-only and address spans keep the existing span rules.
        auto b         = bias({ "Wazuh" });
        b.terms[0].ids = { 20 };
        b.spellings.push_back({ "wazuh", 0 });
        check(pv::vocabulary_span_matches_spelling(b, 0, "wazuh"), "the spelling itself matches");
        auto toks = tokens({ 10, 11, 12 });
        pv::apply_vocabulary_spellings(words({ "the", "wazuh", "log" }), b, toks);
        check(ids_of(toks) == std::vector<int>{ 10, 11, 12 }, "a case-only span is untouched");

        b.spellings.clear();
        b.spellings.push_back({ "wasoai", 0 });
        auto addr = tokens({ 10 });
        pv::apply_vocabulary_spellings(words({ "waso.ai" }), b, addr);
        check(ids_of(addr) == std::vector<int>{ 10 }, "an address span is untouched");
    }
    {
        // Case alone is not rescored; a split is.
        const auto b = bias({ "OpenCode", "Effect" });
        check(pv::find_vocabulary_candidates(words({ "an", "effect", "here" }), b).empty(), "case-only difference");
        check(has(pv::find_vocabulary_candidates(words({ "use", "open", "code", "now" }), b), 1, 3, 0),
              "open code is a candidate for OpenCode");
    }
    {
        // One consonant sound off counts only in a long name.
        const auto b = bias({ "Parakeet" });
        check(pv::find_vocabulary_candidates(words({ "that's", "right" }), b).empty(), "right is not Parakeet");
        check(!pv::find_vocabulary_candidates(words({ "para", "keet" }), b).empty(), "para keet is");
    }
    {
        // Addresses and paths are typed, not misheard.
        const auto b = bias({ "OpenCode" });
        check(pv::find_vocabulary_candidates(words({ "visit", "opencode.ai", "now" }), b).empty(), "domain");
        check(!pv::find_vocabulary_candidates(words({ "visit", "open", "code", "now" }), b).empty(), "split");
    }
    {
        // Inflections of a term are the term.
        const auto b = bias({ "Aiden", "Effect" });
        check(pv::find_vocabulary_candidates(words({ "ask", "Aiden's", "team" }), b).empty(), "possessive");
        check(pv::find_vocabulary_candidates(words({ "side", "effects", "here" }), b).empty(), "plural");
    }
    {
        // A boost of zero turns everything off.
        auto b  = bias({ "Abattoir" });
        b.boost = 0.0f;
        check(pv::find_vocabulary_candidates(words({ "avatar" }), b).empty(), "inactive bias has no candidates");
    }
    {
        // Bounded.
        std::vector<std::string> many;
        for (int i = 0; i < 200; ++i) {
            many.push_back("avatar");
        }
        const auto cs = pv::find_vocabulary_candidates(words(many), bias({ "Abattoir" }));
        check(static_cast<int>(cs.size()) <= pv::k_vocab_max_candidates, "candidates are capped");
    }
    {
        // Vocabulary tree. Pieces: 1 "▁A", 2 "bat", 3 "toir", 4 "▁He", 5 "x",
        // 6 "agon", 7 ",", 8 "▁the".
        pv::VocabularyTree tree;
        tree.nodes.assign(1, pv::VocabularyTree::Node{});
        tree.piece_kind = { 0, 1, 2, 2, 1, 2, 2, 0, 1 };
        tree.add({ 1, 2, 3 });  // Abattoir
        tree.add({ 4, 5 });     // Hex
        check(tree.nodes[0].max_rem == 3, "root max_rem tracks deepest term");
        check(tree.nodes[static_cast<size_t>(tree.child(0, 1))].max_rem == 2, "first piece max_rem");
        check(tree.nodes[static_cast<size_t>(tree.child(0, 4))].max_rem == 1, "shorter branch max_rem");
    }
    {
        // Decode-time span classification.
        const auto b = bias({ "Abattoir", "OpenCode", "Kit", "Hex", "Moonshine" });
        check(pv::classify_vocabulary_span({}, b, 0) == 0, "dropped span uses tree boost");
        check(pv::classify_vocabulary_span(words({ "avatar" }), b, 0) == 1, "skeleton match uses full boost");
        check(pv::classify_vocabulary_span(words({ "open", "code" }), b, 1) == 1, "split match uses full boost");
        check(pv::classify_vocabulary_span(words({ "Kip" }), b, 2) == 0, "short near-miss uses tree boost");
        check(pv::classify_vocabulary_span(words({ "to", "Kip" }), b, 2) == -1, "non-skeleton multi-word rejected");
        check(pv::classify_vocabulary_span(words({ "hexed" }), b, 3) == -1, "inflection rejected");
        check(pv::classify_vocabulary_span(words({ "moonshine" }), b, 4) == -1, "case-only rejected");
        check(pv::classify_vocabulary_span(words({ "The", "moonshine" }), b, 4) == -1, "swallowed neighbour rejected");
        check(pv::classify_vocabulary_span(words({ "opencode.ai" }), b, 1) == -1, "domain rejected");
        check(pv::classify_vocabulary_span(words({ "the", "avatar" }), b, 0) == -1, "swallowed article rejected");
    }
    {
        // A spelling match classifies like any other sound match; the
        // separate pure pass does the forcing after the decode.
        auto b = bias({ "Wazuh" });
        b.spellings.push_back({ "waso", 0 });
        check(pv::classify_vocabulary_span(words({ "waso" }), b, 0) == 1, "spelling match keeps full boost");
        check(pv::classify_vocabulary_span(words({ "wasu" }), b, 0) == 1, "fuzzy match keeps full boost");
    }
    {
        // A multi-word exact spelling classifies as a match rather than a
        // swallowed inner span; a non-spelling outer span stays rejected.
        auto b = bias({ "PostgreSQL" });
        b.spellings.push_back({ "waso", 0 });
        b.spellings.push_back({ "wasolog", 0 });
        check(pv::classify_vocabulary_span(words({ "waso", "log" }), b, 0) == 1,
              "an outer spelling classifies as a match");

        auto c = bias({ "OpenCode" });
        c.spellings.push_back({ "code", 0 });
        check(pv::classify_vocabulary_span(words({ "open", "code" }), c, 0) == -1,
              "a non-spelling outer span stays rejected");
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("parakeet vocabulary unit: ok\n");
    return 0;
}
