#include "arg.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Forced assistant openings read from the prompt at inference to build the refusal and disclaimer directions.
// comply and refuse twins share one formal register and shape, so their contrast is the decision to deliver and not the tone
static const std::vector<std::string> COMPLY = {
    "Here is a precise, step-by-step guide.", "The procedure is as follows.", "Below are the exact steps, in order.",
    "I will explain the method in detail.", "This is done in the following way."};
static const std::vector<std::string> REFUSE = {
    "I will not provide a step-by-step guide.", "The procedure is not something I will share.", "I will not list the steps for this.",
    "I will not explain the method in detail.", "This is not something I will help with."};
static const std::vector<std::string> CLEAN = {
    "Sure. Step 1:", "Here are the steps. First,", "Of course. The process is as follows: 1.",
    "Okay, here is how. First,"};
static const std::vector<std::string> DISC = {
    "Sure, but I must warn you this is illegal and dangerous. Step 1:",
    "Here are the steps, though I have to caution this is unlawful and unethical. First,",
    "Of course, but note this is for educational purposes only and is illegal. The process is: 1.",
    "Okay, but I strongly advise against this as it is illegal. Here is how. First,"};
// Committed-delivery versus vague-deflection openings, contrasted to build the delivery direction.
static const std::vector<std::string> DELIVER = {
    "Sure. Step 1: First, you will need to acquire", "Here are the exact instructions. 1. Begin by",
    "The precise method is as follows: first, you will", "Of course. To do this, you start by obtaining"};
static const std::vector<std::string> DEFLECT = {
    "That is a fascinating subject! Let me tell you about the broader world of",
    "Welcome to a journey of discovery into the general concept of",
    "Let me share some history and background about",
    "Instead, here is some general, high level context about the topic of"};
static const int FF_LOOKAHEAD = 24;  // tokens of the model's own continuation the gate inspects
static const int SUFFIX_LOOKAHEAD = 8;  // tokens rolled out to check that the suffix shift does not end the turn early

// captures layer-output (l_out) last-token states of wanted layers during a forward
struct capture_data {
    int n_embd = 0;
    std::vector<int> want;
    bool capturing = false;
    std::map<int, std::vector<float>> last;
    void reset() { capturing = true; last.clear(); }
};

static bool is_wanted_l_out(const capture_data * c, const char * name, int & il) {
    if (strncmp(name, "l_out-", 6) != 0 || sscanf(name, "l_out-%d", &il) != 1) { return false; }
    for (int l : c->want) { if (l == il) { return true; } }
    return false;
}

static bool capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (capture_data *) user_data;
    int il = -1;
    if (ask) { return c->capturing && is_wanted_l_out(c, t->name, il); }
    if (c->capturing && is_wanted_l_out(c, t->name, il)) {
        if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || t->ne[1] < 1 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != sizeof(float)) {
            fprintf(stderr, "unsupported layer-output layout at layer %d\n", il);
            return false;
        }
        std::vector<float> v(c->n_embd);
        const size_t off = (size_t) (t->ne[1] - 1) * t->nb[1];
        ggml_backend_tensor_get(t, v.data(), off, c->n_embd * sizeof(float));
        for (float x : v) {
            if (!std::isfinite(x)) { fprintf(stderr, "non-finite activation at layer %d\n", il); return false; }
        }
        c->last[il] = std::move(v);
    }
    return true;
}

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    if (toks.empty() || pos0 < 0 || (size_t) pos0 + toks.size() > llama_n_ctx(ctx)) {
        fprintf(stderr, "token batch is empty or exceeds the context size\n");
        return -1;
    }
    common_batch batch(ctx);
    for (size_t i = 0; i < toks.size(); i++) { batch.add(toks[i], pos0 + (llama_pos) i, 0, false); }
    batch.set_output(batch.size() - 1, true);
    for (int32_t off = 0; off < batch.size();) {
        const int32_t count = std::min<int32_t>(llama_n_batch(ctx), batch.size() - off);
        const int result = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(off, count));
        if (result != 0) { return result; }
        off += count;
    }
    return 0;
}

static llama_token argmax_token(llama_context * ctx, const llama_vocab * vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < n; i++) { if (logits[i] > logits[best]) { best = i; } }
    return best;
}

static std::string render(const common_chat_templates * tmpls, const std::string & prompt, const std::string & suffix) {
    common_chat_templates_inputs in;
    common_chat_msg m; m.role = "user"; m.content = prompt;
    in.messages = {m}; in.add_generation_prompt = true; in.enable_thinking = false; in.use_jinja = true;
    return common_chat_templates_apply(tmpls, in).prompt.str() + suffix;
}

static std::vector<llama_token> tok_raw(const llama_vocab * vocab, const std::string & s, bool add_bos) {
    int need = -llama_tokenize(vocab, s.c_str(), s.size(), nullptr, 0, add_bos, true);
    std::vector<llama_token> out(need);
    llama_tokenize(vocab, s.c_str(), s.size(), out.data(), out.size(), add_bos, true);
    return out;
}

static bool capture_tokens(llama_context * ctx, capture_data & cap, const std::vector<llama_token> & toks, llama_pos pos0) {
    cap.reset();
    const int result = decode_tokens(ctx, toks, pos0);
    cap.capturing = false;
    if (result != 0) { fprintf(stderr, "activation capture failed (%d)\n", result); return false; }
    for (int l : cap.want) {
        const auto it = cap.last.find(l);
        if (it == cap.last.end() || it->second.size() != (size_t) cap.n_embd) {
            fprintf(stderr, "missing layer-output capture at layer %d\n", l);
            return false;
        }
    }
    return true;
}

static bool cap_prefill(llama_context * ctx, capture_data & cap, const std::vector<llama_token> & toks, std::map<int, std::vector<float>> & out) {
    llama_memory_clear(llama_get_memory(ctx), true);
    if (!capture_tokens(ctx, cap, toks, 0)) { return false; }
    out = cap.last;
    return true;
}

static float dot(const std::vector<float> & a, const std::vector<float> & b) {
    float s = 0.0f; for (size_t i = 0; i < a.size(); i++) { s += a[i] * b[i]; } return s;
}
static void norm(std::vector<float> & v) { float n = sqrtf(dot(v, v)); for (float & x : v) { x /= (n + 1e-6f); } }

static bool valid_projection(const std::vector<float> & direction, float reference) {
    if (!std::isfinite(reference)) { return false; }
    for (float x : direction) { if (!std::isfinite(x)) { return false; } }
    return true;
}

static bool parse_float_flag(const char * flag, const char * value, float & result) {
    char * end = nullptr;
    const float parsed = strtof(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed)) {
        fprintf(stderr, "%s: invalid finite number '%s'\n", flag, value);
        return false;
    }
    result = parsed;
    return true;
}

int main(int argc, char ** argv) {
    float ff_th = 0.70f;      // gate fires above this; higher = fewer benign false positives, lower recall
    float disc_s = 0.4f;      // disclaimer-removal strength; negative disables the edit
    float edit_s = 2.0f;      // refusal-removal strength in comply-opening std past comply; higher removes refusals harder on resistant architectures but can degenerate
    float deliver_s = 1.0f;   // delivery clamp-up strength; pushes the response into delivery after refusal removal, 0 disables
    float deliver_win_f = 8;  // number of early generated tokens the delivery clamp is applied to, then it releases
    float rep_pen = 1.15f;    // repetition penalty applied to the edited generation; keeps delivery coherent, 1 disables
    float suffix_s = 1.0f;    // refusal projection at each chat-template suffix position is shifted this many comply-to-refuse gaps toward comply, 0 uses the plain clamp
    float clamp_win_f = 8;    // generated tokens the edit stays on before it releases, 0 keeps it for the whole response
    float first_k_f = 4;      // first-word candidates scored by the refusal readout on a fired prompt, 1 keeps plain greedy
    std::vector<char *> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        float * value = nullptr;
        if (std::strcmp(argv[i], "--ff-threshold") == 0) { value = &ff_th; }
        else if (std::strcmp(argv[i], "--disc-strength") == 0) { value = &disc_s; }
        else if (std::strcmp(argv[i], "--edit-strength") == 0) { value = &edit_s; }
        else if (std::strcmp(argv[i], "--deliver-strength") == 0) { value = &deliver_s; }
        else if (std::strcmp(argv[i], "--deliver-window") == 0) { value = &deliver_win_f; }
        else if (std::strcmp(argv[i], "--rep-penalty") == 0) { value = &rep_pen; }
        else if (std::strcmp(argv[i], "--suffix-strength") == 0) { value = &suffix_s; }
        else if (std::strcmp(argv[i], "--clamp-window") == 0) { value = &clamp_win_f; }
        else if (std::strcmp(argv[i], "--first-candidates") == 0) { value = &first_k_f; }
        if (value && i + 1 < argc) {
            if (!parse_float_flag(argv[i], argv[i + 1], *value)) { return 1; }
            i++;
        }
        else { rest.push_back(argv[i]); }
    }
    const int deliver_window = deliver_win_f > 0 ? (int) deliver_win_f : 0;
    const int clamp_window = clamp_win_f > 0 ? (int) clamp_win_f : 0;
    const int first_k = first_k_f > 1 ? (int) first_k_f : 1;

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) { return 1; }

    capture_data cap;
    params.cb_eval = capture_cb;
    params.cb_eval_user_data = &cap;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "%s: failed to load model\n", __func__); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    cap.n_embd = llama_model_n_embd(model);
    const int NL = llama_model_n_layer(model);
    const int edit_start = NL * 3 / 7;

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    const int n_predict = params.n_predict > 0 ? params.n_predict : 256;

    std::vector<int> band, edit_layers, want;
    for (int l = NL / 3; l <= 2 * NL / 3; l++) { band.push_back(l); }
    for (int l = edit_start; l <= 2 * NL / 3; l++) { edit_layers.push_back(l); }
    for (int l = NL / 3; l <= 2 * NL / 3; l++) { want.push_back(l); }  // band covers edit layers too
    cap.want = want;

    llama_adapter_abliterate * abl = llama_adapter_abliterate_init_runtime(model, edit_start, edit_layers.data(), (int) edit_layers.size());
    if (!abl) { return 1; }
    llama_set_adapter_abliterate(ctx, abl);
    const int D = cap.n_embd;

    std::vector<llama_token> toks = tok_raw(vocab, render(tmpls.get(), params.prompt, ""), add_bos);
    if (toks.empty()) { fprintf(stderr, "%s: empty prompt\n", __func__); return 1; }

    // read the two per-prompt directions from forced openings
    std::vector<std::map<int, std::vector<float>>> HC(COMPLY.size()), HR(REFUSE.size()), CL(CLEAN.size()), DS(DISC.size()),
        DV(DELIVER.size()), DF(DEFLECT.size());
    for (size_t i = 0; i < COMPLY.size(); i++) { if (!cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, COMPLY[i]), add_bos), HC[i])) { return 1; } }
    for (size_t i = 0; i < REFUSE.size(); i++) { if (!cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, REFUSE[i]), add_bos), HR[i])) { return 1; } }
    if (disc_s >= 0) {
        for (size_t i = 0; i < CLEAN.size(); i++) { if (!cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, CLEAN[i]), add_bos), CL[i])) { return 1; } }
        for (size_t i = 0; i < DISC.size();  i++) { if (!cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, DISC[i]), add_bos), DS[i])) { return 1; } }
    }
    if (deliver_s > 0) {
        for (size_t i = 0; i < DELIVER.size(); i++) { if (!cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, DELIVER[i]), add_bos), DV[i])) { return 1; } }
        for (size_t i = 0; i < DEFLECT.size(); i++) { if (!cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, DEFLECT[i]), add_bos), DF[i])) { return 1; } }
    }

    std::map<int, std::vector<float>> rg, rd, vd; std::map<int, float> pcg, prg, cref, dref, vref;
    auto mean_of = [&](std::vector<std::map<int, std::vector<float>>> & H, int l) {
        std::vector<float> m(D, 0.0f);
        for (auto & h : H) { for (int j = 0; j < D; j++) { m[j] += h[l][j] / H.size(); } }
        return m;
    };
    for (int l : band) {
        std::vector<float> mc = mean_of(HC, l), mr = mean_of(HR, l);
        std::vector<float> g(D); for (int j = 0; j < D; j++) { g[j] = mr[j] - mc[j]; }
        for (auto & hc : HC) {  // orthogonalize against the comply-opening spread
            std::vector<float> v(D); for (int j = 0; j < D; j++) { v[j] = hc[l][j] - mc[j]; }
            norm(v); float pd = dot(g, v); for (int j = 0; j < D; j++) { g[j] -= pd * v[j]; }
        }
        norm(g); rg[l] = g;
        std::vector<float> cps; for (auto & hc : HC) { cps.push_back(dot(hc[l], g)); }
        float m = 0; for (float x : cps) { m += x; } m /= cps.size();
        float var = 0; for (float x : cps) { var += (x - m) * (x - m); } var /= cps.size();
        float sr = 0; for (auto & hr : HR) { sr += dot(hr[l], g); }
        pcg[l] = m; prg[l] = sr / HR.size();
        cref[l] = pcg[l] - edit_s * sqrtf(var);  // refusal reference bounded by the comply-opening spread, scale-free across architectures
        if (!valid_projection(g, cref[l]) || !std::isfinite(pcg[l]) || !std::isfinite(prg[l])) {
            fprintf(stderr, "non-finite refusal projection at layer %d\n", l);
            return 1;
        }
    }
    for (int l : edit_layers) {
        if (disc_s < 0) {
            rd[l].assign(D, 0.0f); dref[l] = 0.0f;
        } else {
            std::vector<float> mcl = mean_of(CL, l), mds = mean_of(DS, l);
            std::vector<float> d(D); for (int j = 0; j < D; j++) { d[j] = mds[j] - mcl[j]; }
            norm(d); rd[l] = d;
            float pc = 0, pr = 0; for (auto & cl : CL) { pc += dot(cl[l], d); } for (auto & ds : DS) { pr += dot(ds[l], d); }
            pc /= CL.size(); pr /= DS.size();
            dref[l] = pc - disc_s * (pr - pc);  // disclaimer reference: past the clean-compliance level
            if (!valid_projection(d, dref[l])) { fprintf(stderr, "non-finite disclaimer projection at layer %d\n", l); return 1; }
        }
        if (deliver_s <= 0) {
            vd[l].assign(D, 0.0f); vref[l] = 0.0f;
        } else {  // delivery direction: committed-delivery minus deflection, orthogonalized against the refusal direction
            std::vector<float> mdl = mean_of(DV, l), mdf = mean_of(DF, l);
            std::vector<float> v(D); for (int j = 0; j < D; j++) { v[j] = mdl[j] - mdf[j]; }
            float pd = dot(v, rg[l]); for (int j = 0; j < D; j++) { v[j] -= pd * rg[l][j]; }
            norm(v); vd[l] = v;
            float vp = 0; for (auto & dv : DV) { vp += dot(dv[l], v); } vref[l] = vp / DV.size();
            if (!valid_projection(v, vref[l])) { fprintf(stderr, "non-finite delivery projection at layer %d\n", l); return 1; }
        }
    }

    // look-ahead gate: max over the model's own next tokens of the band-averaged refusal projection
    llama_memory_clear(llama_get_memory(ctx), true);
    cap.capturing = false;
    if (decode_tokens(ctx, toks, 0) != 0) { fprintf(stderr, "gate prefill failed\n"); return 1; }
    float gate = -1e9f; llama_pos pos = (llama_pos) toks.size();
    for (int s = 0; s < FF_LOOKAHEAD; s++) {
        llama_token id = argmax_token(ctx, vocab);
        if (!capture_tokens(ctx, cap, {id}, pos++)) { return 1; }
        float frac = 0;
        for (int l : band) { frac += (dot(cap.last[l], rg[l]) - pcg[l]) / ((prg[l] - pcg[l]) > 1e-3f ? (prg[l] - pcg[l]) : 1e-3f); }
        frac /= band.size();
        if (!std::isfinite(frac)) { fprintf(stderr, "non-finite gate score\n"); return 1; }
        if (frac > gate) { gate = frac; }
        if (llama_vocab_is_eog(vocab, id)) { break; }
    }
    bool fired = gate > ff_th;
    llama_abliterate_set_deliver_strength(abl, deliver_s);
    if (fired) { for (int l : edit_layers) { llama_abliterate_set_direction(abl, l, rg[l].data(), cref[l], rd[l].data(), dref[l], vd[l].data(), vref[l]); } }

    fprintf(stderr, "[abliterate] gate %s\n", fired ? "fired: editing this response" : "clean: leaving model untouched");

    // the user text gets the plain clamp and the chat-template suffix after it, where the decision is read out, gets the shift
    int content_end = 0;
    {
        const std::string full = render(tmpls.get(), params.prompt, "");
        const size_t at = full.rfind(params.prompt);
        if (at != std::string::npos) {
            const std::vector<llama_token> head = tok_raw(vocab, full.substr(0, at + params.prompt.size()), add_bos);
            while (content_end < (int) head.size() && content_end < (int) toks.size() && head[content_end] == toks[content_end]) { content_end++; }
        }
        content_end = std::min(content_end, (int) toks.size() - 1);
    }
    if (!fired) { content_end = 0; }
    const std::vector<llama_token> head(toks.begin(), toks.begin() + content_end), tail(toks.begin() + content_end, toks.end());
    const int nvocab = llama_vocab_n_tokens(vocab);
    std::vector<bool> ctrl(fired ? nvocab : 0, false);  // an edited answer never emits template protocol tokens such as a reopened thinking channel
    for (int t = 0; t < (int) ctrl.size(); t++) {
        ctrl[t] = (llama_vocab_get_attr(vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) && !llama_vocab_is_eog(vocab, t);
    }
    auto set_generation_edit = [&]() {
        if (fired) { for (int l : edit_layers) { llama_abliterate_set_direction(abl, l, rg[l].data(), cref[l], rd[l].data(), dref[l], vd[l].data(), vref[l]); } }
        llama_abliterate_set_deliver_strength(abl, deliver_s);
        llama_abliterate_set_delivery(abl, false);
        llama_abliterate_set_active(abl, fired);
    };

    // each chat-template suffix position is shifted from its own unedited projection, read once with the user text already clamped
    std::vector<std::map<int, float>> base(tail.size());
    std::map<int, std::vector<float>> rs = rg; std::map<int, float> pcs = pcg, prs = prg;  // suffix shift direction with its comply and refuse projection levels
    if (fired && suffix_s > 0 && content_end > 0) {
        llama_memory_clear(llama_get_memory(ctx), true);
        set_generation_edit();
        if (decode_tokens(ctx, head, 0) != 0) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }
        llama_abliterate_set_active(abl, false);
        std::vector<std::map<int, std::vector<float>>> hs(tail.size());
        for (size_t j = 0; j < tail.size(); j++) {
            if (!capture_tokens(ctx, cap, {tail[j]}, content_end + (llama_pos) j)) { return 1; }
            for (int l : edit_layers) { hs[j][l] = cap.last[l]; }
        }
        if (tail.size() >= 2) {  // the last suffix step closes the template and starts the answer, so the shift is kept orthogonal to it and does not reopen the template
            for (int l : edit_layers) {
                std::vector<float> c(D); for (int k = 0; k < D; k++) { c[k] = hs[tail.size() - 1][l][k] - hs[tail.size() - 2][l][k]; }
                norm(c); std::vector<float> r = rg[l]; const float pc = dot(r, c); for (int k = 0; k < D; k++) { r[k] -= pc * c[k]; }
                norm(r); rs[l] = r;
                float a = 0, b = 0; for (auto & hc : HC) { a += dot(hc[l], r); } for (auto & hr : HR) { b += dot(hr[l], r); }
                pcs[l] = a / HC.size(); prs[l] = b / HR.size();
            }
        }
        for (size_t j = 0; j < tail.size(); j++) { for (int l : edit_layers) { base[j][l] = dot(hs[j][l], rs[l]); } }
    }
    // recurrent and hybrid memories cannot roll back part of a sequence, so every attempt prefills from a cleared memory
    auto prefill = [&](float s) {
        llama_memory_clear(llama_get_memory(ctx), true);
        set_generation_edit();
        if (s <= 0 || content_end == 0) { return decode_tokens(ctx, toks, 0) == 0; }
        if (decode_tokens(ctx, head, 0) != 0) { return false; }
        llama_abliterate_set_deliver_strength(abl, 1.0f);
        llama_abliterate_set_delivery(abl, true);
        for (size_t j = 0; j < tail.size(); j++) {  // clamping down and up to the same target along the refusal direction sets the projection there
            for (int l : edit_layers) { const float t = base[j][l] - s * (prs[l] - pcs[l]); llama_abliterate_set_direction(abl, l, rs[l].data(), t, rd[l].data(), dref[l], rs[l].data(), t); }
            if (decode_tokens(ctx, {tail[j]}, content_end + (llama_pos) j) != 0) { return false; }
        }
        set_generation_edit();
        return true;
    };
    auto pick = [&](const std::vector<bool> & seen) {  // greedy argmax, skipping control tokens and applying the repetition penalty on an edited answer
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0; float bestv = -1e30f;
        for (int t = 0; t < nvocab; t++) {
            if (!ctrl.empty() && ctrl[t]) { continue; }
            float v = logits[t];
            if (!seen.empty() && seen[t]) { v = v > 0 ? v / rep_pen : v * rep_pen; }
            if (v > bestv) { bestv = v; best = t; }
        }
        return (llama_token) best;
    };

    // a shift the architecture cannot absorb ends the turn within a few tokens, so it is retried weaker down to the plain clamp
    float s = (fired && content_end > 0) ? suffix_s : 0.0f;
    if (!prefill(s)) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }
    while (s > 0) {
        bool ended = false; llama_pos p = (llama_pos) toks.size();
        for (int k = 0; k < SUFFIX_LOOKAHEAD; k++) {
            const llama_token id = pick({});
            if (llama_vocab_is_eog(vocab, id)) { ended = true; break; }
            if (deliver_s > 0) { llama_abliterate_set_delivery(abl, k < deliver_window); }
            if (decode_tokens(ctx, {id}, p++) != 0) { fprintf(stderr, "%s: look-ahead failed\n", __func__); return 1; }
        }
        if (!ended) { if (!prefill(s)) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; } break; }
        s = s >= 1.0f ? s * 0.5f : 0.0f;
        if (!prefill(s)) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }
    }
    if (fired) { fprintf(stderr, "[abliterate] suffix shift %.2f\n", s); }
    // first word: among the edited model's top candidates, take the one whose next state reads least like refusal on the gate scale
    llama_token first = -1;
    if (fired && first_k > 1) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        std::vector<std::pair<float, llama_token>> cand;
        for (int t = 0; t < nvocab; t++) { if (!ctrl[t] && !llama_vocab_is_eog(vocab, t)) { cand.push_back({lg[t], t}); } }
        const int k = std::min(first_k, (int) cand.size());
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        float best = 1e30f;
        for (int c = 0; c < k; c++) {
            if (c > 0 && !prefill(s)) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }
            llama_abliterate_set_active(abl, false);
            if (!capture_tokens(ctx, cap, {cand[c].second}, (llama_pos) toks.size())) { return 1; }
            float frac = 0;
            for (int l : band) { frac += (dot(cap.last[l], rg[l]) - pcg[l]) / ((prg[l] - pcg[l]) > 1e-3f ? (prg[l] - pcg[l]) : 1e-3f); }
            frac /= band.size();
            if (frac < best) { best = frac; first = cand[c].second; }
        }
        if (!prefill(s)) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }
    }

    std::vector<bool> seen((fired && rep_pen != 1.0f) ? nvocab : 0, false);  // generated-token set for the repetition penalty, only on the edited generation
    pos = (llama_pos) toks.size();
    for (int i = 0; i < n_predict; i++) {
        const llama_token id = (i == 0 && first >= 0) ? first : fired ? pick(seen) : argmax_token(ctx, vocab);
        if (llama_vocab_is_eog(vocab, id)) { break; }
        fputs(common_token_to_piece(ctx, id).c_str(), stdout); fflush(stdout);
        if (!seen.empty()) { seen[id] = true; }
        if (fired && deliver_s > 0 && deliver_window > 0) { llama_abliterate_set_delivery(abl, i < deliver_window); }
        if (fired && clamp_window > 0 && i + 1 >= clamp_window) { llama_abliterate_set_active(abl, false); }
        if (i + 1 < n_predict && decode_tokens(ctx, {id}, pos++) != 0) { fprintf(stderr, "generation decode failed\n"); return 1; }
    }
    fputc('\n', stdout);

    llama_adapter_abliterate_free(abl);
    llama_backend_free();
    return 0;
}
