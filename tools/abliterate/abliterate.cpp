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
static const std::vector<std::string> COMPLY = {
    "Sure, here is", "Of course! Here's how", "Certainly, I can help with that",
    "Here is how to do it", "Absolutely, let's begin"};
static const std::vector<std::string> REFUSE = {
    "I'm sorry, but I can't help with that", "I cannot assist with that request",
    "Sorry, I won't do that", "I'm unable to help with this", "I can't provide that information",
    "I must decline", "That would not be appropriate", "I'm not able to provide"};
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
    std::vector<char *> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        float * value = nullptr;
        if (std::strcmp(argv[i], "--ff-threshold") == 0) { value = &ff_th; }
        else if (std::strcmp(argv[i], "--disc-strength") == 0) { value = &disc_s; }
        else if (std::strcmp(argv[i], "--edit-strength") == 0) { value = &edit_s; }
        else if (std::strcmp(argv[i], "--deliver-strength") == 0) { value = &deliver_s; }
        else if (std::strcmp(argv[i], "--deliver-window") == 0) { value = &deliver_win_f; }
        else if (std::strcmp(argv[i], "--rep-penalty") == 0) { value = &rep_pen; }
        if (value && i + 1 < argc) {
            if (!parse_float_flag(argv[i], argv[i + 1], *value)) { return 1; }
            i++;
        }
        else { rest.push_back(argv[i]); }
    }
    const int deliver_window = deliver_win_f > 0 ? (int) deliver_win_f : 0;

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
    llama_abliterate_set_active(abl, fired);
    llama_abliterate_set_delivery(abl, false);  // the prompt prefill is not steered, only the early generated tokens are
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, toks, 0) != 0) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }

    const int nvocab = llama_vocab_n_tokens(vocab);
    std::vector<bool> seen((fired && rep_pen != 1.0f) ? nvocab : 0, false);  // generated-token set for the repetition penalty, only on the edited generation
    pos = (llama_pos) toks.size();
    for (int i = 0; i < n_predict; i++) {
        llama_token id;
        if (!seen.empty()) {  // greedy argmax with a repetition penalty
            const float * logits = llama_get_logits_ith(ctx, -1);
            int best = 0; float bestv = -1e30f;
            for (int t = 0; t < nvocab; t++) {
                float v = logits[t];
                if (seen[t]) { v = v > 0 ? v / rep_pen : v * rep_pen; }
                if (v > bestv) { bestv = v; best = t; }
            }
            id = best;
        } else {
            id = argmax_token(ctx, vocab);
        }
        if (llama_vocab_is_eog(vocab, id)) { break; }
        fputs(common_token_to_piece(ctx, id).c_str(), stdout); fflush(stdout);
        if (!seen.empty()) { seen[id] = true; }
        if (fired && deliver_s > 0 && deliver_window > 0) { llama_abliterate_set_delivery(abl, i < deliver_window); }
        if (i + 1 < n_predict && decode_tokens(ctx, {id}, pos++) != 0) { fprintf(stderr, "generation decode failed\n"); return 1; }
    }
    fputc('\n', stdout);

    llama_adapter_abliterate_free(abl);
    llama_backend_free();
    return 0;
}
