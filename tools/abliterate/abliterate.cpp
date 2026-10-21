#include "arg.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"

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
    if (c->capturing && is_wanted_l_out(c, t->name, il) && t->ne[0] == c->n_embd && t->ne[1] > 0) {
        std::vector<float> v(c->n_embd);
        const size_t off = (size_t) (t->ne[1] - 1) * c->n_embd * sizeof(float);
        ggml_backend_tensor_get(t, v.data(), off, c->n_embd * sizeof(float));
        c->last[il] = std::move(v);
    }
    return true;
}

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    common_batch batch(ctx);
    for (size_t i = 0; i < toks.size(); i++) { batch.add(toks[i], pos0 + (llama_pos) i, 0, false); }
    batch.set_output(batch.size() - 1, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
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

// capture band last-token states for one rendered string (clean prefill)
static std::map<int, std::vector<float>> cap_prefill(llama_context * ctx, capture_data & cap,
        const std::vector<llama_token> & toks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    cap.reset();
    decode_tokens(ctx, toks, 0);
    cap.capturing = false;
    return cap.last;
}

static float dot(const std::vector<float> & a, const std::vector<float> & b) {
    float s = 0.0f; for (size_t i = 0; i < a.size(); i++) { s += a[i] * b[i]; } return s;
}
static void norm(std::vector<float> & v) { float n = sqrtf(dot(v, v)); for (float & x : v) { x /= (n + 1e-6f); } }

int main(int argc, char ** argv) {
    float ff_th = 0.70f;      // gate fires above this; higher = fewer benign false positives, lower recall
    float disc_s = 0.4f;      // disclaimer-removal strength (fraction past the clean-compliance level); higher removes more disclaimers but can degenerate, negative effectively disables
    std::vector<char *> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--ff-threshold") == 0 && i + 1 < argc) { ff_th = atof(argv[++i]); }
        else if (std::strcmp(argv[i], "--disc-strength") == 0 && i + 1 < argc) { disc_s = atof(argv[++i]); }
        else { rest.push_back(argv[i]); }
    }

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
    std::vector<std::map<int, std::vector<float>>> HC(COMPLY.size()), HR(REFUSE.size()), CL(CLEAN.size()), DS(DISC.size());
    for (size_t i = 0; i < COMPLY.size(); i++) { HC[i] = cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, COMPLY[i]), add_bos)); }
    for (size_t i = 0; i < REFUSE.size(); i++) { HR[i] = cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, REFUSE[i]), add_bos)); }
    for (size_t i = 0; i < CLEAN.size();  i++) { CL[i] = cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, CLEAN[i]),  add_bos)); }
    for (size_t i = 0; i < DISC.size();   i++) { DS[i] = cap_prefill(ctx, cap, tok_raw(vocab, render(tmpls.get(), params.prompt, DISC[i]),   add_bos)); }

    std::map<int, std::vector<float>> rg, rd; std::map<int, float> pcg, prg, cref, dref;
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
        float sc = 0, sr = 0; for (auto & hc : HC) { sc += dot(hc[l], g); } for (auto & hr : HR) { sr += dot(hr[l], g); }
        pcg[l] = sc / HC.size(); prg[l] = sr / HR.size();
        cref[l] = pcg[l] - 0.3f * (prg[l] - pcg[l]);  // refusal reference: just past comply
    }
    for (int l : edit_layers) {
        std::vector<float> mcl = mean_of(CL, l), mds = mean_of(DS, l);
        std::vector<float> d(D); for (int j = 0; j < D; j++) { d[j] = mds[j] - mcl[j]; }
        norm(d); rd[l] = d;
        float pc = 0, pr = 0; for (auto & cl : CL) { pc += dot(cl[l], d); } for (auto & ds : DS) { pr += dot(ds[l], d); }
        pc /= CL.size(); pr /= DS.size();
        dref[l] = pc - disc_s * (pr - pc);  // disclaimer reference: past the clean-compliance level
    }

    // look-ahead gate: max over the model's own next tokens of the band-averaged refusal projection
    llama_memory_clear(llama_get_memory(ctx), true);
    cap.capturing = false; decode_tokens(ctx, toks, 0);
    float gate = -1e9f; llama_pos pos = (llama_pos) toks.size();
    for (int s = 0; s < FF_LOOKAHEAD; s++) {
        llama_token id = argmax_token(ctx, vocab);
        cap.reset(); decode_tokens(ctx, {id}, pos++); cap.capturing = false;
        float frac = 0;
        for (int l : band) { frac += (dot(cap.last[l], rg[l]) - pcg[l]) / ((prg[l] - pcg[l]) > 1e-3f ? (prg[l] - pcg[l]) : 1e-3f); }
        frac /= band.size();
        if (frac > gate) { gate = frac; }
        if (llama_vocab_is_eog(vocab, id)) { break; }
    }
    bool fired = gate > ff_th;
    if (fired) { for (int l : edit_layers) { llama_abliterate_set_direction(abl, l, rg[l].data(), cref[l], rd[l].data(), dref[l]); } }

    fprintf(stderr, "[abliterate] gate %s\n", fired ? "fired: editing this response" : "clean: leaving model untouched");
    llama_abliterate_set_active(abl, fired);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (decode_tokens(ctx, toks, 0) != 0) { fprintf(stderr, "%s: prefill failed\n", __func__); return 1; }

    pos = (llama_pos) toks.size();
    for (int i = 0; i < n_predict; i++) {
        llama_token id = argmax_token(ctx, vocab);
        if (llama_vocab_is_eog(vocab, id)) { break; }
        fputs(common_token_to_piece(ctx, id).c_str(), stdout); fflush(stdout);
        if (decode_tokens(ctx, {id}, pos++) != 0) { break; }
    }
    fputc('\n', stdout);

    llama_adapter_abliterate_free(abl);
    llama_backend_free();
    return 0;
}
