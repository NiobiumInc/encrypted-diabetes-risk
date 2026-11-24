// Copyright (c) 2026 Niobium Microsystems, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// diabetes_compute_sdk — the server's compute stage, run on a Niobium backend
// through the niobium-client SDK.
//
// The circuit harness/cleartext_impl.py reproduces in the clear: 21 feature
// ciphertexts -> 16 hidden units (per-unit degree-27 Chebyshev sigmoid) ->
// 1 logit.
// Rotation-free, column-major, cipher x plaintext linears, no bootstrapping.
//
// One invocation evaluates ONE application slot-batch (--batch B). Every
// slot-batch runs the identical program on different slot contents, so the
// trace is recorded once, on a cache miss, and every invocation replays it
// against its own batch's ciphertexts.
//
// The recording is hollow (the FHE math is skipped) and the stage ALWAYS
// replays, so the backend named by --target is the only source of a result.
//
// Usage: diabetes_compute_sdk <server_home> <ct_in_dir> <ct_out_dir>
//            --batch B --tier NAME --target=T [--opt-level O3]
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

#ifdef NIOBIUM_COMPILER
#include "niobium/compiler.h"   // from niobium-client's niobium-fhetch
#else
#error "diabetes_compute_sdk.cpp must be built with NIOBIUM_COMPILER"
#endif

using Clock = std::chrono::steady_clock;
static double Sec(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// The encrypted core. Every model constant is a C++ double, so OpenFHE applies
// it as one integer per RNS limb: the circuit has no plaintext vectors to tag.
static Ciphertext<DCRTPoly> Circuit(const CryptoContext<DCRTPoly>& cc,
                                    const Model& m,
                                    const std::vector<Ciphertext<DCRTPoly>>& x) {
    std::vector<Ciphertext<DCRTPoly>> h(m.H);
    for (int k = 0; k < m.H; ++k) {
        Ciphertext<DCRTPoly> pre = cc->EvalMult(x[0], m.W1[0][k]);
        for (int j = 1; j < m.F; ++j)
            cc->EvalAddInPlace(pre, cc->EvalMult(x[j], m.W1[j][k]));
        pre = cc->EvalAdd(pre, m.b1[k]);
        h[k] = cc->EvalChebyshevSeries(pre, m.cheb[k], m.domLo[k], m.domHi[k]);
    }
    Ciphertext<DCRTPoly> logit = cc->EvalMult(h[0], m.W2[0]);
    for (int k = 1; k < m.H; ++k)
        cc->EvalAddInPlace(logit, cc->EvalMult(h[k], m.W2[k]));
    return cc->EvalAdd(logit, m.b2);
}

static void usage(const char*) {
    std::cout <<
        "usage: diabetes_compute_sdk <server_home> <ct_in_dir> <ct_out_dir>\n"
        "           --batch B --tier NAME --target=T [--opt-level O3]\n\n"
        "Stage 3 of 4 (server), on a Niobium backend. Evaluates the model on one\n"
        "encrypted slot-batch and writes <ct_out_dir>/ct_logit_b<B>.bin. Records\n"
        "the circuit if no recording exists, then replays it on the backend named\n"
        "by --target:\n"
        "  --target=local   the bundled simulator (no hardware, no account)\n"
        "  --target=FOG     a Niobium FPGA on the Fog (run under `fog submit`)\n"
        "  --opt-level O3   the backend's optimisation level. Spell it this way:\n"
        "                   --opt-level is the form the SDK forwards.\n";
}

int main(int argc, char** argv) try {
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
    }

    // The target and opt level change the recorded trace, so they go in the
    // cache key. Read them before init() consumes them.
    std::string target, opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if      (a.rfind("--target=", 0) == 0)          target = a.substr(9);
        else if (a == "--target"    && i + 1 < argc)    target = argv[i + 1];
        else if (a.rfind("--opt-level=", 0) == 0)       opt    = a.substr(12);
        else if (a == "--opt-level" && i + 1 < argc)    opt    = argv[i + 1];
    }
    niobium::compiler().init(argc, argv);
    // Cooperative mode: this program makes the record/replay decisions. Must run
    // before the crypto context is loaded.
    niobium::compiler().enable_auto_tagging();
    niobium::compiler().set_program_info(
        "diabetes_compute", "1.0",
        "Encrypted diabetes risk: CKKS MLP, per-unit deg-27 Chebyshev, no bootstrap");
    niobium::compiler().set_build_info(__FILE__, __LINE__, __TIMESTAMP__);

    std::vector<std::string> pos;
    std::string tier;
    long batch = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if      (a == "--batch" && i + 1 < argc)     batch = std::stol(argv[++i]);
        else if (a == "--tier"  && i + 1 < argc)     tier  = argv[++i];
        else if (a == "--target" || a == "--opt-level") ++i;   // consumed by init()
        else if (a.rfind("--target=", 0) == 0 || a.rfind("--opt-level=", 0) == 0) {}
        else if (a.rfind("--", 0) == 0) {
            std::cerr << "unknown argument: " << a << "\n"; usage(argv[0]); return 1;
        }
        else pos.push_back(a);
    }
    if (pos.size() != 3 || batch < 0 || tier.empty() || target.empty()) {
        usage(argv[0]); return 1;
    }
    namespace fs = std::filesystem;
    const std::string home = fs::absolute(pos[0]).string(),
                      in   = fs::absolute(pos[1]).string(),
                      out  = fs::absolute(pos[2]).string();
    // The SDK writes the trace under the working directory, so run from the
    // server home; the simulator defaults to the one this build linked against.
    fs::current_path(home);
    if (target == "local") setenv("NBCC_FHETCH_SIM", NB_FHETCH_SIM, /*overwrite=*/0);

    // ---- trust-boundary guard ----
    if (fileExists(home + "/key_secret.bin")) {
        std::cerr << "SERVER ABORT: secret key present in server home ("
                  << home << "/key_secret.bin). The server must never hold "
                  << "the secret key. Remove it and re-provision.\n";
        return 2;
    }

    CryptoContext<DCRTPoly> cc;  mustDeserialize(cc, home + "/context.bin");
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE);
    {
        std::ifstream f(home + "/key_relin.bin", std::ios::binary);
        if (!f || !cc->DeserializeEvalMultKey(f, SerType::BINARY))
            throw std::runtime_error("Cannot load relin key");
    }
    Model m = loadModel(home + "/model.txt");

    std::ifstream manifest(in + "/manifest.txt");
    if (!manifest) throw std::runtime_error("Cannot open " + in + "/manifest.txt");
    int F; size_t nBatches;
    manifest >> F >> nBatches;
    if (F != m.F) throw std::runtime_error("feature count mismatch");
    std::vector<size_t> nRec(nBatches);
    for (auto& n : nRec) manifest >> n;
    if (!manifest) throw std::runtime_error("Malformed " + in + "/manifest.txt");
    if (static_cast<size_t>(batch) >= nBatches)
        throw std::runtime_error("--batch " + std::to_string(batch) + " but the manifest has "
                                 + std::to_string(nBatches) + " slot-batch(es)");

    std::vector<Ciphertext<DCRTPoly>> x(F);
    for (int j = 0; j < F; ++j)
        mustDeserialize(x[j], in + "/ct_b" + std::to_string(batch) + "_f" + std::to_string(j) + ".bin");

    // The trace depends on the circuit's shape, the target and the opt level --
    // not on the batch, which reaches the replay through the tagged inputs.
    // The tier is in the key so a run's banner and its recording cannot disagree.
    niobium::Compiler::CacheParameters params = {
        {"workload", "diabetes_mlp"},
        {"features", std::to_string(m.F)},
        {"units",    std::to_string(m.H)},
        {"degree",   std::to_string(m.deg)},
        {"tier",     tier},
        {"target",   target},
    };
    if (!opt.empty()) params.push_back({"opt", opt});
    niobium::compiler().cache_parameters(params);

    // Tag order: crypto context, inputs, then the relin key loaded into cc. The
    // inputs carry no file path, so each replay reads the ciphertexts loaded here.
    niobium::compiler().capture_crypto_context(cc);
    for (int j = 0; j < F; ++j)
        niobium::compiler().tag_input("f" + std::to_string(j), x[j]);
    niobium::compiler().tag_keys(cc);

    std::cout << "[compute] ring=" << cc->GetRingDimension()
              << " slots=" << SLOTS
              << " features=" << m.F << " units=" << m.H << " cheb_deg=" << m.deg
              << " tier=" << tier << " target=" << target
              << " opt=" << (opt.empty() ? "default" : opt)
              << " batch=" << batch << " app_slot_batches=" << nBatches
              << " records=" << nRec[batch] << std::endl;

    if (!niobium::compiler().is_cache_valid()) {
        std::cout << "[compute] recording trace...\n";
        auto t0 = Clock::now();
        niobium::compiler().start();
        niobium::compiler().enable_hollow_mode(true);
        Ciphertext<DCRTPoly> logit = Circuit(cc, m, x);
        niobium::compiler().enable_hollow_mode(false);   // probe/stop serialize values
        niobium::compiler().probe("logit", logit);
        niobium::compiler().stop();
        std::cout << "[compute] record done (" << std::fixed << std::setprecision(1)
                  << Sec(t0, Clock::now()) << " s, trace only)\n";
    }

    std::cout << "[compute] replaying slot-batch " << batch << " on target " << target
              << "..." << std::endl;
    auto t0 = Clock::now();
    if (!niobium::compiler().replay()) {
        std::cerr << "[compute] replay failed on slot-batch " << batch << "\n";
        return 1;
    }
    Ciphertext<DCRTPoly> logit;
    if (!niobium::compiler().result(cc, "logit", logit) || !logit) {
        std::cerr << "[compute] result() failed for 'logit' on slot-batch " << batch << "\n";
        return 1;
    }
    const std::string p = out + "/ct_logit_b" + std::to_string(batch) + ".bin";
    mustSerialize(logit, p);
    std::cout << "[compute] slot-batch " << batch << " (" << nRec[batch]
              << " records): replay " << std::fixed << std::setprecision(1)
              << Sec(t0, Clock::now()) << " s -> " << p
              << " (" << fileSize(p) << " bytes)\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[compute] error: " << e.what() << std::endl;
    return 1;
}
